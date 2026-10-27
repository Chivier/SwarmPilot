//! OpenAI Responses, stateless subset
//! (`swarmpilot/api/styles/openai_responses.py`).

use serde_json::{json, Map, Value};

use super::base::{
    as_list, as_mapping, as_str, first_truthy, obj, opt_float, opt_int, text_of, unsupported,
    Context, Env, TrResult,
};
use super::canonical::{
    Event, Message, Part, Request, Response, ResponseFormat, Role, StopReason, Tool, ToolChoice,
    Usage,
};
use super::openai_chat::openai_error;
use super::py::{self, PyResult};
use super::sse::frame;
use super::style::{Decoder, Encoder, Flow, Style};

const KNOWN: [&str; 15] = [
    "model",
    "input",
    "instructions",
    "max_output_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "text",
    "reasoning",
    "stream",
    "user",
    "store",
    "metadata",
];
const STATEFUL: [&str; 3] = ["previous_response_id", "conversation", "background"];

fn content(content: &Value, where_: &str) -> TrResult<Vec<Part>> {
    if let Value::String(s) = content {
        return Ok(vec![Part::Text(s.clone())]);
    }
    let mut parts = Vec::new();
    for (i, p) in as_list(content, where_)?.iter().enumerate() {
        let w = format!("{where_}[{i}]");
        let p = as_mapping(p, &w)?;
        let kind = py::get(p, "type");
        match kind.as_str() {
            Some("input_text" | "output_text" | "text") => parts.push(Part::Text(
                as_str(py::get(p, "text"), &format!("{w}.text"))?.to_string(),
            )),
            Some("refusal") => {
                parts.push(Part::Text(py::str_of(py::or(p.get("refusal"), &json!("")))))
            }
            Some("input_image") => {
                let Value::String(url) = py::get(p, "image_url") else {
                    return Err(unsupported(
                        &format!("{w}.file_id"),
                        "only image_url images",
                    ));
                };
                parts.push(Part::Image {
                    url: url.clone(),
                    detail: py::get(p, "detail").clone(),
                });
            }
            _ => {
                return Err(unsupported(
                    &format!("{w}.type"),
                    &format!("content type {}", py::repr(kind)),
                ))
            }
        }
    }
    Ok(parts)
}

fn last_assistant(req: &mut Request) -> &mut Message {
    if !matches!(req.messages.last(), Some(m) if m.role == Role::Assistant) {
        req.messages.push(Message::new(Role::Assistant, Vec::new()));
    }
    req.messages.last_mut().expect("just pushed")
}

/// `"".join(s.get("text", "") for s in items if <filter>)`.
fn join_texts(items: &Value, mappings_only: bool) -> PyResult<String> {
    let blank = json!("");
    let mut texts = Vec::new();
    for s in py::iter_or_empty(items)? {
        match &s {
            Value::Object(m) => texts.push(py::get_or(m, "text", &blank).clone()),
            _ if mappings_only => {}
            _ => py::attr_get(&s, "text").map(|_| ())?,
        }
    }
    py::join(&texts)
}

/// `dict | other`: keys of `other` replace or follow those of `base`.
fn merged(mut base: Value, other: Value) -> Value {
    if let (Value::Object(b), Value::Object(o)) = (&mut base, other) {
        for (k, v) in o {
            b.insert(k, v);
        }
    }
    base
}

fn response_obj(
    rid: &str,
    model: &Value,
    created: i128,
    output: Vec<Value>,
    req: &Request,
    env: &mut dyn Env,
) -> Value {
    let rid = if rid.starts_with("resp_") {
        rid.to_string()
    } else {
        format!("resp_{rid}")
    };
    let created = if created != 0 { created } else { env.now() };
    let instructions = req.system_text();
    obj([
        ("id", json!(rid)),
        ("object", json!("response")),
        ("created_at", py::int_value(created)),
        ("model", first_truthy(&[model])),
        ("output", Value::Array(output)),
        ("error", Value::Null),
        ("incomplete_details", Value::Null),
        (
            "instructions",
            if instructions.is_empty() {
                Value::Null
            } else {
                json!(instructions)
            },
        ),
        (
            "max_output_tokens",
            req.max_output_tokens.map_or(Value::Null, py::int_value),
        ),
        (
            "parallel_tool_calls",
            json!(req.parallel_tool_calls != Value::Bool(false)),
        ),
        (
            "temperature",
            req.temperature.map_or(Value::Null, py::float_value),
        ),
        ("top_p", req.top_p.map_or(Value::Null, py::float_value)),
        ("tool_choice", json!("auto")),
        ("tools", json!([])),
        ("text", json!({"format": {"type": "text"}})),
        ("store", json!(false)),
        ("metadata", json!({})),
    ])
}

fn status(stop: StopReason, usage: &Usage) -> PyResult<Value> {
    let mut out = Map::new();
    out.insert("status".into(), json!("completed"));
    out.insert("usage".into(), usage_out(usage)?);
    let reason = match stop {
        StopReason::MaxTokens => Some("max_output_tokens"),
        StopReason::ContentFilter => Some("content_filter"),
        _ => None,
    };
    if let Some(r) = reason {
        out.insert("status".into(), json!("incomplete"));
        out.insert("incomplete_details".into(), json!({"reason": r}));
    }
    Ok(Value::Object(out))
}

fn stop_in(body: &Map<String, Value>, has_call: bool, refused: bool) -> PyResult<StopReason> {
    let empty = json!({});
    let details = py::or(body.get("incomplete_details"), &empty);
    if py::get(body, "status").as_str() == Some("incomplete") {
        if py::attr_get(details, "reason")?.as_str() == Some("max_output_tokens") {
            return Ok(StopReason::MaxTokens);
        }
        return Ok(StopReason::ContentFilter);
    }
    if refused {
        return Ok(StopReason::ContentFilter);
    }
    if has_call {
        return Ok(StopReason::ToolUse);
    }
    Ok(StopReason::EndTurn)
}

fn usage_in(u: &Value) -> PyResult<Usage> {
    let Value::Object(u) = u else {
        return Ok(Usage::default());
    };
    let empty = json!({});
    let idt = py::or(u.get("input_tokens_details"), &empty);
    let odt = py::or(u.get("output_tokens_details"), &empty);
    Ok(Usage {
        input_tokens: py::get(u, "input_tokens").clone(),
        output_tokens: py::get(u, "output_tokens").clone(),
        cached_input_tokens: if py::truthy(idt) {
            py::attr_get(idt, "cached_tokens")?.clone()
        } else {
            Value::Null
        },
        reasoning_tokens: if py::truthy(odt) {
            py::attr_get(odt, "reasoning_tokens")?.clone()
        } else {
            Value::Null
        },
    })
}

fn usage_out(u: &Usage) -> PyResult<Value> {
    let zero = json!(0);
    let i = py::or(Some(&u.input_tokens), &zero);
    let o = py::or(Some(&u.output_tokens), &zero);
    Ok(json!({
        "input_tokens": i,
        "input_tokens_details": {"cached_tokens": py::or(Some(&u.cached_input_tokens), &zero)},
        "output_tokens": o,
        "output_tokens_details": {"reasoning_tokens": py::or(Some(&u.reasoning_tokens), &zero)},
        "total_tokens": py::add(i, o)?,
    }))
}

fn item(p: &Part, env: &mut dyn Env) -> Value {
    match p {
        Part::ToolCall {
            id,
            name,
            arguments,
        } => obj([
            ("type", json!("function_call")),
            ("id", json!(env.new_id("fc_"))),
            ("call_id", json!(id)),
            ("name", json!(name)),
            ("arguments", arguments.clone()),
            ("status", json!("completed")),
        ]),
        Part::Reasoning { text, .. } => obj([
            ("type", json!("reasoning")),
            ("id", json!(env.new_id("rs_"))),
            ("summary", json!([{"type": "summary_text", "text": text}])),
        ]),
        _ => {
            let text = match p {
                Part::Text(t) => t.as_str(),
                _ => "",
            };
            obj([
                ("type", json!("message")),
                ("id", json!(env.new_id("msg_"))),
                ("status", json!("completed")),
                ("role", json!("assistant")),
                (
                    "content",
                    json!([{"type": "output_text", "text": text, "annotations": []}]),
                ),
            ])
        }
    }
}

pub struct OpenAiResponses;

impl OpenAiResponses {
    fn decode_item(req: &mut Request, it: &Map<String, Value>, w: &str) -> TrResult<()> {
        let message = json!("message");
        let kind = py::get_or(it, "type", &message);
        match kind.as_str() {
            Some("message") => {
                let role = py::get(it, "role");
                match role.as_str() {
                    Some("system" | "developer") => {
                        req.system.push(text_of(py::get(it, "content")))
                    }
                    Some(r @ ("user" | "assistant")) => {
                        let parts = content(py::get(it, "content"), &format!("{w}.content"))?;
                        if r == "assistant" {
                            last_assistant(req).content.extend(parts);
                        } else {
                            req.messages.push(Message::new(Role::User, parts));
                        }
                    }
                    _ => {
                        return Err(unsupported(
                            &format!("{w}.role"),
                            &format!("role {}", py::repr(role)),
                        ))
                    }
                }
            }
            Some("function_call") => {
                let id = as_str(py::get(it, "call_id"), &format!("{w}.call_id"))?.to_string();
                let name = as_str(py::get(it, "name"), &format!("{w}.name"))?.to_string();
                let arguments = json!(py::str_of(py::or(it.get("arguments"), &json!("{}"))));
                last_assistant(req).content.push(Part::ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
            Some("function_call_output") => {
                let out = py::get(it, "output");
                let parts = match out {
                    Value::String(s) => vec![Part::Text(s.clone())],
                    _ => content(py::or(Some(out), &json!([])), &format!("{w}.output"))?,
                };
                let call_id = as_str(py::get(it, "call_id"), &format!("{w}.call_id"))?.to_string();
                req.messages.push(Message::new(
                    Role::Tool,
                    vec![Part::ToolResult {
                        call_id,
                        content: parts,
                        is_error: false,
                    }],
                ));
            }
            Some("reasoning") => {
                let text = join_texts(py::get(it, "summary"), true)?;
                last_assistant(req).content.push(Part::Reasoning {
                    text,
                    signature: Value::Null,
                });
            }
            _ => {
                return Err(unsupported(
                    &format!("{w}.type"),
                    &format!("input item {}", py::repr(kind)),
                ))
            }
        }
        Ok(())
    }
}

impl Style for OpenAiResponses {
    fn name(&self) -> &'static str {
        "openai_responses"
    }
    fn path(&self) -> &'static str {
        "v1/responses"
    }
    fn semantic_extensions(&self) -> &'static [&'static str] {
        &["top_logprobs", "include"]
    }

    fn decode_request(
        &self,
        body: &Map<String, Value>,
        ctx: &mut Context,
        _env: &mut dyn Env,
    ) -> TrResult<Request> {
        for key in STATEFUL {
            if py::truthy(py::get(body, key)) {
                return Err(unsupported(key, "SwarmPilot keeps no response state"));
            }
        }
        let mut req = Request {
            model: py::get(body, "model").clone(),
            ..Request::default()
        };
        let instructions = py::get(body, "instructions");
        if !instructions.is_null() {
            req.system
                .push(as_str(instructions, "instructions")?.to_string());
        }
        match py::get(body, "input") {
            Value::String(s) => req
                .messages
                .push(Message::new(Role::User, vec![Part::Text(s.clone())])),
            items => {
                for (i, it) in as_list(items, "input")?.iter().enumerate() {
                    let w = format!("input[{i}]");
                    Self::decode_item(&mut req, as_mapping(it, &w)?, &w)?;
                }
            }
        }
        req.max_output_tokens = opt_int(body, "max_output_tokens")?;
        req.temperature = opt_float(body, "temperature")?;
        req.top_p = opt_float(body, "top_p")?;
        for (j, t) in py::iter_or_empty(py::get(body, "tools"))?
            .iter()
            .enumerate()
        {
            let t = as_mapping(t, &format!("tools[{j}]"))?;
            let kind = py::get(t, "type");
            if kind.as_str() != Some("function") {
                return Err(unsupported(
                    &format!("tools[{j}].type"),
                    &format!("hosted tool {}", py::repr(kind)),
                ));
            }
            req.tools.push(Tool {
                name: as_str(py::get(t, "name"), &format!("tools[{j}].name"))?.to_string(),
                description: py::get(t, "description").clone(),
                parameters: py::or(
                    t.get("parameters"),
                    &json!({"type": "object", "properties": {}}),
                )
                .clone(),
                strict: py::get(t, "strict").clone(),
            });
        }
        let tc = py::get(body, "tool_choice");
        if py::is_one_of(tc, &["auto", "none", "required"]) {
            req.tool_choice = Some(ToolChoice {
                mode: tc.as_str().unwrap_or_default().into(),
                name: Value::Null,
            });
        } else if let Some(m) = tc
            .as_object()
            .filter(|m| py::get(m, "type").as_str() == Some("function"))
        {
            req.tool_choice = Some(ToolChoice {
                mode: "tool".into(),
                name: json!(as_str(py::get(m, "name"), "tool_choice.name")?),
            });
        } else if !tc.is_null() {
            return Err(unsupported("tool_choice", &py::repr(tc)));
        }
        req.parallel_tool_calls = py::get(body, "parallel_tool_calls").clone();
        if let Some(Value::Object(fmt)) = py::get(body, "text")
            .as_object()
            .map(|t| py::get(t, "format"))
        {
            if let Some(k @ ("text" | "json_object" | "json_schema")) =
                py::get(fmt, "type").as_str()
            {
                req.response_format = Some(ResponseFormat {
                    kind: k.into(),
                    name: py::get(fmt, "name").clone(),
                    schema: py::get(fmt, "schema").clone(),
                    strict: py::get(fmt, "strict").clone(),
                });
            }
        }
        if let Value::Object(r) = py::get(body, "reasoning") {
            let effort = py::get(r, "effort");
            if py::truthy(effort) {
                req.reasoning_effort = json!(py::str_of(effort));
            }
        }
        req.stream = py::get(body, "stream") == &Value::Bool(true);
        req.user = py::get(body, "user").clone();
        if py::truthy(py::get(body, "store")) {
            ctx.drop_field("store", false)?;
        }
        let ext: Map<String, Value> = body
            .iter()
            .filter(|(k, _)| !KNOWN.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if !ext.is_empty() {
            req.extensions.push((self.name().into(), ext));
        }
        Ok(req)
    }

    fn encode_response(
        &self,
        resp: &Response,
        req: &Request,
        env: &mut dyn Env,
    ) -> PyResult<Value> {
        let mut out: Vec<Value> = Vec::new();
        for p in &resp.output {
            let it = item(p, env);
            let is_msg = it["type"] == "message";
            match out.last_mut() {
                Some(Value::Object(last)) if is_msg && last["type"] == "message" => {
                    if let (Some(Value::Array(dst)), Value::Array(src)) =
                        (last.get_mut("content"), &it["content"])
                    {
                        dst.extend(src.iter().cloned());
                    }
                }
                _ => out.push(it),
            }
        }
        let model = first_truthy(&[&resp.model, &req.model]);
        let base = response_obj(&resp.id, &model, resp.created, out, req, env);
        Ok(merged(base, status(resp.stop_reason, &resp.usage)?))
    }

    fn encoder(&self, req: &Request, env: &mut dyn Env) -> Box<dyn Encoder> {
        let rid = env.new_id("resp_");
        let created = env.now();
        Box::new(ResponsesEncoder {
            req: req.clone(),
            seq: 0,
            rid,
            model: req.model.clone(),
            created,
            output: Vec::new(),
            items: Vec::new(),
        })
    }

    fn encode_error(
        &self,
        status: u16,
        message: &str,
        error_type: Option<&str>,
        param: Option<&str>,
    ) -> Value {
        openai_error(status, message, error_type, param)
    }

    fn encode_request(&self, req: &Request, ctx: &mut Context) -> TrResult<(String, Value)> {
        let mut items = Vec::new();
        for m in &req.messages {
            if m.role == Role::Tool {
                for p in &m.content {
                    if let Part::ToolResult {
                        call_id, content, ..
                    } = p
                    {
                        if content.iter().any(|x| matches!(x, Part::Image { .. })) {
                            ctx.drop_field("image in tool result", true)?;
                        }
                        let text: String = content
                            .iter()
                            .filter_map(|x| match x {
                                Part::Text(t) => Some(t.as_str()),
                                _ => None,
                            })
                            .collect();
                        items.push(json!({"type": "function_call_output", "call_id": call_id, "output": text}));
                    }
                }
                continue;
            }
            let mut content: Vec<Value> = Vec::new();
            for p in &m.content {
                match p {
                    Part::Text(t) => {
                        let kind = if m.role == Role::Assistant {
                            "output_text"
                        } else {
                            "input_text"
                        };
                        content.push(json!({"type": kind, "text": t}));
                    }
                    Part::Image { url, detail } => {
                        let mut img = Map::new();
                        img.insert("type".into(), json!("input_image"));
                        img.insert("image_url".into(), json!(url));
                        if py::truthy(detail) {
                            img.insert("detail".into(), detail.clone());
                        }
                        content.push(Value::Object(img));
                    }
                    Part::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        if !content.is_empty() {
                            items.push(json!({"role": m.role.name(), "content": std::mem::take(&mut content)}));
                        }
                        items.push(json!({
                            "type": "function_call",
                            "call_id": id,
                            "name": name,
                            "arguments": arguments,
                        }));
                    }
                    _ => {}
                }
            }
            if !content.is_empty() {
                items.push(json!({"role": m.role.name(), "content": content}));
            }
        }
        let mut body = Map::new();
        body.insert("input".into(), Value::Array(items));
        body.insert("store".into(), json!(false));
        if !req.model.is_null() {
            body.insert("model".into(), req.model.clone());
        }
        if !req.system.is_empty() {
            body.insert("instructions".into(), json!(req.system_text()));
        }
        if let Some(n) = req.max_output_tokens {
            body.insert("max_output_tokens".into(), py::int_value(n));
        }
        if let Some(t) = req.temperature {
            body.insert("temperature".into(), py::float_value(t));
        }
        if let Some(t) = req.top_p {
            body.insert("top_p".into(), py::float_value(t));
        }
        if !req.user.is_null() {
            body.insert("user".into(), req.user.clone());
        }
        if !req.parallel_tool_calls.is_null() {
            body.insert(
                "parallel_tool_calls".into(),
                req.parallel_tool_calls.clone(),
            );
        }
        if req.top_k.is_some() {
            ctx.drop_field("top_k", false)?;
        }
        if req.seed.is_some() {
            ctx.drop_field("seed", false)?;
        }
        if !req.stop.is_empty() {
            ctx.drop_field("stop", true)?;
        }
        if !req.tools.is_empty() {
            let tools = req
                .tools
                .iter()
                .map(|t| {
                    let mut o = Map::new();
                    o.insert("type".into(), json!("function"));
                    o.insert("name".into(), json!(t.name));
                    for (k, v) in [
                        ("description", &t.description),
                        ("parameters", &t.parameters),
                        ("strict", &t.strict),
                    ] {
                        if !v.is_null() {
                            o.insert(k.into(), v.clone());
                        }
                    }
                    Value::Object(o)
                })
                .collect();
            body.insert("tools".into(), Value::Array(tools));
        }
        if let Some(tc) = &req.tool_choice {
            body.insert(
                "tool_choice".into(),
                if tc.mode == "tool" {
                    json!({"type": "function", "name": tc.name})
                } else {
                    json!(tc.mode)
                },
            );
        }
        if let Some(rf) = req.response_format.as_ref().filter(|rf| rf.kind != "text") {
            let mut fmt = Map::new();
            fmt.insert("type".into(), json!(rf.kind));
            if rf.kind == "json_schema" {
                fmt.insert(
                    "name".into(),
                    py::or(Some(&rf.name), &json!("response")).clone(),
                );
                fmt.insert(
                    "schema".into(),
                    py::or(Some(&rf.schema), &json!({})).clone(),
                );
                if !rf.strict.is_null() {
                    fmt.insert("strict".into(), rf.strict.clone());
                }
            }
            body.insert("text".into(), json!({"format": Value::Object(fmt)}));
        }
        if !req.reasoning_effort.is_null() {
            body.insert("reasoning".into(), json!({"effort": req.reasoning_effort}));
        } else if req.reasoning_budget.is_some() {
            ctx.drop_field("reasoning budget", false)?;
        }
        if req.stream {
            body.insert("stream".into(), json!(true));
        }
        Ok((self.path().into(), Value::Object(body)))
    }

    fn decode_response(&self, body: &Map<String, Value>, env: &mut dyn Env) -> PyResult<Response> {
        let mut output = Vec::new();
        let mut refused = false;
        let blank = json!("");
        for it in py::iter_or_empty(py::get(body, "output"))? {
            match py::attr_get(&it, "type")?.as_str() {
                Some("message") => {
                    for p in py::iter_or_empty(py::attr_get(&it, "content")?)? {
                        match py::attr_get(&p, "type")?.as_str() {
                            Some("output_text") => output.push(Part::Text(py::str_of(py::or(
                                Some(py::attr_get(&p, "text")?),
                                &blank,
                            )))),
                            Some("refusal") => {
                                refused = true;
                                output.push(Part::Text(py::str_of(py::or(
                                    Some(py::attr_get(&p, "refusal")?),
                                    &blank,
                                ))));
                            }
                            _ => {}
                        }
                    }
                }
                Some("function_call") => {
                    let id = match py::attr_get(&it, "call_id")? {
                        v if py::truthy(v) => py::str_of(v),
                        _ => env.new_id("call_"),
                    };
                    let name = py::str_of(py::or(Some(py::attr_get(&it, "name")?), &blank));
                    let arguments = json!(py::str_of(py::or(
                        Some(py::attr_get(&it, "arguments")?),
                        &json!("{}")
                    )));
                    output.push(Part::ToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                Some("reasoning") => {
                    let mut text = join_texts(py::attr_get(&it, "summary")?, false)?;
                    if text.is_empty() {
                        text = join_texts(py::attr_get(&it, "content")?, false)?;
                    }
                    output.push(Part::Reasoning {
                        text,
                        signature: Value::Null,
                    });
                }
                _ => {}
            }
        }
        let id = match py::get(body, "id") {
            v if py::truthy(v) => py::str_of(v),
            _ => env.new_id("resp_"),
        };
        let has_call = output.iter().any(|p| matches!(p, Part::ToolCall { .. }));
        Ok(Response {
            id,
            model: py::get(body, "model").clone(),
            stop_reason: stop_in(body, has_call, refused)?,
            output,
            stop_sequence: Value::Null,
            usage: usage_in(py::get(body, "usage"))?,
            created: py::int(py::or(body.get("created_at"), &json!(0)))?,
        })
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        Box::new(ResponsesDecoder {
            index: -1,
            open_part: None,
            started: false,
            final_: None,
            calls: false,
        })
    }
}

struct OpenItem {
    /// The canonical part index.
    index: i128,
    /// Position in the output list.
    oi: usize,
    texts: Vec<Value>,
}

struct ResponsesEncoder {
    req: Request,
    seq: i128,
    rid: String,
    model: Value,
    created: i128,
    output: Vec<Value>,
    /// Open parts, the latest for an index last.
    items: Vec<OpenItem>,
}

impl ResponsesEncoder {
    fn ev(&mut self, kind: &str, data: Vec<(&str, Value)>) -> Vec<u8> {
        self.seq += 1;
        let mut m = Map::new();
        m.insert("type".into(), json!(kind));
        m.insert("sequence_number".into(), py::int_value(self.seq - 1));
        for (k, v) in data {
            m.insert(k.into(), v);
        }
        frame(&Value::Object(m), Some(kind))
    }

    fn find(&self, index: i128) -> Option<usize> {
        self.items.iter().rposition(|i| i.index == index)
    }
}

impl Encoder for ResponsesEncoder {
    fn on(&mut self, e: &Event, env: &mut dyn Env, out: &mut Vec<u8>) -> PyResult<bool> {
        match e {
            Event::ResponseStart { model, .. } => {
                if py::truthy(model) {
                    self.model = model.clone();
                }
                let base = merged(
                    response_obj(
                        &self.rid,
                        &self.model,
                        self.created,
                        Vec::new(),
                        &self.req,
                        env,
                    ),
                    json!({"status": "in_progress", "usage": null}),
                );
                let a = self.ev("response.created", vec![("response", base.clone())]);
                out.extend(a);
                let b = self.ev("response.in_progress", vec![("response", base)]);
                out.extend(b);
            }
            Event::PartStart {
                index,
                kind,
                id,
                name,
            } => {
                let oi = self.output.len();
                let it = match *kind {
                    "text" => obj([
                        ("type", json!("message")),
                        ("id", json!(env.new_id("msg_"))),
                        ("status", json!("in_progress")),
                        ("role", json!("assistant")),
                        ("content", json!([])),
                    ]),
                    "tool_call" => obj([
                        ("type", json!("function_call")),
                        ("id", json!(env.new_id("fc_"))),
                        ("call_id", id.clone()),
                        ("name", name.clone()),
                        ("arguments", json!("")),
                        ("status", json!("in_progress")),
                    ]),
                    _ => obj([
                        ("type", json!("reasoning")),
                        ("id", json!(env.new_id("rs_"))),
                        ("summary", json!([])),
                    ]),
                };
                let item_id = it["id"].clone();
                self.items.push(OpenItem {
                    index: *index,
                    oi,
                    texts: Vec::new(),
                });
                self.output.push(it.clone());
                let a = self.ev(
                    "response.output_item.added",
                    vec![("output_index", json!(oi)), ("item", it)],
                );
                out.extend(a);
                if *kind == "text" {
                    let b = self.ev(
                        "response.content_part.added",
                        vec![
                            ("item_id", item_id),
                            ("output_index", json!(oi)),
                            ("content_index", json!(0)),
                            (
                                "part",
                                json!({"type": "output_text", "text": "", "annotations": []}),
                            ),
                        ],
                    );
                    out.extend(b);
                } else if *kind == "reasoning" {
                    let b = self.ev(
                        "response.reasoning_summary_part.added",
                        vec![
                            ("item_id", item_id),
                            ("output_index", json!(oi)),
                            ("summary_index", json!(0)),
                            ("part", json!({"type": "summary_text", "text": ""})),
                        ],
                    );
                    out.extend(b);
                }
            }
            Event::TextDelta { index, text } => {
                if let Some(k) = self.find(*index) {
                    self.items[k].texts.push(text.clone());
                    let oi = self.items[k].oi;
                    let b = self.ev(
                        "response.output_text.delta",
                        vec![
                            ("item_id", self.output[oi]["id"].clone()),
                            ("output_index", json!(oi)),
                            ("content_index", json!(0)),
                            ("delta", text.clone()),
                            ("logprobs", json!([])),
                        ],
                    );
                    out.extend(b);
                }
            }
            Event::ToolArgsDelta { index, delta } => {
                if let Some(k) = self.find(*index) {
                    self.items[k].texts.push(delta.clone());
                    let oi = self.items[k].oi;
                    let b = self.ev(
                        "response.function_call_arguments.delta",
                        vec![
                            ("item_id", self.output[oi]["id"].clone()),
                            ("output_index", json!(oi)),
                            ("delta", delta.clone()),
                        ],
                    );
                    out.extend(b);
                }
            }
            Event::ReasoningDelta { index, text, .. } => {
                if let Some(k) = self.find(*index) {
                    if !text.is_empty() {
                        self.items[k].texts.push(json!(text));
                        let oi = self.items[k].oi;
                        let b = self.ev(
                            "response.reasoning_summary_text.delta",
                            vec![
                                ("item_id", self.output[oi]["id"].clone()),
                                ("output_index", json!(oi)),
                                ("summary_index", json!(0)),
                                ("delta", json!(text)),
                            ],
                        );
                        out.extend(b);
                    }
                }
            }
            Event::PartStop { index } => {
                if let Some(k) = self.find(*index) {
                    let oi = self.items[k].oi;
                    let full = py::join(&self.items[k].texts)?;
                    let item_id = self.output[oi]["id"].clone();
                    let kind = self.output[oi]["type"].clone();
                    if kind == "message" {
                        let part = json!({"type": "output_text", "text": full, "annotations": []});
                        let a = self.ev(
                            "response.output_text.done",
                            vec![
                                ("item_id", item_id.clone()),
                                ("output_index", json!(oi)),
                                ("content_index", json!(0)),
                                ("text", json!(full)),
                                ("logprobs", json!([])),
                            ],
                        );
                        out.extend(a);
                        let b = self.ev(
                            "response.content_part.done",
                            vec![
                                ("item_id", item_id),
                                ("output_index", json!(oi)),
                                ("content_index", json!(0)),
                                ("part", part.clone()),
                            ],
                        );
                        out.extend(b);
                        self.output[oi]["content"] = json!([part]);
                        self.output[oi]["status"] = json!("completed");
                    } else if kind == "function_call" {
                        let a = self.ev(
                            "response.function_call_arguments.done",
                            vec![
                                ("item_id", item_id),
                                ("output_index", json!(oi)),
                                ("arguments", json!(full)),
                            ],
                        );
                        out.extend(a);
                        self.output[oi]["arguments"] = json!(full);
                        self.output[oi]["status"] = json!("completed");
                    } else {
                        let part = json!({"type": "summary_text", "text": full});
                        let a = self.ev(
                            "response.reasoning_summary_text.done",
                            vec![
                                ("item_id", item_id.clone()),
                                ("output_index", json!(oi)),
                                ("summary_index", json!(0)),
                                ("text", json!(full)),
                            ],
                        );
                        out.extend(a);
                        let b = self.ev(
                            "response.reasoning_summary_part.done",
                            vec![
                                ("item_id", item_id),
                                ("output_index", json!(oi)),
                                ("summary_index", json!(0)),
                                ("part", part.clone()),
                            ],
                        );
                        out.extend(b);
                        self.output[oi]["summary"] = json!([part]);
                    }
                    let it = self.output[oi].clone();
                    let c = self.ev(
                        "response.output_item.done",
                        vec![("output_index", json!(oi)), ("item", it)],
                    );
                    out.extend(c);
                }
            }
            Event::ResponseStop {
                stop_reason, usage, ..
            } => {
                let base = response_obj(
                    &self.rid,
                    &self.model,
                    self.created,
                    self.output.clone(),
                    &self.req,
                    env,
                );
                let fin = merged(base, status(*stop_reason, usage)?);
                let kind = if fin["status"] == "completed" {
                    "response.completed"
                } else {
                    "response.incomplete"
                };
                let a = self.ev(kind, vec![("response", fin)]);
                out.extend(a);
                return Ok(true);
            }
            Event::Error {
                message,
                error_type,
            } => {
                let a = self.ev(
                    "error",
                    vec![
                        ("code", json!(error_type)),
                        ("message", json!(message)),
                        ("param", Value::Null),
                    ],
                );
                out.extend(a);
                return Ok(true);
            }
        }
        Ok(false)
    }
}

struct ResponsesDecoder {
    index: i128,
    open_part: Option<&'static str>,
    started: bool,
    final_: Option<Value>,
    calls: bool,
}

impl Decoder for ResponsesDecoder {
    fn on_data(&mut self, data: &str, env: &mut dyn Env, out: &mut Vec<Event>) -> PyResult<Flow> {
        let Some(o) = py::loads(data) else {
            return Ok(Flow::Continue);
        };
        let empty = json!({});
        let blank = json!("");
        let kind = py::attr_get(&o, "type")?
            .as_str()
            .unwrap_or_default()
            .to_string();
        let delta = |o: &Value| -> PyResult<String> {
            Ok(py::str_of(py::or(Some(py::attr_get(o, "delta")?), &blank)))
        };
        match kind.as_str() {
            "response.created" => {
                let r = py::or(Some(py::attr_get(&o, "response")?), &empty);
                self.started = true;
                let id = match py::attr_get(r, "id")? {
                    v if py::truthy(v) => py::str_of(v),
                    _ => env.new_id("resp_"),
                };
                let model = py::attr_get(r, "model")?.clone();
                let created = py::int(py::or(Some(py::attr_get(r, "created_at")?), &json!(0)))?;
                out.push(Event::ResponseStart { id, model, created });
            }
            "response.output_item.added" => {
                let it = py::or(Some(py::attr_get(&o, "item")?), &empty);
                match py::attr_get(it, "type")?.as_str() {
                    Some("function_call") => {
                        self.index += 1;
                        self.open_part = Some("tool_call");
                        self.calls = true;
                        out.push(Event::PartStart {
                            index: self.index,
                            kind: "tool_call",
                            id: py::attr_get(it, "call_id")?.clone(),
                            name: py::attr_get(it, "name")?.clone(),
                        });
                    }
                    Some("reasoning") => {
                        self.index += 1;
                        self.open_part = Some("reasoning");
                        out.push(Event::PartStart {
                            index: self.index,
                            kind: "reasoning",
                            id: Value::Null,
                            name: Value::Null,
                        });
                    }
                    _ => {}
                }
            }
            "response.content_part.added" => {
                self.index += 1;
                self.open_part = Some("text");
                out.push(Event::PartStart {
                    index: self.index,
                    kind: "text",
                    id: Value::Null,
                    name: Value::Null,
                });
            }
            "response.output_text.delta" if self.open_part == Some("text") => {
                out.push(Event::TextDelta {
                    index: self.index,
                    text: json!(delta(&o)?),
                })
            }
            "response.function_call_arguments.delta" => out.push(Event::ToolArgsDelta {
                index: self.index,
                delta: json!(delta(&o)?),
            }),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                out.push(Event::ReasoningDelta {
                    index: self.index,
                    text: delta(&o)?,
                    signature: Value::Null,
                })
            }
            "response.content_part.done" if self.open_part == Some("text") => {
                self.open_part = None;
                out.push(Event::PartStop { index: self.index });
            }
            "response.output_item.done"
                if matches!(self.open_part, Some("tool_call" | "reasoning")) =>
            {
                self.open_part = None;
                out.push(Event::PartStop { index: self.index });
            }
            "response.completed" | "response.incomplete" => {
                self.final_ = Some(py::or(Some(py::attr_get(&o, "response")?), &empty).clone());
                return Ok(Flow::Break);
            }
            "response.failed" | "error" => {
                let r = py::or(Some(py::attr_get(&o, "response")?), &empty);
                let e = py::or(Some(py::attr_get(r, "error")?), &o);
                let msg = py::or(Some(py::attr_get(e, "message")?), &json!("Engine error")).clone();
                out.push(Event::error(py::str_of(&msg)));
                return Ok(Flow::Return);
            }
            _ => {}
        }
        Ok(Flow::Continue)
    }

    fn finish(&mut self, _env: &mut dyn Env, out: &mut Vec<Event>) -> PyResult<()> {
        if !self.started {
            out.push(Event::error("Engine stream ended without a response"));
            return Ok(());
        }
        if self.open_part.is_some() {
            out.push(Event::PartStop { index: self.index });
        }
        let empty = Map::new();
        let fin = match &self.final_ {
            Some(Value::Object(m)) => m,
            Some(v) if py::truthy(v) => return py::attr_get(v, "incomplete_details").map(|_| ()),
            _ => &empty,
        };
        let mut stop = stop_in(fin, false, false)?;
        if self.calls && stop == StopReason::EndTurn {
            stop = StopReason::ToolUse;
        }
        out.push(Event::ResponseStop {
            stop_reason: stop,
            stop_sequence: Value::Null,
            usage: usage_in(py::get(fin, "usage"))?,
        });
        Ok(())
    }
}

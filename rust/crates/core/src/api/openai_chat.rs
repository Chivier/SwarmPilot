//! OpenAI Chat Completions (`swarmpilot/api/styles/openai_chat.py`).

use serde_json::{json, Map, Value};

use super::base::{
    as_list, as_mapping, as_str, first_truthy, invalid, obj, opt_float, opt_int, stop_list,
    unsupported, Context, Env, TrResult,
};
use super::canonical::{
    Event, Message, Part, Request, Response, ResponseFormat, Role, StopReason, Tool, ToolChoice,
    Usage,
};
use super::py::{self, PyResult};
use super::sse::frame;
use super::style::{Decoder, Encoder, Flow, Style};

const KNOWN: [&str; 16] = [
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop",
    "seed",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "stream",
    "user",
];

fn finish_to_stop(finish: &str) -> Option<StopReason> {
    Some(match finish {
        "stop" => StopReason::EndTurn,
        "length" => StopReason::MaxTokens,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "content_filter" => StopReason::ContentFilter,
        _ => return None,
    })
}

fn stop_to_finish(s: StopReason) -> &'static str {
    match s {
        StopReason::EndTurn | StopReason::StopSequence | StopReason::Error => "stop",
        StopReason::MaxTokens => "length",
        StopReason::ToolUse => "tool_calls",
        StopReason::ContentFilter => "content_filter",
    }
}

pub(crate) fn error_type_for(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_exceeded",
        503 => "service_unavailable",
        504 => "gateway_timeout",
        _ => "api_error",
    }
}

/// The OpenAI error object (shared with the Responses style).
pub(crate) fn openai_error(
    status: u16,
    message: &str,
    error_type: Option<&str>,
    param: Option<&str>,
) -> Value {
    let kind = match error_type {
        Some(t) if !t.is_empty() => t,
        _ => error_type_for(status),
    };
    json!({"error": {"message": message, "type": kind, "param": param, "code": null}})
}

fn user_parts(content: &Value, where_: &str) -> TrResult<Vec<Part>> {
    if let Value::String(s) = content {
        return Ok(vec![Part::Text(s.clone())]);
    }
    let mut parts = Vec::new();
    for (i, p) in as_list(content, where_)?.iter().enumerate() {
        let w = format!("{where_}[{i}]");
        let p = as_mapping(p, &w)?;
        let kind = py::get(p, "type");
        match kind.as_str() {
            Some("text") => parts.push(Part::Text(
                as_str(py::get(p, "text"), &format!("{w}.text"))?.to_string(),
            )),
            Some("image_url") => {
                let img = py::get(p, "image_url");
                let (url, detail) = match img {
                    Value::Object(o) => (py::get(o, "url"), py::get(o, "detail")),
                    _ => (img, &Value::Null),
                };
                let url = as_str(url, &format!("{w}.image_url"))?.to_string();
                parts.push(Part::Image {
                    url,
                    detail: if py::truthy(detail) {
                        detail.clone()
                    } else {
                        Value::Null
                    },
                });
            }
            Some("refusal") => {
                parts.push(Part::Text(py::str_of(py::or(p.get("refusal"), &json!("")))))
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

fn tool_results(m: &Map<String, Value>, where_: &str) -> TrResult<Message> {
    let call_id = as_str(
        py::get(m, "tool_call_id"),
        &format!("{where_}.tool_call_id"),
    )?
    .to_string();
    let content = user_parts(
        py::or(m.get("content"), &json!("")),
        &format!("{where_}.content"),
    )?;
    Ok(Message::new(
        Role::Tool,
        vec![Part::ToolResult {
            call_id,
            content,
            is_error: false,
        }],
    ))
}

fn assistant(m: &Map<String, Value>, where_: &str, env: &mut dyn Env) -> TrResult<Message> {
    let mut parts = Vec::new();
    let reasoning = py::or(m.get("reasoning_content"), py::get(m, "reasoning"));
    if let Value::String(r) = reasoning {
        if !r.is_empty() {
            parts.push(Part::Reasoning {
                text: r.clone(),
                signature: Value::Null,
            });
        }
    }
    let content = py::get(m, "content");
    if py::truthy(content) {
        parts.extend(user_parts(content, &format!("{where_}.content"))?);
    }
    for (j, call) in py::iter_or_empty(py::get(m, "tool_calls"))?
        .iter()
        .enumerate()
    {
        let w = format!("{where_}.tool_calls[{j}]");
        let call = as_mapping(call, &w)?;
        let func = as_mapping(py::get(call, "function"), &w)?;
        let args = py::get(func, "arguments");
        let id = match call.get("id") {
            Some(v) if py::truthy(v) => py::str_of(v),
            _ => env.new_id("call_"),
        };
        let name = as_str(py::get(func, "name"), &format!("{w}.name"))?.to_string();
        parts.push(Part::ToolCall {
            id,
            name,
            arguments: if args.is_string() {
                args.clone()
            } else {
                json!("{}")
            },
        });
    }
    Ok(Message::new(Role::Assistant, parts))
}

fn tool_choice(v: &Value) -> TrResult<Option<ToolChoice>> {
    if v.is_null() {
        return Ok(None);
    }
    if py::is_one_of(v, &["auto", "none", "required"]) {
        return Ok(Some(ToolChoice {
            mode: v.as_str().unwrap_or_default().into(),
            name: Value::Null,
        }));
    }
    if let Some(Value::Object(f)) = v.as_object().and_then(|o| o.get("function")) {
        return Ok(Some(ToolChoice {
            mode: "tool".into(),
            name: Value::String(as_str(py::get(f, "name"), "tool_choice")?.to_string()),
        }));
    }
    Err(invalid("invalid 'tool_choice'", Some("tool_choice")))
}

fn response_format(v: &Value) -> TrResult<Option<ResponseFormat>> {
    if v.is_null() {
        return Ok(None);
    }
    let v = as_mapping(v, "response_format")?;
    let kind = py::get(v, "type");
    match kind.as_str() {
        Some(k @ ("text" | "json_object")) => Ok(Some(ResponseFormat {
            kind: k.into(),
            name: Value::Null,
            schema: Value::Null,
            strict: Value::Null,
        })),
        Some("json_schema") => {
            let js = as_mapping(py::get(v, "json_schema"), "response_format.json_schema")?;
            Ok(Some(ResponseFormat {
                kind: "json_schema".into(),
                name: py::get(js, "name").clone(),
                schema: py::get(js, "schema").clone(),
                strict: py::get(js, "strict").clone(),
            }))
        }
        _ => Err(invalid(
            "invalid 'response_format'",
            Some("response_format"),
        )),
    }
}

pub struct OpenAiChat;

impl Style for OpenAiChat {
    fn name(&self) -> &'static str {
        "openai_chat"
    }
    fn path(&self) -> &'static str {
        "v1/chat/completions"
    }
    fn semantic_extensions(&self) -> &'static [&'static str] {
        &[
            "n",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "audio",
            "modalities",
        ]
    }
    fn handled_extensions(&self) -> &'static [&'static str] {
        &["stream_options"]
    }

    fn decode_request(
        &self,
        body: &Map<String, Value>,
        _ctx: &mut Context,
        env: &mut dyn Env,
    ) -> TrResult<Request> {
        let mut req = Request {
            model: py::get(body, "model").clone(),
            ..Request::default()
        };
        for (i, m) in as_list(py::get(body, "messages"), "messages")?
            .iter()
            .enumerate()
        {
            let w = format!("messages[{i}]");
            let m = as_mapping(m, &w)?;
            let role = py::get(m, "role");
            match role.as_str() {
                Some("system" | "developer") => {
                    let text: String = user_parts(py::or(m.get("content"), &json!("")), &w)?
                        .into_iter()
                        .filter_map(|p| match p {
                            Part::Text(t) => Some(t),
                            _ => None,
                        })
                        .collect();
                    req.system.push(text);
                }
                Some("user") => {
                    let parts = user_parts(py::get(m, "content"), &w)?;
                    req.messages.push(Message::new(Role::User, parts));
                }
                Some("assistant") => req.messages.push(assistant(m, &w, env)?),
                Some("tool") => req.messages.push(tool_results(m, &w)?),
                _ => {
                    return Err(unsupported(
                        &format!("{w}.role"),
                        &format!("role {}", py::repr(role)),
                    ))
                }
            }
        }
        req.max_output_tokens = opt_int(body, "max_completion_tokens")?;
        if req.max_output_tokens.is_none() {
            req.max_output_tokens = opt_int(body, "max_tokens")?;
        }
        req.temperature = opt_float(body, "temperature")?;
        req.top_p = opt_float(body, "top_p")?;
        req.top_k = opt_int(body, "top_k")?;
        req.stop = stop_list(py::get(body, "stop"), "stop")?;
        req.seed = opt_int(body, "seed")?;
        for (j, t) in py::iter_or_empty(py::get(body, "tools"))?
            .iter()
            .enumerate()
        {
            let t = as_mapping(t, &format!("tools[{j}]"))?;
            let kind = py::get(t, "type");
            if kind.as_str() != Some("function") {
                return Err(unsupported(
                    &format!("tools[{j}].type"),
                    &format!("tool type {}", py::repr(kind)),
                ));
            }
            let f = as_mapping(py::get(t, "function"), &format!("tools[{j}].function"))?;
            req.tools.push(Tool {
                name: as_str(py::get(f, "name"), &format!("tools[{j}].function.name"))?.to_string(),
                description: py::get(f, "description").clone(),
                parameters: py::or(
                    f.get("parameters"),
                    &json!({"type": "object", "properties": {}}),
                )
                .clone(),
                strict: py::get(f, "strict").clone(),
            });
        }
        req.tool_choice = tool_choice(py::get(body, "tool_choice"))?;
        req.parallel_tool_calls = py::get(body, "parallel_tool_calls").clone();
        req.response_format = response_format(py::get(body, "response_format"))?;
        req.reasoning_effort = py::get(body, "reasoning_effort").clone();
        req.stream = py::get(body, "stream") == &Value::Bool(true);
        req.user = py::get(body, "user").clone();
        let mut ext: Map<String, Value> = body
            .iter()
            .filter(|(k, _)| !KNOWN.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let n = py::get(&ext, "n");
        if n.is_null() || py::eq(n, &json!(1)) {
            ext.shift_remove("n");
        }
        let lp = py::get(&ext, "logprobs");
        if lp.is_null() || py::eq(lp, &json!(false)) {
            ext.shift_remove("logprobs");
        }
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
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut calls = Vec::new();
        for p in &resp.output {
            match p {
                Part::Text(t) => text.push_str(t),
                Part::Reasoning { text: r, .. } => reasoning.push_str(r),
                Part::ToolCall {
                    id,
                    name,
                    arguments,
                } => calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                })),
                _ => {}
            }
        }
        let mut message = Map::new();
        message.insert("role".into(), json!("assistant"));
        message.insert(
            "content".into(),
            if !text.is_empty() || calls.is_empty() {
                json!(text)
            } else {
                Value::Null
            },
        );
        if !calls.is_empty() {
            message.insert("tool_calls".into(), Value::Array(calls));
        }
        if !reasoning.is_empty() {
            message.insert("reasoning_content".into(), json!(reasoning));
        }
        let id = if resp.id.starts_with("chatcmpl") {
            resp.id.clone()
        } else {
            format!("chatcmpl-{}", resp.id)
        };
        let created = if resp.created != 0 {
            resp.created
        } else {
            env.now()
        };
        Ok(obj([
            ("id", json!(id)),
            ("object", json!("chat.completion")),
            ("created", py::int_value(created)),
            ("model", first_truthy(&[&resp.model, &req.model])),
            (
                "choices",
                json!([{
                    "index": 0,
                    "message": Value::Object(message),
                    "logprobs": null,
                    "finish_reason": stop_to_finish(resp.stop_reason),
                }]),
            ),
            ("usage", usage_out(&resp.usage)?),
        ]))
    }

    fn encoder(&self, req: &Request, env: &mut dyn Env) -> Box<dyn Encoder> {
        let include_usage = req
            .extension(self.name())
            .and_then(|e| e.get("stream_options"))
            .and_then(Value::as_object)
            .is_some_and(|o| py::truthy(py::get(o, "include_usage")));
        let cid = env.new_id("chatcmpl-");
        let model = first_truthy(&[&req.model]);
        let created = env.now();
        Box::new(ChatEncoder {
            include_usage,
            cid,
            model,
            created,
            tool_index: Vec::new(),
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
        let mut messages = Vec::new();
        if !req.system.is_empty() {
            messages.push(json!({"role": "system", "content": req.system_text()}));
        }
        for m in &req.messages {
            messages.extend(encode_message(m, ctx)?);
        }
        let mut body = Map::new();
        body.insert("messages".into(), Value::Array(messages));
        if !req.model.is_null() {
            body.insert("model".into(), req.model.clone());
        }
        if let Some(n) = req.max_output_tokens {
            body.insert("max_tokens".into(), py::int_value(n));
        }
        if let Some(t) = req.temperature {
            body.insert("temperature".into(), py::float_value(t));
        }
        if let Some(t) = req.top_p {
            body.insert("top_p".into(), py::float_value(t));
        }
        if let Some(k) = req.top_k {
            body.insert("top_k".into(), py::int_value(k));
        }
        if let Some(s) = req.seed {
            body.insert("seed".into(), py::int_value(s));
        }
        if !req.user.is_null() {
            body.insert("user".into(), req.user.clone());
        }
        if !req.stop.is_empty() {
            body.insert("stop".into(), json!(req.stop));
        }
        if !req.tools.is_empty() {
            let tools = req
                .tools
                .iter()
                .map(|t| {
                    let mut f = Map::new();
                    f.insert("name".into(), json!(t.name));
                    for (k, v) in [
                        ("description", &t.description),
                        ("parameters", &t.parameters),
                        ("strict", &t.strict),
                    ] {
                        if !v.is_null() {
                            f.insert(k.into(), v.clone());
                        }
                    }
                    json!({"type": "function", "function": Value::Object(f)})
                })
                .collect();
            body.insert("tools".into(), Value::Array(tools));
        }
        if let Some(tc) = &req.tool_choice {
            body.insert(
                "tool_choice".into(),
                if tc.mode == "tool" {
                    json!({"type": "function", "function": {"name": tc.name}})
                } else {
                    json!(tc.mode)
                },
            );
        }
        if !req.parallel_tool_calls.is_null() {
            body.insert(
                "parallel_tool_calls".into(),
                req.parallel_tool_calls.clone(),
            );
        }
        if let Some(rf) = &req.response_format {
            if rf.kind != "text" {
                if rf.kind == "json_object" {
                    body.insert("response_format".into(), json!({"type": "json_object"}));
                } else {
                    let mut js = Map::new();
                    js.insert(
                        "name".into(),
                        py::or(Some(&rf.name), &json!("response")).clone(),
                    );
                    js.insert(
                        "schema".into(),
                        py::or(Some(&rf.schema), &json!({})).clone(),
                    );
                    if !rf.strict.is_null() {
                        js.insert("strict".into(), rf.strict.clone());
                    }
                    body.insert(
                        "response_format".into(),
                        json!({"type": "json_schema", "json_schema": Value::Object(js)}),
                    );
                }
            }
        }
        if !req.reasoning_effort.is_null() {
            body.insert("reasoning_effort".into(), req.reasoning_effort.clone());
        } else if req.reasoning_budget.is_some() {
            ctx.drop_field("reasoning budget", false)?;
        }
        if req.stream {
            body.insert("stream".into(), json!(true));
            body.insert("stream_options".into(), json!({"include_usage": true}));
        }
        Ok((self.path().into(), Value::Object(body)))
    }

    fn decode_response(&self, body: &Map<String, Value>, env: &mut dyn Env) -> PyResult<Response> {
        let empty = json!({});
        let choices = py::or(body.get("choices"), &Value::Null);
        let choice = match choices {
            Value::Null => &empty,
            Value::Array(a) => match &a[0] {
                c @ Value::Object(_) => c,
                _ => &empty,
            },
            Value::String(_) => &empty,
            Value::Object(_) => return py::err("KeyError: 0"),
            other => {
                return py::err(format!(
                    "'{}' object is not subscriptable",
                    py::type_name(other)
                ))
            }
        };
        let msg = py::or(py::attr_get(choice, "message").ok(), &empty);
        let mut output = Vec::new();
        let reasoning = py::or(
            Some(py::attr_get(msg, "reasoning_content")?),
            py::attr_get(msg, "reasoning")?,
        );
        if let Value::String(r) = reasoning {
            if !r.is_empty() {
                output.push(Part::Reasoning {
                    text: r.clone(),
                    signature: Value::Null,
                });
            }
        }
        if let Value::String(t) = py::attr_get(msg, "content")? {
            if !t.is_empty() {
                output.push(Part::Text(t.clone()));
            }
        }
        for call in py::iter_or_empty(py::attr_get(msg, "tool_calls")?)? {
            let f = py::or(Some(py::attr_get(&call, "function")?), &empty).clone();
            let id = match py::attr_get(&call, "id")? {
                v if py::truthy(v) => py::str_of(v),
                _ => env.new_id("call_"),
            };
            let name = py::str_of(py::or(Some(py::attr_get(&f, "name")?), &json!("")));
            let arguments = py::or(Some(py::attr_get(&f, "arguments")?), &json!("{}")).clone();
            output.push(Part::ToolCall {
                id,
                name,
                arguments,
            });
        }
        let (stop, seq) = stop_reason(choice)?;
        let id = match py::get(body, "id") {
            v if py::truthy(v) => py::str_of(v),
            _ => env.new_id("chatcmpl-"),
        };
        Ok(Response {
            id,
            model: py::get(body, "model").clone(),
            output,
            stop_reason: stop,
            stop_sequence: seq,
            usage: usage_in(py::get(body, "usage"))?,
            created: py::int(py::or(body.get("created"), &json!(0)))?,
        })
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        Box::new(ChatDecoder {
            started: false,
            index: -1,
            open_key: None,
            stop: StopReason::EndTurn,
            seq: Value::Null,
            usage: Usage::default(),
        })
    }
}

fn encode_message(m: &Message, ctx: &mut Context) -> TrResult<Vec<Value>> {
    if m.role == Role::Tool {
        let mut out = Vec::new();
        for p in &m.content {
            if let Part::ToolResult {
                call_id, content, ..
            } = p
            {
                let text: String = content
                    .iter()
                    .filter_map(|x| match x {
                        Part::Text(t) => Some(t.as_str()),
                        _ => None,
                    })
                    .collect();
                if content.iter().any(|x| matches!(x, Part::Image { .. })) {
                    ctx.drop_field("image in tool result", true)?;
                }
                out.push(json!({"role": "tool", "tool_call_id": call_id, "content": text}));
            }
        }
        return Ok(out);
    }
    let mut content: Vec<Value> = Vec::new();
    let mut texts: Vec<&str> = Vec::new();
    let mut all_text = true;
    let mut calls = Vec::new();
    for p in &m.content {
        match p {
            Part::Text(t) => {
                content.push(json!({"type": "text", "text": t}));
                texts.push(t);
            }
            Part::Image { url, detail } => {
                let mut img = Map::new();
                img.insert("url".into(), json!(url));
                if py::truthy(detail) {
                    img.insert("detail".into(), detail.clone());
                }
                content.push(json!({"type": "image_url", "image_url": Value::Object(img)}));
                all_text = false;
            }
            Part::ToolCall {
                id,
                name,
                arguments,
            } => calls.push(json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })),
            _ => {}
        }
    }
    let mut out = Map::new();
    out.insert("role".into(), json!(m.role.name()));
    if all_text {
        let joined: String = texts.concat();
        out.insert(
            "content".into(),
            if !joined.is_empty() {
                json!(joined)
            } else if !calls.is_empty() {
                Value::Null
            } else {
                json!("")
            },
        );
    } else {
        out.insert("content".into(), Value::Array(content));
    }
    if !calls.is_empty() {
        out.insert("tool_calls".into(), Value::Array(calls));
    }
    Ok(vec![Value::Object(out)])
}

fn stop_reason(choice: &Value) -> PyResult<(StopReason, Value)> {
    let finish = py::attr_get(choice, "finish_reason")?;
    let matched = py::attr_get(choice, "stop_reason")?;
    if finish.as_str() == Some("stop") {
        if let Value::String(m) = matched {
            return Ok((StopReason::StopSequence, json!(m)));
        }
    }
    let stop = py::str_key(finish)?
        .and_then(finish_to_stop)
        .unwrap_or(StopReason::EndTurn);
    Ok((stop, Value::Null))
}

fn usage_in(u: &Value) -> PyResult<Usage> {
    let Value::Object(u) = u else {
        return Ok(Usage::default());
    };
    let empty = json!({});
    let pd = py::or(u.get("prompt_tokens_details"), &empty);
    let cd = py::or(u.get("completion_tokens_details"), &empty);
    Ok(Usage {
        input_tokens: py::get(u, "prompt_tokens").clone(),
        output_tokens: py::get(u, "completion_tokens").clone(),
        cached_input_tokens: if py::truthy(pd) {
            py::attr_get(pd, "cached_tokens")?.clone()
        } else {
            Value::Null
        },
        reasoning_tokens: if py::truthy(cd) {
            py::attr_get(cd, "reasoning_tokens")?.clone()
        } else {
            Value::Null
        },
    })
}

fn usage_out(u: &Usage) -> PyResult<Value> {
    let zero = json!(0);
    let i = py::or(Some(&u.input_tokens), &zero);
    let o = py::or(Some(&u.output_tokens), &zero);
    let mut out = Map::new();
    out.insert("prompt_tokens".into(), i.clone());
    out.insert("completion_tokens".into(), o.clone());
    out.insert("total_tokens".into(), py::add(i, o)?);
    if !u.cached_input_tokens.is_null() {
        out.insert(
            "prompt_tokens_details".into(),
            json!({"cached_tokens": u.cached_input_tokens}),
        );
    }
    if !u.reasoning_tokens.is_null() {
        out.insert(
            "completion_tokens_details".into(),
            json!({"reasoning_tokens": u.reasoning_tokens}),
        );
    }
    Ok(Value::Object(out))
}

struct ChatEncoder {
    include_usage: bool,
    cid: String,
    model: Value,
    created: i128,
    /// Canonical part index to the chat tool-call index.
    tool_index: Vec<(i128, usize)>,
}

impl ChatEncoder {
    fn chunk(&self, delta: Value, finish: Option<&str>) -> Vec<u8> {
        frame(
            &obj([
                ("id", json!(self.cid)),
                ("object", json!("chat.completion.chunk")),
                ("created", py::int_value(self.created)),
                ("model", self.model.clone()),
                (
                    "choices",
                    json!([{"index": 0, "delta": delta, "finish_reason": finish}]),
                ),
            ]),
            None,
        )
    }
}

impl Encoder for ChatEncoder {
    fn on(&mut self, ev: &Event, _env: &mut dyn Env, out: &mut Vec<u8>) -> PyResult<bool> {
        match ev {
            Event::ResponseStart { model, created, .. } => {
                if py::truthy(model) {
                    self.model = model.clone();
                }
                if *created != 0 {
                    self.created = *created;
                }
                out.extend(self.chunk(json!({"role": "assistant", "content": ""}), None));
            }
            Event::TextDelta { text, .. } => {
                if py::truthy(text) {
                    out.extend(self.chunk(json!({"content": text}), None));
                }
            }
            Event::ReasoningDelta { text, .. } => {
                if !text.is_empty() {
                    out.extend(self.chunk(json!({"reasoning_content": text}), None));
                }
            }
            Event::PartStart {
                index,
                kind: "tool_call",
                id,
                name,
            } => {
                let k = self.tool_index.len();
                match self.tool_index.iter_mut().find(|(i, _)| i == index) {
                    Some(e) => e.1 = k,
                    None => self.tool_index.push((*index, k)),
                }
                out.extend(self.chunk(
                    json!({"tool_calls": [{
                        "index": k,
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }]}),
                    None,
                ));
            }
            Event::ToolArgsDelta { index, delta } => {
                let k = self
                    .tool_index
                    .iter()
                    .find(|(i, _)| i == index)
                    .map_or(0, |e| e.1);
                out.extend(self.chunk(
                    json!({"tool_calls": [{"index": k, "function": {"arguments": delta}}]}),
                    None,
                ));
            }
            Event::ResponseStop {
                stop_reason, usage, ..
            } => {
                out.extend(self.chunk(json!({}), Some(stop_to_finish(*stop_reason))));
                if self.include_usage {
                    out.extend(frame(
                        &obj([
                            ("id", json!(self.cid)),
                            ("object", json!("chat.completion.chunk")),
                            ("created", py::int_value(self.created)),
                            ("model", self.model.clone()),
                            ("choices", json!([])),
                            ("usage", usage_out(usage)?),
                        ]),
                        None,
                    ));
                }
                out.extend_from_slice(b"data: [DONE]\n\n");
                return Ok(true);
            }
            Event::Error {
                message,
                error_type,
            } => {
                out.extend(frame(
                    &openai_error(500, message, Some(error_type), None),
                    None,
                ));
                out.extend_from_slice(b"data: [DONE]\n\n");
                return Ok(true);
            }
            _ => {}
        }
        Ok(false)
    }
}

struct ChatDecoder {
    started: bool,
    index: i128,
    /// The open part: its kind and, for tool calls, the chat index.
    open_key: Option<(&'static str, Value)>,
    stop: StopReason,
    seq: Value,
    usage: Usage,
}

impl Decoder for ChatDecoder {
    fn on_data(&mut self, data: &str, env: &mut dyn Env, out: &mut Vec<Event>) -> PyResult<Flow> {
        if py::strip(data) == "[DONE]" {
            return Ok(Flow::Break);
        }
        let Some(o) = py::loads(data) else {
            return Ok(Flow::Continue);
        };
        if py::contains(&o, "error")? && !py::contains(&o, "choices")? {
            out.push(Event::error(OpenAiChat.decode_error(&o)));
            return Ok(Flow::Return);
        }
        if !self.started {
            self.started = true;
            let id = match py::attr_get(&o, "id")? {
                v if py::truthy(v) => py::str_of(v),
                _ => env.new_id("chatcmpl-"),
            };
            let model = py::attr_get(&o, "model")?.clone();
            let created = py::int(py::or(Some(py::attr_get(&o, "created")?), &json!(0)))?;
            out.push(Event::ResponseStart { id, model, created });
        }
        let usage = py::attr_get(&o, "usage")?;
        if py::truthy(usage) {
            self.usage = usage_in(usage)?;
        }
        let empty = json!({});
        let zero = json!(0);
        for choice in py::iter_or_empty(py::attr_get(&o, "choices")?)? {
            let Value::Object(c) = &choice else {
                return py::err("'choice' object has no attribute 'get'");
            };
            if !py::eq(py::get_or(c, "index", &zero), &zero) {
                continue;
            }
            let delta = py::or(c.get("delta"), &empty);
            let mut pieces: Vec<((&'static str, Value), Value)> = Vec::new();
            let r = py::or(
                Some(py::attr_get(delta, "reasoning_content")?),
                py::attr_get(delta, "reasoning")?,
            );
            if let Value::String(s) = r {
                if !s.is_empty() {
                    pieces.push((("reasoning", zero.clone()), r.clone()));
                }
            }
            if let Value::String(s) = py::attr_get(delta, "content")? {
                if !s.is_empty() {
                    pieces.push((("text", zero.clone()), json!(s)));
                }
            }
            for call in py::iter_or_empty(py::attr_get(delta, "tool_calls")?)? {
                let Value::Object(co) = &call else {
                    return py::err("'call' object has no attribute 'get'");
                };
                let idx = py::get_or(co, "index", &zero).clone();
                pieces.push((("tool_call", idx), call.clone()));
            }
            for (key, value) in pieces {
                let same = self
                    .open_key
                    .as_ref()
                    .is_some_and(|(k, i)| *k == key.0 && py::eq(i, &key.1));
                if !same {
                    if self.open_key.is_some() {
                        out.push(Event::PartStop { index: self.index });
                    }
                    self.index += 1;
                    let kind = key.0;
                    self.open_key = Some(key);
                    if kind == "tool_call" {
                        let f = py::or(Some(py::attr_get(&value, "function")?), &empty).clone();
                        let id = match py::attr_get(&value, "id")? {
                            v if py::truthy(v) => v.clone(),
                            _ => json!(env.new_id("call_")),
                        };
                        let name = py::or(Some(py::attr_get(&f, "name")?), &json!("")).clone();
                        out.push(Event::PartStart {
                            index: self.index,
                            kind: "tool_call",
                            id,
                            name,
                        });
                    } else {
                        out.push(Event::PartStart {
                            index: self.index,
                            kind,
                            id: Value::Null,
                            name: Value::Null,
                        });
                    }
                }
                let kind = self.open_key.as_ref().map_or("", |k| k.0);
                match kind {
                    "text" => out.push(Event::TextDelta {
                        index: self.index,
                        text: value.clone(),
                    }),
                    "reasoning" => out.push(Event::ReasoningDelta {
                        index: self.index,
                        text: value.as_str().unwrap_or_default().into(),
                        signature: Value::Null,
                    }),
                    _ => {
                        let f = py::or(Some(py::attr_get(&value, "function")?), &empty).clone();
                        let args = py::attr_get(&f, "arguments")?;
                        if py::truthy(args) {
                            out.push(Event::ToolArgsDelta {
                                index: self.index,
                                delta: args.clone(),
                            });
                        }
                    }
                }
            }
            if py::truthy(py::get(c, "finish_reason")) {
                let (s, q) = stop_reason(&choice)?;
                self.stop = s;
                self.seq = q;
            }
        }
        Ok(Flow::Continue)
    }

    fn finish(&mut self, _env: &mut dyn Env, out: &mut Vec<Event>) -> PyResult<()> {
        if !self.started {
            out.push(Event::error("Engine stream ended without a response"));
            return Ok(());
        }
        if self.open_key.is_some() {
            out.push(Event::PartStop { index: self.index });
        }
        out.push(Event::ResponseStop {
            stop_reason: self.stop,
            stop_sequence: self.seq.clone(),
            usage: self.usage.clone(),
        });
        Ok(())
    }
}

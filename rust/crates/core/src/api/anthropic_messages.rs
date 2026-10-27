//! Anthropic Messages (`swarmpilot/api/styles/anthropic_messages.py`).

use serde_json::{json, Map, Value};

use super::base::{
    as_list, as_mapping, as_str, dump_arguments, first_truthy, invalid, obj, opt_float, opt_int,
    parse_arguments, split_data_url, stop_list, unsupported, Context, Env, TrResult,
};
use super::canonical::{
    Event, Message, Part, Request, Response, Role, StopReason, Tool, ToolChoice, Usage,
};
use super::py::{self, PyResult};
use super::sse::frame;
use super::style::{Decoder, Encoder, Flow, Style};

const KNOWN: [&str; 13] = [
    "model",
    "messages",
    "system",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "tools",
    "tool_choice",
    "thinking",
    "stream",
    "metadata",
];

const MIN_THINKING_BUDGET: i128 = 1024;

fn stop_in(v: &Value) -> PyResult<StopReason> {
    Ok(match py::str_key(v)? {
        Some("max_tokens") => StopReason::MaxTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some("tool_use") => StopReason::ToolUse,
        Some("refusal") => StopReason::ContentFilter,
        _ => StopReason::EndTurn,
    })
}

fn stop_out(s: StopReason) -> &'static str {
    match s {
        StopReason::EndTurn | StopReason::Error => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::StopSequence => "stop_sequence",
        StopReason::ToolUse => "tool_use",
        StopReason::ContentFilter => "refusal",
    }
}

fn error_type_for(status: u16) -> &'static str {
    match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    }
}

fn effort_budget(v: &Value) -> PyResult<Option<i128>> {
    Ok(match py::str_key(v)? {
        Some("minimal") => Some(1024),
        Some("low") => Some(2048),
        Some("medium") => Some(8192),
        Some("high") => Some(24576),
        _ => None,
    })
}

fn text_blocks(value: &Value, where_: &str, ctx: &mut Context) -> TrResult<Vec<String>> {
    if let Value::String(s) = value {
        return Ok(vec![s.clone()]);
    }
    let mut out = Vec::new();
    for (i, b) in as_list(value, where_)?.iter().enumerate() {
        let b = as_mapping(b, &format!("{where_}[{i}]"))?;
        let kind = py::get(b, "type");
        if kind.as_str() != Some("text") {
            return Err(unsupported(
                &format!("{where_}[{i}].type"),
                &format!("block {}", py::repr(kind)),
            ));
        }
        if b.contains_key("cache_control") {
            ctx.drop_field("cache_control", false)?;
        }
        out.push(as_str(py::get(b, "text"), &format!("{where_}[{i}].text"))?.to_string());
    }
    Ok(out)
}

fn image(b: &Map<String, Value>, where_: &str) -> TrResult<Part> {
    let src = as_mapping(py::get(b, "source"), &format!("{where_}.source"))?;
    let kind = py::get(src, "type");
    match kind.as_str() {
        Some("base64") => {
            let media = as_str(
                py::get(src, "media_type"),
                &format!("{where_}.source.media_type"),
            )?;
            let data = as_str(py::get(src, "data"), &format!("{where_}.source.data"))?;
            Ok(Part::Image {
                url: format!("data:{media};base64,{data}"),
                detail: Value::Null,
            })
        }
        Some("url") => Ok(Part::Image {
            url: as_str(py::get(src, "url"), &format!("{where_}.source.url"))?.to_string(),
            detail: Value::Null,
        }),
        _ => Err(unsupported(
            &format!("{where_}.source.type"),
            &format!("source {}", py::repr(kind)),
        )),
    }
}

/// A message's parts and, separately, its tool results.
fn blocks(
    content: &Value,
    role: &str,
    where_: &str,
    ctx: &mut Context,
) -> TrResult<(Vec<Part>, Vec<Part>)> {
    if let Value::String(s) = content {
        return Ok((vec![Part::Text(s.clone())], Vec::new()));
    }
    let mut parts = Vec::new();
    let mut results = Vec::new();
    for (i, b) in as_list(content, where_)?.iter().enumerate() {
        let w = format!("{where_}[{i}]");
        let b = as_mapping(b, &w)?;
        if b.contains_key("cache_control") {
            ctx.drop_field("cache_control", false)?;
        }
        let kind = py::get(b, "type");
        match (kind.as_str(), role) {
            (Some("text"), _) => parts.push(Part::Text(
                as_str(py::get(b, "text"), &format!("{w}.text"))?.to_string(),
            )),
            (Some("image"), _) => parts.push(image(b, &w)?),
            (Some("tool_use"), "assistant") => {
                let id = as_str(py::get(b, "id"), &format!("{w}.id"))?.to_string();
                let name = as_str(py::get(b, "name"), &format!("{w}.name"))?.to_string();
                parts.push(Part::ToolCall {
                    id,
                    name,
                    arguments: Value::String(dump_arguments(py::or(b.get("input"), &json!({})))),
                });
            }
            (Some("tool_result"), "user") => {
                let raw = py::get(b, "content");
                let mut inner = Vec::new();
                if let Value::String(s) = raw {
                    inner.push(Part::Text(s.clone()));
                } else if !raw.is_null() {
                    for (j, x) in as_list(raw, &format!("{w}.content"))?.iter().enumerate() {
                        let wj = format!("{w}.content[{j}]");
                        let x = as_mapping(x, &wj)?;
                        if py::get(x, "type").as_str() == Some("image") {
                            inner.push(image(x, &wj)?);
                        } else {
                            inner.push(Part::Text(py::str_of(py::or(x.get("text"), &json!("")))));
                        }
                    }
                }
                results.push(Part::ToolResult {
                    call_id: as_str(py::get(b, "tool_use_id"), &format!("{w}.tool_use_id"))?
                        .to_string(),
                    content: inner,
                    is_error: py::truthy(py::get(b, "is_error")),
                });
            }
            (Some("thinking"), "assistant") => parts.push(Part::Reasoning {
                text: py::str_of(py::or(b.get("thinking"), &json!(""))),
                signature: py::get(b, "signature").clone(),
            }),
            (Some("redacted_thinking"), _) => ctx.drop_field("redacted_thinking", false)?,
            _ => {
                return Err(unsupported(
                    &format!("{w}.type"),
                    &format!("content block {}", py::repr(kind)),
                ))
            }
        }
    }
    Ok((parts, results))
}

fn block_out(p: &Part) -> PyResult<Option<Value>> {
    Ok(Some(match p {
        Part::Text(t) => json!({"type": "text", "text": t}),
        Part::Reasoning { text, signature } => json!({
            "type": "thinking",
            "thinking": text,
            "signature": py::or(Some(signature), &json!("")),
        }),
        Part::ToolCall {
            id,
            name,
            arguments,
        } => json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": parse_arguments(arguments)?,
        }),
        _ => return Ok(None),
    }))
}

fn image_block(url: &str) -> Value {
    match split_data_url(url) {
        Some((media, data)) => json!({
            "type": "image",
            "source": {"type": "base64", "media_type": media, "data": data},
        }),
        None => json!({"type": "image", "source": {"type": "url", "url": url}}),
    }
}

fn block_in(p: &Part, ctx: &mut Context) -> TrResult<Option<Value>> {
    Ok(match p {
        Part::Text(t) => (!t.is_empty()).then(|| json!({"type": "text", "text": t})),
        Part::Image { url, .. } => Some(image_block(url)),
        Part::ToolCall {
            id,
            name,
            arguments,
        } => Some(json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": parse_arguments(arguments)?,
        })),
        Part::ToolResult {
            call_id,
            content,
            is_error,
        } => {
            let inner: Vec<Value> = content
                .iter()
                .filter_map(|x| match x {
                    Part::Text(t) => Some(json!({"type": "text", "text": t})),
                    Part::Image { url, .. } => Some(image_block(url)),
                    _ => None,
                })
                .collect();
            let mut out = Map::new();
            out.insert("type".into(), json!("tool_result"));
            out.insert("tool_use_id".into(), json!(call_id));
            out.insert("content".into(), Value::Array(inner));
            if *is_error {
                out.insert("is_error".into(), json!(true));
            }
            Some(Value::Object(out))
        }
        Part::Reasoning { text, signature } => {
            if py::truthy(signature) {
                Some(json!({"type": "thinking", "thinking": text, "signature": signature}))
            } else {
                // Without its signature an Anthropic engine refuses the block.
                ctx.drop_field("reasoning without signature", false)?;
                None
            }
        }
    })
}

/// `_merge_usage`: counts the engine reported replace the earlier ones.
fn merge_usage(base: &Usage, u: &Value) -> Usage {
    let Value::Object(u) = u else {
        return base.clone();
    };
    Usage {
        input_tokens: py::get_or(u, "input_tokens", &base.input_tokens).clone(),
        output_tokens: py::get_or(u, "output_tokens", &base.output_tokens).clone(),
        cached_input_tokens: py::get_or(u, "cache_read_input_tokens", &base.cached_input_tokens)
            .clone(),
        reasoning_tokens: base.reasoning_tokens.clone(),
    }
}

fn usage_out(u: &Usage) -> Value {
    let zero = json!(0);
    let mut out = Map::new();
    out.insert(
        "input_tokens".into(),
        py::or(Some(&u.input_tokens), &zero).clone(),
    );
    out.insert(
        "output_tokens".into(),
        py::or(Some(&u.output_tokens), &zero).clone(),
    );
    if !u.cached_input_tokens.is_null() {
        out.insert(
            "cache_read_input_tokens".into(),
            u.cached_input_tokens.clone(),
        );
    }
    Value::Object(out)
}

fn msg_id(id: &str) -> String {
    if id.starts_with("msg_") {
        id.to_string()
    } else {
        format!("msg_{id}")
    }
}

pub struct AnthropicMessages;

impl AnthropicMessages {
    /// `_add_thinking`: turn on thinking within Anthropic's limits.
    fn add_thinking(
        body: &mut Map<String, Value>,
        budget: i128,
        client_max: Option<i128>,
        ctx: &mut Context,
    ) -> TrResult<()> {
        let mut budget = budget.max(MIN_THINKING_BUDGET);
        match client_max {
            None => {
                let cur = py::int(py::get(body, "max_tokens"))?;
                body.insert(
                    "max_tokens".into(),
                    py::int_value(cur.max(budget + ctx.default_max_tokens)),
                );
            }
            Some(m) if budget >= m => budget = m - 1,
            Some(_) => {}
        }
        if budget < MIN_THINKING_BUDGET {
            ctx.drop_field("reasoning", false)?;
            return Ok(());
        }
        body.insert(
            "thinking".into(),
            json!({"type": "enabled", "budget_tokens": py::int_value(budget)}),
        );
        for key in ["temperature", "top_k"] {
            if let Some(v) = body.get(key) {
                if !(key == "temperature" && py::eq(v, &json!(1))) {
                    body.shift_remove(key);
                    ctx.drop_field(key, false)?;
                }
            }
        }
        Ok(())
    }
}

impl Style for AnthropicMessages {
    fn name(&self) -> &'static str {
        "anthropic_messages"
    }
    fn path(&self) -> &'static str {
        "v1/messages"
    }

    fn decode_request(
        &self,
        body: &Map<String, Value>,
        ctx: &mut Context,
        _env: &mut dyn Env,
    ) -> TrResult<Request> {
        let mut req = Request {
            model: py::get(body, "model").clone(),
            ..Request::default()
        };
        let system = py::get(body, "system");
        if !system.is_null() {
            req.system = text_blocks(system, "system", ctx)?;
        }
        for (i, m) in as_list(py::get(body, "messages"), "messages")?
            .iter()
            .enumerate()
        {
            let w = format!("messages[{i}]");
            let m = as_mapping(m, &w)?;
            let role_v = py::get(m, "role");
            let role = match role_v.as_str() {
                Some(r @ ("user" | "assistant")) => r,
                _ => {
                    return Err(unsupported(
                        &format!("{w}.role"),
                        &format!("role {}", py::repr(role_v)),
                    ))
                }
            };
            let (parts, results) = blocks(py::get(m, "content"), role, &w, ctx)?;
            let had_results = !results.is_empty();
            // Tool results lead the user turn that carries them.
            for r in results {
                req.messages.push(Message::new(Role::Tool, vec![r]));
            }
            if !parts.is_empty() || !had_results {
                let role = if role == "user" {
                    Role::User
                } else {
                    Role::Assistant
                };
                req.messages.push(Message::new(role, parts));
            }
        }
        req.max_output_tokens = opt_int(body, "max_tokens")?;
        if req.max_output_tokens.is_none() {
            return Err(invalid("'max_tokens' is required", Some("max_tokens")));
        }
        req.temperature = opt_float(body, "temperature")?;
        req.top_p = opt_float(body, "top_p")?;
        req.top_k = opt_int(body, "top_k")?;
        req.stop = stop_list(py::get(body, "stop_sequences"), "stop_sequences")?;
        for (j, t) in py::iter_or_empty(py::get(body, "tools"))?
            .iter()
            .enumerate()
        {
            let t = as_mapping(t, &format!("tools[{j}]"))?;
            let kind = py::get(t, "type");
            if !(kind.is_null() || kind.as_str() == Some("custom")) {
                return Err(unsupported(
                    &format!("tools[{j}].type"),
                    &format!("server tool {}", py::repr(kind)),
                ));
            }
            if t.contains_key("cache_control") {
                ctx.drop_field("cache_control", false)?;
            }
            req.tools.push(Tool {
                name: as_str(py::get(t, "name"), &format!("tools[{j}].name"))?.to_string(),
                description: py::get(t, "description").clone(),
                parameters: py::or(
                    t.get("input_schema"),
                    &json!({"type": "object", "properties": {}}),
                )
                .clone(),
                strict: Value::Null,
            });
        }
        let tc = py::get(body, "tool_choice");
        if !tc.is_null() {
            let tc = as_mapping(tc, "tool_choice")?;
            let mode = match py::str_key(py::get(tc, "type"))? {
                Some("auto") => "auto",
                Some("any") => "required",
                Some("tool") => "tool",
                Some("none") => "none",
                _ => return Err(invalid("invalid 'tool_choice'", Some("tool_choice"))),
            };
            if mode == "tool" {
                as_str(py::get(tc, "name"), "tool_choice.name")?;
            }
            req.tool_choice = Some(ToolChoice {
                mode: mode.into(),
                name: py::get(tc, "name").clone(),
            });
            let dp = py::get(tc, "disable_parallel_tool_use");
            if !dp.is_null() {
                req.parallel_tool_calls = Value::Bool(!py::truthy(dp));
            }
        }
        if let Value::Object(th) = py::get(body, "thinking") {
            if py::get(th, "type").as_str() == Some("enabled") {
                req.reasoning_budget = opt_int(th, "budget_tokens")?;
            }
        }
        if let Value::Object(meta) = py::get(body, "metadata") {
            if let u @ Value::String(_) = py::get(meta, "user_id") {
                req.user = u.clone();
            }
        }
        req.stream = py::get(body, "stream") == &Value::Bool(true);
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
        _env: &mut dyn Env,
    ) -> PyResult<Value> {
        let mut content = Vec::new();
        for p in &resp.output {
            if let Some(b) = block_out(p)? {
                content.push(b);
            }
        }
        Ok(obj([
            ("id", json!(msg_id(&resp.id))),
            ("type", json!("message")),
            ("role", json!("assistant")),
            ("model", first_truthy(&[&resp.model, &req.model])),
            ("content", Value::Array(content)),
            ("stop_reason", json!(stop_out(resp.stop_reason))),
            ("stop_sequence", resp.stop_sequence.clone()),
            ("usage", usage_out(&resp.usage)),
        ]))
    }

    fn encoder(&self, req: &Request, _env: &mut dyn Env) -> Box<dyn Encoder> {
        Box::new(MessagesEncoder {
            model: req.model.clone(),
        })
    }

    fn encode_error(
        &self,
        status: u16,
        message: &str,
        error_type: Option<&str>,
        _param: Option<&str>,
    ) -> Value {
        let mut kind = error_type_for(status);
        if status == 400 && error_type == Some("invalid_request_error") {
            kind = "invalid_request_error";
        }
        json!({"type": "error", "error": {"type": kind, "message": message}})
    }

    fn encode_request(&self, req: &Request, ctx: &mut Context) -> TrResult<(String, Value)> {
        let mut messages: Vec<(Role, Vec<Value>)> = Vec::new();
        for m in &req.messages {
            let role = if m.role == Role::Tool {
                Role::User
            } else {
                m.role
            };
            let mut blocks = Vec::new();
            for p in &m.content {
                if let Some(b) = block_in(p, ctx)? {
                    blocks.push(b);
                }
            }
            if blocks.is_empty() {
                // Anthropic refuses empty turns; neighbours of the same
                // role merge below.
                continue;
            }
            match messages.last_mut() {
                Some((r, content)) if *r == role => content.extend(blocks),
                _ => messages.push((role, blocks)),
            }
        }
        let messages: Vec<Value> = messages
            .into_iter()
            .map(|(r, content)| json!({"role": r.name(), "content": content}))
            .collect();
        let mut body = Map::new();
        body.insert("messages".into(), Value::Array(messages));
        let max = match req.max_output_tokens {
            Some(n) if n != 0 => n,
            _ => ctx.default_max_tokens,
        };
        body.insert("max_tokens".into(), py::int_value(max));
        if !req.model.is_null() {
            body.insert("model".into(), req.model.clone());
        }
        if !req.system.is_empty() {
            body.insert("system".into(), json!(req.system_text()));
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
        if req.temperature.is_some_and(|t| t > 1.0) {
            // OpenAI styles allow 0-2, Anthropic 0-1.
            body.insert("temperature".into(), py::float_value(1.0));
            ctx.drop_field("temperature above 1", false)?;
        }
        if !req.stop.is_empty() {
            body.insert("stop_sequences".into(), json!(req.stop));
        }
        if req.seed.is_some() {
            ctx.drop_field("seed", false)?;
        }
        if !req.tools.is_empty() {
            let tools = req
                .tools
                .iter()
                .map(|t| {
                    let mut o = Map::new();
                    o.insert("name".into(), json!(t.name));
                    if !t.description.is_null() {
                        o.insert("description".into(), t.description.clone());
                    }
                    if !t.parameters.is_null() {
                        o.insert("input_schema".into(), t.parameters.clone());
                    }
                    Value::Object(o)
                })
                .collect();
            body.insert("tools".into(), Value::Array(tools));
        }
        let no_parallel = req.parallel_tool_calls == Value::Bool(false);
        if req.tool_choice.is_some() || no_parallel {
            let mode = req
                .tool_choice
                .as_ref()
                .map_or("auto", |tc| tc.mode.as_str());
            let mut out = Map::new();
            out.insert(
                "type".into(),
                json!(if mode == "required" { "any" } else { mode }),
            );
            if mode == "tool" {
                out.insert(
                    "name".into(),
                    req.tool_choice
                        .as_ref()
                        .map_or(Value::Null, |tc| tc.name.clone()),
                );
            }
            if no_parallel && mode != "none" {
                out.insert("disable_parallel_tool_use".into(), json!(true));
            }
            body.insert("tool_choice".into(), Value::Object(out));
        }
        if req
            .response_format
            .as_ref()
            .is_some_and(|rf| rf.kind != "text")
        {
            ctx.drop_field("response_format", true)?;
        }
        let mut budget = req.reasoning_budget;
        if budget.is_none() {
            budget = effort_budget(&req.reasoning_effort)?;
        }
        if let Some(b) = budget {
            Self::add_thinking(&mut body, b, req.max_output_tokens, ctx)?;
        }
        if py::truthy(&req.user) {
            body.insert("metadata".into(), json!({"user_id": req.user}));
        }
        if req.stream {
            body.insert("stream".into(), json!(true));
        }
        Ok((self.path().into(), Value::Object(body)))
    }

    fn decode_response(&self, body: &Map<String, Value>, env: &mut dyn Env) -> PyResult<Response> {
        let mut output = Vec::new();
        let empty = json!({});
        let blank = json!("");
        for b in py::iter_or_empty(py::get(body, "content"))? {
            match py::attr_get(&b, "type")?.as_str() {
                Some("text") => output.push(Part::Text(py::str_of(py::or(
                    Some(py::attr_get(&b, "text")?),
                    &blank,
                )))),
                Some("thinking") => output.push(Part::Reasoning {
                    text: py::str_of(py::or(Some(py::attr_get(&b, "thinking")?), &blank)),
                    signature: py::attr_get(&b, "signature")?.clone(),
                }),
                Some("tool_use") => {
                    let id = match py::attr_get(&b, "id")? {
                        v if py::truthy(v) => py::str_of(v),
                        _ => env.new_id("toolu_"),
                    };
                    let name = py::str_of(py::or(Some(py::attr_get(&b, "name")?), &blank));
                    let input = py::or(Some(py::attr_get(&b, "input")?), &empty);
                    output.push(Part::ToolCall {
                        id,
                        name,
                        arguments: Value::String(dump_arguments(input)),
                    });
                }
                _ => {}
            }
        }
        let id = match py::get(body, "id") {
            v if py::truthy(v) => py::str_of(v),
            _ => env.new_id("msg_"),
        };
        Ok(Response {
            id,
            model: py::get(body, "model").clone(),
            output,
            stop_reason: stop_in(py::get(body, "stop_reason"))?,
            stop_sequence: py::get(body, "stop_sequence").clone(),
            usage: merge_usage(&Usage::default(), py::get(body, "usage")),
            created: 0,
        })
    }

    fn decoder(&self) -> Box<dyn Decoder> {
        Box::new(MessagesDecoder {
            usage: Usage::default(),
            stop: StopReason::EndTurn,
            seq: Value::Null,
            kinds: Vec::new(),
            offset: None,
            started: false,
        })
    }
}

fn delta(index: i128, d: Value) -> Vec<u8> {
    frame(
        &obj([
            ("type", json!("content_block_delta")),
            ("index", py::int_value(index)),
            ("delta", d),
        ]),
        Some("content_block_delta"),
    )
}

struct MessagesEncoder {
    model: Value,
}

impl Encoder for MessagesEncoder {
    fn on(&mut self, ev: &Event, _env: &mut dyn Env, out: &mut Vec<u8>) -> PyResult<bool> {
        match ev {
            Event::ResponseStart { id, model, .. } => out.extend(frame(
                &json!({
                    "type": "message_start",
                    "message": {
                        "id": msg_id(id),
                        "type": "message",
                        "role": "assistant",
                        "model": first_truthy(&[model, &self.model]),
                        "content": [],
                        "stop_reason": null,
                        "stop_sequence": null,
                        // The OpenAI engines report usage only at the end.
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }),
                Some("message_start"),
            )),
            Event::PartStart {
                index,
                kind,
                id,
                name,
            } => {
                let block = match *kind {
                    "text" => json!({"type": "text", "text": ""}),
                    "reasoning" => json!({"type": "thinking", "thinking": "", "signature": ""}),
                    _ => json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
                };
                out.extend(frame(
                    &obj([
                        ("type", json!("content_block_start")),
                        ("index", py::int_value(*index)),
                        ("content_block", block),
                    ]),
                    Some("content_block_start"),
                ));
            }
            Event::TextDelta { index, text } => {
                out.extend(delta(*index, json!({"type": "text_delta", "text": text})))
            }
            Event::ToolArgsDelta { index, delta: d } => out.extend(delta(
                *index,
                json!({"type": "input_json_delta", "partial_json": d}),
            )),
            Event::ReasoningDelta {
                index,
                text,
                signature,
            } => {
                if !text.is_empty() {
                    out.extend(delta(
                        *index,
                        json!({"type": "thinking_delta", "thinking": text}),
                    ));
                }
                if py::truthy(signature) {
                    out.extend(delta(
                        *index,
                        json!({"type": "signature_delta", "signature": signature}),
                    ));
                }
            }
            Event::PartStop { index } => out.extend(frame(
                &obj([
                    ("type", json!("content_block_stop")),
                    ("index", py::int_value(*index)),
                ]),
                Some("content_block_stop"),
            )),
            Event::ResponseStop {
                stop_reason,
                stop_sequence,
                usage,
            } => {
                out.extend(frame(
                    &json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_out(*stop_reason), "stop_sequence": stop_sequence},
                        "usage": usage_out(usage),
                    }),
                    Some("message_delta"),
                ));
                out.extend(frame(
                    &json!({"type": "message_stop"}),
                    Some("message_stop"),
                ));
                return Ok(true);
            }
            Event::Error { message, .. } => {
                out.extend(frame(
                    &AnthropicMessages.encode_error(500, message, None, None),
                    Some("error"),
                ));
                return Ok(true);
            }
        }
        Ok(false)
    }
}

struct MessagesDecoder {
    usage: Usage,
    stop: StopReason,
    seq: Value,
    /// Part indexes opened so far.
    kinds: Vec<i128>,
    offset: Option<i128>,
    started: bool,
}

impl Decoder for MessagesDecoder {
    fn on_data(&mut self, data: &str, env: &mut dyn Env, out: &mut Vec<Event>) -> PyResult<Flow> {
        let Some(o) = py::loads(data) else {
            return Ok(Flow::Continue);
        };
        let empty = json!({});
        let blank = json!("");
        let zero = json!(0);
        let index_of = |o: &Value| -> PyResult<i128> {
            match o {
                Value::Object(m) => py::int(py::get_or(m, "index", &zero)),
                _ => unreachable!(),
            }
        };
        match py::attr_get(&o, "type")?.as_str() {
            Some("message_start") => {
                let msg = py::or(Some(py::attr_get(&o, "message")?), &empty);
                self.usage = merge_usage(&self.usage, py::attr_get(msg, "usage")?);
                self.started = true;
                let id = match py::attr_get(msg, "id")? {
                    v if py::truthy(v) => py::str_of(v),
                    _ => env.new_id("msg_"),
                };
                out.push(Event::ResponseStart {
                    id,
                    model: py::attr_get(msg, "model")?.clone(),
                    created: 0,
                });
            }
            Some("content_block_start") => {
                let b = py::or(Some(py::attr_get(&o, "content_block")?), &empty);
                let raw = index_of(&o)?;
                let offset = *self.offset.get_or_insert(raw);
                let idx = raw - offset;
                let t = py::attr_get(b, "type")?;
                match t.as_str() {
                    Some("tool_use") => {
                        self.kinds.push(idx);
                        out.push(Event::PartStart {
                            index: idx,
                            kind: "tool_call",
                            id: py::attr_get(b, "id")?.clone(),
                            name: py::attr_get(b, "name")?.clone(),
                        });
                    }
                    Some("thinking") => {
                        self.kinds.push(idx);
                        out.push(Event::PartStart {
                            index: idx,
                            kind: "reasoning",
                            id: Value::Null,
                            name: Value::Null,
                        });
                    }
                    Some("text") => {
                        self.kinds.push(idx);
                        out.push(Event::PartStart {
                            index: idx,
                            kind: "text",
                            id: Value::Null,
                            name: Value::Null,
                        });
                        let text = py::attr_get(b, "text")?;
                        if py::truthy(text) {
                            out.push(Event::TextDelta {
                                index: idx,
                                text: text.clone(),
                            });
                        }
                    }
                    _ => {}
                }
            }
            Some("content_block_delta") => {
                let idx = index_of(&o)? - self.offset.unwrap_or(0);
                let d = py::or(Some(py::attr_get(&o, "delta")?), &empty);
                let dt = py::attr_get(d, "type")?;
                if !self.kinds.contains(&idx) {
                    return Ok(Flow::Continue);
                }
                let text_of = |key: &str| -> PyResult<String> {
                    Ok(py::str_of(py::or(Some(py::attr_get(d, key)?), &blank)))
                };
                match dt.as_str() {
                    Some("text_delta") => out.push(Event::TextDelta {
                        index: idx,
                        text: Value::String(text_of("text")?),
                    }),
                    Some("input_json_delta") => out.push(Event::ToolArgsDelta {
                        index: idx,
                        delta: Value::String(text_of("partial_json")?),
                    }),
                    Some("thinking_delta") => out.push(Event::ReasoningDelta {
                        index: idx,
                        text: text_of("thinking")?,
                        signature: Value::Null,
                    }),
                    Some("signature_delta") => out.push(Event::ReasoningDelta {
                        index: idx,
                        text: String::new(),
                        signature: py::attr_get(d, "signature")?.clone(),
                    }),
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                let idx = index_of(&o)? - self.offset.unwrap_or(0);
                if self.kinds.contains(&idx) {
                    out.push(Event::PartStop { index: idx });
                }
            }
            Some("message_delta") => {
                let d = py::or(Some(py::attr_get(&o, "delta")?), &empty);
                let sr = py::attr_get(d, "stop_reason")?;
                if py::truthy(sr) {
                    self.stop = stop_in(sr)?;
                    self.seq = py::attr_get(d, "stop_sequence")?.clone();
                }
                self.usage = merge_usage(&self.usage, py::attr_get(&o, "usage")?);
            }
            Some("message_stop") => return Ok(Flow::Break),
            Some("error") => {
                let e = py::or(Some(py::attr_get(&o, "error")?), &empty);
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
        out.push(Event::ResponseStop {
            stop_reason: self.stop,
            stop_sequence: self.seq.clone(),
            usage: self.usage.clone(),
        });
        Ok(())
    }
}

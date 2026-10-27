//! The canonical interaction (`swarmpilot/api/canonical.py`).
//!
//! Fields the Python styles copy from a body without checking its type
//! (`model`, a tool's `description`, usage counts, ...) are kept as JSON
//! values here, so whatever the client or engine sent travels on exactly
//! as in Python.

use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    ContentFilter,
    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    Text(String),
    Image {
        url: String,
        detail: Value,
    },
    ToolCall {
        id: String,
        name: String,
        /// JSON text as generated; the chat decoder may leave another
        /// JSON value here, as Python does.
        arguments: Value,
    },
    ToolResult {
        call_id: String,
        content: Vec<Part>,
        is_error: bool,
    },
    Reasoning {
        text: String,
        signature: Value,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    Tool,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Part>,
}

impl Message {
    pub fn new(role: Role, content: Vec<Part>) -> Self {
        Message { role, content }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: Value,
    pub parameters: Value,
    pub strict: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolChoice {
    /// `auto`, `none`, `required` or `tool`.
    pub mode: String,
    pub name: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResponseFormat {
    /// `text`, `json_object` or `json_schema`.
    pub kind: String,
    pub name: Value,
    pub schema: Value,
    pub strict: Value,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Request {
    pub messages: Vec<Message>,
    pub model: Value,
    pub system: Vec<String>,
    pub max_output_tokens: Option<i128>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i128>,
    pub stop: Vec<String>,
    pub seed: Option<i128>,
    pub tools: Vec<Tool>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Value,
    pub response_format: Option<ResponseFormat>,
    pub reasoning_effort: Value,
    pub reasoning_budget: Option<i128>,
    pub stream: bool,
    pub user: Value,
    /// Style name to the body fields with no canonical home, in body order.
    pub extensions: Vec<(String, Map<String, Value>)>,
}

impl Request {
    pub fn system_text(&self) -> String {
        self.system.join("\n\n")
    }

    pub fn extension(&self, style: &str) -> Option<&Map<String, Value>> {
        self.extensions
            .iter()
            .find(|(n, _)| n == style)
            .map(|(_, m)| m)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Usage {
    pub input_tokens: Value,
    pub output_tokens: Value,
    pub cached_input_tokens: Value,
    pub reasoning_tokens: Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Response {
    pub id: String,
    pub model: Value,
    pub output: Vec<Part>,
    pub stop_reason: StopReason,
    pub stop_sequence: Value,
    pub usage: Usage,
    pub created: i128,
}

/// Stream events; parts are numbered from 0 in output order.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    ResponseStart {
        id: String,
        model: Value,
        created: i128,
    },
    PartStart {
        index: i128,
        /// `text`, `tool_call` or `reasoning`.
        kind: &'static str,
        id: Value,
        name: Value,
    },
    TextDelta {
        index: i128,
        /// A string, except when an Anthropic engine opens a text block
        /// with a non-string `text`, which Python passes on as is.
        text: Value,
    },
    ToolArgsDelta {
        index: i128,
        delta: Value,
    },
    ReasoningDelta {
        index: i128,
        text: String,
        signature: Value,
    },
    PartStop {
        index: i128,
    },
    ResponseStop {
        stop_reason: StopReason,
        stop_sequence: Value,
        usage: Usage,
    },
    Error {
        message: String,
        error_type: String,
    },
}

impl Event {
    pub fn error(message: impl Into<String>) -> Self {
        Event::Error {
            message: message.into(),
            error_type: "api_error".into(),
        }
    }
}

/// `events_from_response`: the stream equivalent to a complete answer.
pub fn events_from_response(resp: &Response) -> Vec<Event> {
    let mut out = vec![Event::ResponseStart {
        id: resp.id.clone(),
        model: resp.model.clone(),
        created: resp.created,
    }];
    for (i, p) in resp.output.iter().enumerate() {
        let i = i as i128;
        match p {
            Part::Text(t) => {
                out.push(Event::PartStart {
                    index: i,
                    kind: "text",
                    id: Value::Null,
                    name: Value::Null,
                });
                out.push(Event::TextDelta {
                    index: i,
                    text: Value::String(t.clone()),
                });
            }
            Part::Reasoning { text, signature } => {
                out.push(Event::PartStart {
                    index: i,
                    kind: "reasoning",
                    id: Value::Null,
                    name: Value::Null,
                });
                out.push(Event::ReasoningDelta {
                    index: i,
                    text: text.clone(),
                    signature: signature.clone(),
                });
            }
            Part::ToolCall {
                id,
                name,
                arguments,
            } => {
                out.push(Event::PartStart {
                    index: i,
                    kind: "tool_call",
                    id: Value::String(id.clone()),
                    name: Value::String(name.clone()),
                });
                out.push(Event::ToolArgsDelta {
                    index: i,
                    delta: arguments.clone(),
                });
            }
            _ => continue,
        }
        out.push(Event::PartStop { index: i });
    }
    out.push(Event::ResponseStop {
        stop_reason: resp.stop_reason,
        stop_sequence: resp.stop_sequence.clone(),
        usage: resp.usage.clone(),
    });
    out
}

//! Shared pieces of the style interface (`swarmpilot/api/base.py` and
//! `swarmpilot/api/styles/_common.py`).

use serde_json::{Map, Value};

use super::py::{self, PyErr};

/// Output limit sent to engines whose style needs one when the client set
/// none.
pub const DEFAULT_MAX_OUTPUT_TOKENS: i128 = 4096;

/// Why a request or answer could not be translated.
#[derive(Debug, Clone, PartialEq)]
pub enum TrError {
    /// `TranslationError`: answered with this status and message.
    Translation {
        message: String,
        param: Option<String>,
        status: u16,
        error_type: String,
    },
    /// Any other Python exception: a 500.
    Py(PyErr),
}

impl From<PyErr> for TrError {
    fn from(e: PyErr) -> Self {
        TrError::Py(e)
    }
}

pub type TrResult<T> = Result<T, TrError>;

/// A 400 naming `param`.
pub fn invalid(message: impl Into<String>, param: Option<&str>) -> TrError {
    TrError::Translation {
        message: message.into(),
        param: param.map(str::to_string),
        status: 400,
        error_type: "invalid_request_error".into(),
    }
}

/// `unsupported(name, why)`.
pub fn unsupported(name: &str, why: &str) -> TrError {
    let reason = if why.is_empty() {
        String::new()
    } else {
        format!(": {why}")
    };
    invalid(format!("'{name}' is not supported{reason}"), Some(name))
}

/// Ids and the clock, replaceable so tests can compare runs exactly.
pub trait Env {
    /// `new_id(prefix)`: the prefix and 24 random hex digits.
    fn new_id(&mut self, prefix: &str) -> String;
    /// `now()`: Unix time in whole seconds.
    fn now(&mut self) -> i128;
}

/// Random ids and the system clock.
pub struct SystemEnv;

impl Env for SystemEnv {
    fn new_id(&mut self, prefix: &str) -> String {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        let mut h = RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        let a = h.finish();
        let mut h2 = RandomState::new().build_hasher();
        h2.write_u64(a);
        let b = h2.finish();
        format!("{prefix}{a:016x}{:08x}", b >> 32)
    }

    fn now(&mut self) -> i128 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| i128::from(d.as_secs()))
    }
}

/// Per-request translation state (`Context`).
#[derive(Clone, Debug)]
pub struct Context {
    pub strict: bool,
    pub dropped: Vec<String>,
    pub default_max_tokens: i128,
}

impl Context {
    pub fn new(strict: bool, default_max_tokens: i128) -> Self {
        Context {
            strict,
            dropped: Vec::new(),
            default_max_tokens,
        }
    }

    /// `ctx.drop(name, semantic=...)`.
    pub fn drop_field(&mut self, name: &str, semantic: bool) -> TrResult<()> {
        if semantic && self.strict {
            return Err(invalid(
                format!("'{name}' is not supported for this model's engine"),
                Some(name),
            ));
        }
        if !self.dropped.iter().any(|d| d == name) {
            self.dropped.push(name.to_string());
        }
        Ok(())
    }
}

pub fn as_mapping<'a>(v: &'a Value, name: &str) -> TrResult<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| invalid(format!("'{name}' must be an object"), Some(name)))
}

pub fn as_list<'a>(v: &'a Value, name: &str) -> TrResult<&'a Vec<Value>> {
    v.as_array()
        .ok_or_else(|| invalid(format!("'{name}' must be an array"), Some(name)))
}

pub fn as_str<'a>(v: &'a Value, name: &str) -> TrResult<&'a str> {
    v.as_str()
        .ok_or_else(|| invalid(format!("'{name}' must be a string"), Some(name)))
}

/// `opt_int`: None, or an int (bools refused).
pub fn opt_int(body: &Map<String, Value>, key: &str) -> TrResult<Option<i128>> {
    let v = py::get(body, key);
    if v.is_null() {
        return Ok(None);
    }
    py::as_int(v)
        .map(Some)
        .ok_or_else(|| invalid(format!("'{key}' must be an integer"), Some(key)))
}

/// `opt_float`: None, or a finite int or float, as a float.
pub fn opt_float(body: &Map<String, Value>, key: &str) -> TrResult<Option<f64>> {
    let v = py::get(body, key);
    if v.is_null() {
        return Ok(None);
    }
    match v {
        Value::Number(n) => match py::int_of(n) {
            Some(i) => Ok(Some(i as f64)),
            None => n
                .as_f64()
                .filter(|f| f.is_finite())
                .map(Some)
                .ok_or_else(|| invalid(format!("'{key}' must be a number"), Some(key))),
        },
        _ => Err(invalid(format!("'{key}' must be a number"), Some(key))),
    }
}

/// `stop_list`: a string or a list of strings.
pub fn stop_list(v: &Value, name: &str) -> TrResult<Vec<String>> {
    match v {
        Value::Null => Ok(Vec::new()),
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(a) if a.iter().all(Value::is_string) => Ok(a
            .iter()
            .map(|s| s.as_str().unwrap_or_default().to_string())
            .collect()),
        _ => Err(invalid(
            format!("'{name}' must be a string or an array of strings"),
            Some(name),
        )),
    }
}

/// `parse_arguments`: tool-call arguments as an object, invalid JSON kept
/// under `_raw`.
pub fn parse_arguments(arguments: &Value) -> Result<Value, PyErr> {
    if !py::truthy(arguments) {
        return Ok(Value::Object(Map::new()));
    }
    let Value::String(text) = arguments else {
        return py::err(format!(
            "the JSON object must be str, bytes or bytearray, not {}",
            py::type_name(arguments)
        ));
    };
    let raw = || {
        let mut m = Map::new();
        m.insert("_raw".into(), arguments.clone());
        Value::Object(m)
    };
    match py::loads(text) {
        Some(v @ Value::Object(_)) => Ok(v),
        _ => Ok(raw()),
    }
}

/// `dump_arguments`: compact JSON text.
pub fn dump_arguments(v: &Value) -> String {
    py::dumps(v)
}

/// `split_data_url`: `(media_type, data)` of a base64 `data:` URL.
pub fn split_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (head, data) = rest.split_once(',')?;
    let media = head.strip_suffix(";base64")?;
    let media = if media.is_empty() {
        "application/octet-stream"
    } else {
        media
    };
    Some((media.to_string(), data.to_string()))
}

/// `text_of`: a string, or the joined `text` of a list of parts.
pub fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter_map(|p| p.as_object()?.get("text")?.as_str())
            .collect(),
        _ => String::new(),
    }
}

/// Build a JSON object from `(key, value)` pairs in order.
pub fn obj<const N: usize>(pairs: [(&str, Value); N]) -> Value {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    Value::Object(m)
}

/// `a or b or ""` over JSON values.
pub fn first_truthy(vals: &[&Value]) -> Value {
    vals.iter()
        .find(|v| py::truthy(v))
        .map_or(Value::String(String::new()), |v| (*v).clone())
}

/// Deterministic ids (`<prefix><counter as 24 hex digits>`) and a fixed
/// clock, for comparing runs against Python with the same stand-ins.
pub struct CountingEnv {
    pub next: u64,
    pub now: i128,
}

impl Env for CountingEnv {
    fn new_id(&mut self, prefix: &str) -> String {
        self.next += 1;
        format!("{prefix}{:024x}", self.next)
    }

    fn now(&mut self) -> i128 {
        self.now
    }
}

//! Predictor features of a request (`_extract_features` of the proxy).

use std::collections::HashMap;

use serde_json::Value;

use crate::state::{Adapter, Features};

/// The adapter's features for one request body; empty when the body is
/// malformed for them, as in Python.
pub fn extract(adapter: &Adapter, body: &Value) -> HashMap<String, f64> {
    match &adapter.features {
        Features::Extractors(list) => list
            .iter()
            .map(|(name, x)| (name.clone(), x.eval(&body)))
            .collect(),
        Features::Passthrough => passthrough(body).unwrap_or_default(),
        Features::None => HashMap::new(),
    }
}

/// `len()` of a JSON value as Python sees it; None where Python raises.
fn py_len(v: &Value) -> Option<f64> {
    match v {
        Value::String(s) => Some(s.chars().count() as f64),
        Value::Array(a) => Some(a.len() as f64),
        Value::Object(o) => Some(o.len() as f64),
        _ => None,
    }
}

/// `PassthroughAdapter.features`. A non-numeric `max_tokens` is left out
/// (Python keeps the raw value, which no model can use).
fn passthrough(body: &Value) -> Option<HashMap<String, f64>> {
    let mut out = HashMap::new();
    let Some(obj) = body.as_object() else {
        return Some(out);
    };
    match obj.get("max_tokens") {
        Some(Value::Number(n)) => {
            out.insert("max_tokens".into(), n.as_f64()?);
        }
        Some(Value::Bool(b)) => {
            out.insert("max_tokens".into(), f64::from(u8::from(*b)));
        }
        _ => {}
    }
    match obj.get("messages") {
        Some(Value::Array(messages)) => {
            let mut total = 0.0;
            for m in messages {
                if let Some(m) = m.as_object() {
                    total += match m.get("content") {
                        None => 0.0,
                        Some(c) => py_len(c)?,
                    };
                }
            }
            out.insert("prompt_length".into(), total);
        }
        _ => {
            if let Some(p) = obj.get("prompt") {
                out.insert("prompt_length".into(), py_len(p)?);
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn passthrough_follows_python() {
        let f = |v: Value| passthrough(&v);
        let got = f(json!({"max_tokens": 7, "messages": [{"content": "héllo"}, "x", {}]})).unwrap();
        assert_eq!(got["max_tokens"], 7.0);
        assert_eq!(got["prompt_length"], 5.0);
        assert_eq!(
            f(json!({"prompt": ["a", "b"]})).unwrap()["prompt_length"],
            2.0
        );
        // len(5) raises in Python, so the whole dict is empty.
        assert!(f(json!({"prompt": 5})).is_none());
        assert!(f(json!({"messages": [{"content": 3}]})).is_none());
    }
}

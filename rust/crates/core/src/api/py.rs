//! Python semantics the translation layer relies on.
//!
//! The Python styles read loosely typed JSON with `dict.get`, truthiness,
//! `str()`, `int()`, `repr()` in error messages and `json.dumps` on the way
//! out. These helpers reproduce that behaviour on `serde_json::Value`, so a
//! value of an unexpected type gives the same result (or the same failure)
//! as in Python.

use serde_json::{Map, Number, Value};

/// An exception Python would raise and nothing catches: the request ends
/// with a 500 (or a broken stream).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct PyErr(pub String);

pub type PyResult<T> = Result<T, PyErr>;

pub fn err<T>(msg: impl Into<String>) -> PyResult<T> {
    Err(PyErr(msg.into()))
}

/// `bool(v)`.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => {
            if let Some(i) = int_of(n) {
                i != 0
            } else {
                n.as_f64().is_some_and(|f| f != 0.0)
            }
        }
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The value of a JSON integer (Python `int`), None for floats.
pub fn int_of(n: &Number) -> Option<i128> {
    if let Some(i) = n.as_i64() {
        Some(i128::from(i))
    } else {
        n.as_u64().map(i128::from)
    }
}

/// `isinstance(v, int) and not isinstance(v, bool)`.
pub fn as_int(v: &Value) -> Option<i128> {
    match v {
        Value::Number(n) => int_of(n),
        _ => None,
    }
}

/// `v or default`: `v` when truthy.
pub fn or<'a>(v: Option<&'a Value>, default: &'a Value) -> &'a Value {
    match v {
        Some(x) if truthy(x) => x,
        _ => default,
    }
}

/// `d.get(key)` on a mapping; `None` (missing) and JSON null look alike.
pub fn get<'a>(d: &'a Map<String, Value>, key: &str) -> &'a Value {
    d.get(key).unwrap_or(&Value::Null)
}

/// `obj.get(key)` where `obj` may not be a mapping: Python raises
/// `AttributeError` on lists, strings and numbers.
pub fn attr_get<'a>(obj: &'a Value, key: &str) -> PyResult<&'a Value> {
    match obj {
        Value::Object(o) => Ok(get(o, key)),
        _ => err(format!(
            "'{}' object has no attribute 'get'",
            type_name(obj)
        )),
    }
}

pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) => {
            if int_of(n).is_some() {
                "int"
            } else {
                "float"
            }
        }
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// Python `==`: bools and numbers compare by value across types.
pub fn eq(a: &Value, b: &Value) -> bool {
    match (num(a), num(b)) {
        (Some(x), Some(y)) => num_eq(x, y),
        (None, None) => match (a, b) {
            (Value::Null, Value::Null) => true,
            (Value::String(x), Value::String(y)) => x == y,
            (Value::Array(x), Value::Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| eq(p, q))
            }
            (Value::Object(x), Value::Object(y)) => {
                x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| eq(v, w)))
            }
            _ => false,
        },
        _ => false,
    }
}

#[derive(Clone, Copy)]
enum Num {
    Int(i128),
    Float(f64),
}

fn num(v: &Value) -> Option<Num> {
    match v {
        Value::Bool(b) => Some(Num::Int(i128::from(*b))),
        Value::Number(n) => Some(match int_of(n) {
            Some(i) => Num::Int(i),
            None => Num::Float(n.as_f64().unwrap_or(f64::NAN)),
        }),
        _ => None,
    }
}

fn num_eq(a: Num, b: Num) -> bool {
    match (a, b) {
        (Num::Int(x), Num::Int(y)) => x == y,
        (Num::Float(x), Num::Float(y)) => x == y,
        (Num::Int(i), Num::Float(f)) | (Num::Float(f), Num::Int(i)) => {
            f.is_finite() && f.fract() == 0.0 && (f as i128) == i && (i as f64) == f
        }
    }
}

/// `v in (choices...)` for string choices.
pub fn is_one_of(v: &Value, choices: &[&str]) -> bool {
    v.as_str().is_some_and(|s| choices.contains(&s))
}

/// The key of a `dict.get(v)` lookup in a dict with string keys:
/// unhashable lists and dicts raise `TypeError`, other non-strings miss.
pub fn str_key(v: &Value) -> PyResult<Option<&str>> {
    match v {
        Value::Array(_) | Value::Object(_) => err(format!("unhashable type: '{}'", type_name(v))),
        Value::String(s) => Ok(Some(s)),
        _ => Ok(None),
    }
}

/// `for x in (v or [])`: lists give their items, dicts their keys,
/// strings their characters; other truthy values are not iterable.
pub fn iter_or_empty(v: &Value) -> PyResult<Vec<Value>> {
    if !truthy(v) {
        return Ok(Vec::new());
    }
    match v {
        Value::Array(a) => Ok(a.clone()),
        Value::Object(o) => Ok(o.keys().map(|k| Value::String(k.clone())).collect()),
        Value::String(s) => Ok(s.chars().map(|c| Value::String(c.to_string())).collect()),
        _ => err(format!("'{}' object is not iterable", type_name(v))),
    }
}

/// `d.get(key, default)`: the default only when the key is missing.
pub fn get_or<'a>(d: &'a Map<String, Value>, key: &str, default: &'a Value) -> &'a Value {
    d.get(key).unwrap_or(default)
}

/// `key in v` for a string key.
pub fn contains(v: &Value, key: &str) -> PyResult<bool> {
    match v {
        Value::Object(o) => Ok(o.contains_key(key)),
        Value::Array(a) => Ok(a.iter().any(|x| x.as_str() == Some(key))),
        Value::String(s) => Ok(s.contains(key)),
        _ => err(format!(
            "argument of type '{}' is not iterable",
            type_name(v)
        )),
    }
}

/// `a + b`.
pub fn add(a: &Value, b: &Value) -> PyResult<Value> {
    match (num(a), num(b)) {
        (Some(Num::Int(x)), Some(Num::Int(y))) => Ok(int_value(x + y)),
        (Some(x), Some(y)) => {
            let f = |n: Num| match n {
                Num::Int(i) => i as f64,
                Num::Float(f) => f,
            };
            Ok(float_value(f(x) + f(y)))
        }
        _ => match (a, b) {
            (Value::String(x), Value::String(y)) => Ok(Value::String(format!("{x}{y}"))),
            (Value::Array(x), Value::Array(y)) => {
                Ok(Value::Array(x.iter().chain(y).cloned().collect()))
            }
            _ => err(format!(
                "unsupported operand type(s) for +: '{}' and '{}'",
                type_name(a),
                type_name(b)
            )),
        },
    }
}

/// `"".join(items)`: every item must be a string.
pub fn join(items: &[Value]) -> PyResult<String> {
    let mut out = String::new();
    for (i, v) in items.iter().enumerate() {
        match v {
            Value::String(s) => out.push_str(s),
            _ => {
                return err(format!(
                    "sequence item {i}: expected str instance, {} found",
                    type_name(v)
                ))
            }
        }
    }
    Ok(out)
}

/// `int(v)`.
pub fn int(v: &Value) -> PyResult<i128> {
    match v {
        Value::Bool(b) => Ok(i128::from(*b)),
        Value::Number(n) => match int_of(n) {
            Some(i) => Ok(i),
            None => {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if !f.is_finite() {
                    return err("cannot convert float to integer");
                }
                if f.abs() >= 1.7e38 {
                    return err("integer too large");
                }
                Ok(f.trunc() as i128)
            }
        },
        Value::String(s) => parse_int(s).ok_or_else(|| {
            PyErr(format!(
                "invalid literal for int() with base 10: {}",
                repr_str(s)
            ))
        }),
        _ => err(format!(
            "int() argument must be a string, a bytes-like object or a real number, not '{}'",
            type_name(v)
        )),
    }
}

/// `int(s)` for a string: optional sign, ASCII digits, single underscores
/// between digits, surrounding whitespace.
fn parse_int(s: &str) -> Option<i128> {
    let t = strip(s);
    let (neg, digits) = match t.as_bytes().first()? {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
    {
        return None;
    }
    let mut value: i128 = 0;
    for c in digits.chars() {
        if c == '_' {
            continue;
        }
        let d = c.to_digit(10)?;
        value = value.checked_mul(10)?.checked_add(i128::from(d))?;
    }
    Some(if neg { -value } else { value })
}

/// Characters `str.strip()` removes (`str.isspace()`).
pub fn is_py_space(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n' | '\x0b' | '\x0c' | '\r' | '\x1c'..='\x1f' | ' ' | '\u{85}' | '\u{a0}'
            | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}'
            | '\u{205f}' | '\u{3000}'
    )
}

/// `s.strip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

/// `s.lstrip()`.
pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(is_py_space)
}

/// `str(v)`.
pub fn str_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => repr(v),
    }
}

/// `repr(v)`.
pub fn repr(v: &Value) -> String {
    let mut out = String::new();
    write_repr(v, &mut out);
    out
}

fn write_repr(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("None"),
        Value::Bool(true) => out.push_str("True"),
        Value::Bool(false) => out.push_str("False"),
        Value::Number(n) => out.push_str(&number_repr(n)),
        Value::String(s) => out.push_str(&repr_str(s)),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_repr(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&repr_str(k));
                out.push_str(": ");
                write_repr(x, out);
            }
            out.push('}');
        }
    }
}

fn number_repr(n: &Number) -> String {
    match int_of(n) {
        Some(i) => i.to_string(),
        None => float_repr(n.as_f64().unwrap_or(f64::NAN)),
    }
}

/// `repr(s)` for a string: single quotes unless the text has a single
/// quote and no double quote; escapes as CPython does for printable
/// text and control characters.
pub fn repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..0xa0).contains(&(c as u32)) => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c if !is_printable(c) => {
                let u = c as u32;
                if u <= 0xff {
                    out.push_str(&format!("\\x{u:02x}"));
                } else if u <= 0xffff {
                    out.push_str(&format!("\\u{u:04x}"));
                } else {
                    out.push_str(&format!("\\U{u:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// An approximation of `str.isprintable()` for characters past Latin-1:
/// separators, format characters and private-use or unassigned planes
/// are not printable.
fn is_printable(c: char) -> bool {
    let u = c as u32;
    !(u == 0xad
        || (0x2000..=0x200f).contains(&u)
        || (0x2028..=0x202f).contains(&u)
        || (0x205f..=0x206f).contains(&u)
        || u == 0x3000
        || u == 0xfeff
        || (0xfff9..=0xfffb).contains(&u)
        || (0xe000..=0xf8ff).contains(&u)
        || u >= 0xf0000
        || u == 0x1680
        || u == 0xa0)
}

/// `float.__repr__`: the shortest digits that round-trip, positional for
/// exponents from -4 to 15, scientific (`1e+16`, `1e-05`) otherwise.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Rust's `{:e}` gives the shortest round-trip digits.
    let sci = format!("{:e}", x.abs());
    let (mant, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    let n = digits.len() as i32;
    if (-4..16).contains(&exp) {
        if exp >= n - 1 {
            // Integral: digits then zeros, then ".0".
            let zeros = "0".repeat((exp - (n - 1)) as usize);
            format!("{sign}{digits}{zeros}.0")
        } else if exp >= 0 {
            let (a, b) = digits.split_at((exp + 1) as usize);
            format!("{sign}{a}.{b}")
        } else {
            let zeros = "0".repeat((-exp - 1) as usize);
            format!("{sign}0.{zeros}{digits}")
        }
    } else {
        let mant = if n > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mant}e{esign}{:02}", exp.abs())
    }
}

/// `json.dumps(v, separators=(",", ":"), ensure_ascii=False)`, which is
/// also what Starlette's `JSONResponse` writes.
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write_json(v, &mut out);
    out
}

pub fn write_json(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => match int_of(n) {
            Some(i) => out.push_str(&i.to_string()),
            None => {
                let f = n.as_f64().unwrap_or(f64::NAN);
                if f.is_nan() {
                    out.push_str("NaN");
                } else if f.is_infinite() {
                    out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
                } else {
                    out.push_str(&float_repr(f));
                }
            }
        },
        Value::String(s) => write_json_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_str(k, out);
                out.push(':');
                write_json(x, out);
            }
            out.push('}');
        }
    }
}

fn write_json_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A JSON float (Python `float`).
pub fn float_value(x: f64) -> Value {
    Number::from_f64(x).map_or(Value::Null, Value::Number)
}

/// A JSON int (Python `int`).
pub fn int_value(i: i128) -> Value {
    if let Ok(v) = i64::try_from(i) {
        Value::from(v)
    } else if let Ok(v) = u64::try_from(i) {
        Value::from(v)
    } else {
        float_value(i as f64)
    }
}

/// `json.loads(s)` for a stream event; None where Python raises
/// `ValueError`. Python also accepts `NaN` and `Infinity` and lone
/// surrogates, which the gateway treats as unparsable.
pub fn loads(s: &str) -> Option<Value> {
    serde_json::from_str(s).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        for (x, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (123.456, "123.456"),
            (-2.5e300, "-2.5e+300"),
            (5e-324, "5e-324"),
            (-0.0, "-0.0"),
        ] {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn repr_and_truthiness() {
        assert_eq!(
            repr(&json!({"a": [1, 2.0, null, true], "b'": "x'y"})),
            r#"{'a': [1, 2.0, None, True], "b'": "x'y"}"#
        );
        assert_eq!(repr(&json!("it's \"q\"")), r#"'it\'s "q"'"#);
        assert!(!truthy(&json!(0.0)) && truthy(&json!([0])) && !truthy(&json!({})));
        assert!(
            eq(&json!(1), &json!(true))
                && eq(&json!(1.0), &json!(1))
                && !eq(&json!("1"), &json!(1))
        );
        assert_eq!(int(&json!(" -1_000 ")).unwrap(), -1000);
        assert!(int(&json!("1.5")).is_err() && int(&json!([])).is_err());
        assert_eq!(int(&json!(2.9)).unwrap(), 2);
    }

    #[test]
    fn dumps_matches_python() {
        assert_eq!(
            dumps(&json!({"a": "é\n\u{1}", "b": [1.0, 2, 1e-7]})),
            "{\"a\":\"é\\n\\u0001\",\"b\":[1.0,2,1e-07]}"
        );
    }
}

//! Server-sent events (`swarmpilot/api/sse.py`).

use serde_json::Value;

use super::py;

/// `iter_sse` as a push parser: feed raw chunks, take `(event, data)`.
#[derive(Default)]
pub struct SseParser {
    buf: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseParser {
    /// Parse one chunk; complete events are appended to `out`.
    pub fn feed(&mut self, chunk: &[u8], out: &mut Vec<(Option<String>, String)>) {
        self.buf.extend_from_slice(chunk);
        let mut start = 0;
        while let Some(nl) = self.buf[start..].iter().position(|b| *b == b'\n') {
            let mut line = &self.buf[start..start + nl];
            start += nl + 1;
            while let [rest @ .., b'\r'] = line {
                line = rest;
            }
            let line = String::from_utf8_lossy(line).into_owned();
            self.line(line, out);
        }
        self.buf.drain(..start);
    }

    fn line(&mut self, line: String, out: &mut Vec<(Option<String>, String)>) {
        if line.is_empty() {
            if !self.data.is_empty() {
                out.push((self.event.take(), self.data.join("\n")));
            }
            self.event = None;
            self.data.clear();
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (key, value) = line.split_once(':').unwrap_or((&line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        if key == "data" {
            self.data.push(value.to_string());
        } else if key == "event" {
            self.event = Some(value.to_string());
        }
    }

    /// The stream ended: a last unterminated `data:` line and any pending
    /// event.
    pub fn finish(&mut self, out: &mut Vec<(Option<String>, String)>) {
        let tail = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        let tail = py::strip(&tail);
        if let Some(rest) = tail.strip_prefix("data:") {
            self.data.push(py::lstrip(rest).to_string());
        }
        if !self.data.is_empty() {
            out.push((self.event.take(), self.data.join("\n")));
        }
        self.data.clear();
    }
}

/// `frame(data, event)`: one SSE event, the data as compact JSON.
pub fn frame(data: &Value, event: Option<&str>) -> Vec<u8> {
    let payload = match data {
        Value::String(s) => s.clone(),
        _ => py::dumps(data),
    };
    let head = event.map_or(String::new(), |e| format!("event: {e}\n"));
    format!("{head}data: {payload}\n\n").into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_anywhere() {
        let text = b"event: a\r\ndata: {\"x\":1}\n\n: c\ndata: one\ndata:two\n\ndata: tail";
        for cut in 0..text.len() {
            let mut p = SseParser::default();
            let mut out = Vec::new();
            p.feed(&text[..cut], &mut out);
            p.feed(&text[cut..], &mut out);
            p.finish(&mut out);
            assert_eq!(
                out,
                vec![
                    (Some("a".into()), "{\"x\":1}".into()),
                    (None, "one\ntwo".into()),
                    (None, "tail".into()),
                ]
            );
        }
    }
}

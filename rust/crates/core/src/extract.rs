//! Declarative feature extractors (`swarmpilot/engines/extractors.py`).
//!
//! An expression such as `sum(chars(messages[*].content), chars(prompt))`
//! is compiled once into an [`Extractor`] and evaluated against request
//! bodies. Bodies are anything implementing [`Doc`]: `serde_json::Value`
//! here, Python objects in the PyO3 module. Results must equal the Python
//! extractor's for every body; see `tests/golden/features.json` and
//! `tests/core/test_parity.py`.

use std::borrow::Cow;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

use crate::fsum::fsum;

/// An expression could not be parsed.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ExtractorError(pub String);

/// A JSON-like request body, with Python's type tests.
pub trait Doc: Sized + Clone {
    /// The value under `key`, when this is a mapping holding it.
    fn get(&self, key: &str) -> Option<Self>;
    /// The elements, when this is exactly a list.
    fn elements(&self) -> Option<Vec<Self>>;
    /// The element at `index`, when this is exactly a list that long.
    fn item(&self, index: usize) -> Option<Self>;
    /// The value as a finite float: ints and floats, never bools.
    fn number(&self) -> Option<f64>;
    /// `len()` of a list, string (code points) or mapping.
    fn sized_len(&self) -> Option<usize>;
    /// Code points, when this is exactly a string.
    fn exact_str_len(&self) -> Option<usize>;
    /// The text, when this is a string.
    fn text(&self) -> Option<Cow<'_, str>>;
}

impl Doc for &Value {
    fn get(&self, key: &str) -> Option<Self> {
        self.as_object()?.get(key)
    }
    fn elements(&self) -> Option<Vec<Self>> {
        Some(self.as_array()?.iter().collect())
    }
    fn item(&self, index: usize) -> Option<Self> {
        self.as_array()?.get(index)
    }
    fn number(&self) -> Option<f64> {
        self.as_f64().filter(|v| v.is_finite())
    }
    fn sized_len(&self) -> Option<usize> {
        match self {
            Value::Array(a) => Some(a.len()),
            Value::String(s) => Some(s.chars().count()),
            Value::Object(o) => Some(o.len()),
            _ => None,
        }
    }
    fn exact_str_len(&self) -> Option<usize> {
        self.as_str().map(|s| s.chars().count())
    }
    fn text(&self) -> Option<Cow<'_, str>> {
        self.as_str().map(Cow::Borrowed)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Step {
    Key(String),
    Index(usize),
    All,
}

#[derive(Clone, Debug, PartialEq)]
struct Path(Vec<Step>);

impl Path {
    fn parse(path: &str) -> Result<Self, ExtractorError> {
        static PART: OnceLock<Regex> = OnceLock::new();
        static INDEX: OnceLock<Regex> = OnceLock::new();
        let part_re =
            PART.get_or_init(|| Regex::new(r"^([^\[\]]*)((?:\[(?:\*|[0-9]+)\])*)$").unwrap());
        let index_re = INDEX.get_or_init(|| Regex::new(r"\[(\*|[0-9]+)\]").unwrap());
        let bad = || ExtractorError(format!("bad path {path:?}"));
        let mut steps = Vec::new();
        for part in path.split('.') {
            let m = part_re.captures(part).ok_or_else(bad)?;
            let (key, idx) = (&m[1], &m[2]);
            if key.is_empty() && idx.is_empty() {
                return Err(bad());
            }
            if !key.is_empty() {
                steps.push(Step::Key(key.to_string()));
            }
            for c in index_re.captures_iter(idx) {
                steps.push(match &c[1] {
                    "*" => Step::All,
                    // An index too large to hold matches nothing.
                    n => Step::Index(n.parse().unwrap_or(usize::MAX)),
                });
            }
        }
        Ok(Path(steps))
    }

    /// Every value the path matches, in order.
    fn walk<D: Doc>(&self, body: &D) -> Vec<D> {
        let mut cur = vec![body.clone()];
        for step in &self.0 {
            let mut next = Vec::new();
            for v in &cur {
                match step {
                    Step::All => {
                        if let Some(items) = v.elements() {
                            next.extend(items);
                        }
                    }
                    Step::Index(i) => next.extend(v.item(*i)),
                    Step::Key(k) => next.extend(v.get(k)),
                }
            }
            if next.is_empty() {
                return next;
            }
            cur = next;
        }
        cur
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Node {
    Const(f64),
    Field(Path),
    Len(Path),
    Count(Path),
    Chars(Path),
    Area(Path),
    Coalesce(Vec<Node>),
    Product(Vec<Node>),
    Sum(Vec<Node>),
    Min(Vec<Node>),
    Max(Vec<Node>),
}

/// `area("WxH")`: both sides 1 to 18 ASCII digits.
fn area_of(text: &str) -> Option<f64> {
    let lower = text.to_lowercase();
    let (w, h) = lower.split_once('x')?;
    let ok = |s: &str| !s.is_empty() && s.len() <= 18 && s.bytes().all(|b| b.is_ascii_digit());
    if !ok(w) || !ok(h) {
        return None;
    }
    let (w, h): (u128, u128) = (w.parse().ok()?, h.parse().ok()?);
    Some((w * h) as f64)
}

impl Node {
    fn eval<D: Doc>(&self, body: &D) -> Option<f64> {
        match self {
            Node::Const(v) => Some(*v),
            Node::Field(p) => p.walk(body).first().and_then(|v| v.number()),
            Node::Len(p) => p
                .walk(body)
                .iter()
                .find_map(|v| v.sized_len())
                .map(|n| n as f64),
            Node::Count(p) => Some(p.walk(body).len() as f64),
            Node::Chars(p) => Some(
                p.walk(body)
                    .iter()
                    .filter_map(|v| v.exact_str_len())
                    .sum::<usize>() as f64,
            ),
            Node::Area(p) => p.walk(body).first().and_then(|v| area_of(&v.text()?)),
            Node::Coalesce(args) => args.iter().find_map(|a| a.eval(body)),
            Node::Product(args) => all(args, body).map(|v| v.iter().product()),
            Node::Sum(args) => all(args, body).and_then(|v| fsum(&v)),
            Node::Min(args) => all(args, body).map(|v| {
                v.iter()
                    .copied()
                    .reduce(|a, b| if b < a { b } else { a })
                    .unwrap()
            }),
            Node::Max(args) => all(args, body).map(|v| {
                v.iter()
                    .copied()
                    .reduce(|a, b| if b > a { b } else { a })
                    .unwrap()
            }),
        }
    }
}

fn all<D: Doc>(args: &[Node], body: &D) -> Option<Vec<f64>> {
    args.iter().map(|a| a.eval(body)).collect()
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Name(String),
    Num(String),
    Op(char),
}

/// Characters Python's `str.strip()` removes.
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

fn tokenize(text: &str) -> Vec<Tok> {
    static TOKEN: OnceLock<Regex> = OnceLock::new();
    let re = TOKEN.get_or_init(|| {
        Regex::new(
            r"[ \t\n\r\x0B\x0C]*(?:([A-Za-z_][A-Za-z0-9_.\[\]*-]*)|(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)|(.))",
        )
        .unwrap()
    });
    re.captures_iter(text)
        .filter_map(|c| {
            if let Some(n) = c.get(1) {
                Some(Tok::Name(n.as_str().into()))
            } else if let Some(n) = c.get(2) {
                Some(Tok::Num(n.as_str().into()))
            } else {
                let op = c.get(3)?.as_str().chars().next()?;
                (!py_space(op)).then_some(Tok::Op(op))
            }
        })
        .collect()
}

struct Parser<'a> {
    text: &'a str,
    tokens: Vec<Tok>,
    pos: usize,
    depth: usize,
}

/// Deepest nesting of combiner calls (`MAX_DEPTH` in Python), so an
/// expression cannot exhaust the stack.
pub const MAX_DEPTH: usize = 64;

fn num(s: &str) -> f64 {
    s.parse()
        .expect("the token regex only matches valid floats")
}

impl Parser<'_> {
    fn err(&self, what: &str) -> ExtractorError {
        ExtractorError(format!("{what} in {:?}", self.text))
    }

    fn peek(&self) -> Option<&Tok> {
        self.tokens.get(self.pos)
    }

    fn take(&mut self) -> Result<Tok, ExtractorError> {
        let tok = self
            .peek()
            .cloned()
            .ok_or_else(|| self.err("unexpected end"))?;
        self.pos += 1;
        Ok(tok)
    }

    fn expect(&mut self, op: char) -> Result<(), ExtractorError> {
        match self.take()? {
            Tok::Op(c) if c == op => Ok(()),
            _ => Err(self.err(&format!("expected {op:?}"))),
        }
    }

    fn keyword_default(&mut self) -> Result<Node, ExtractorError> {
        self.take()?;
        if self.take()? != Tok::Name("default".into()) {
            return Err(self.err("expected default="));
        }
        self.expect('=')?;
        match self.take()? {
            Tok::Num(n) => Ok(Node::Const(num(&n))),
            _ => Err(self.err("default must be a number")),
        }
    }

    fn expr(&mut self) -> Result<Node, ExtractorError> {
        let name = match self.take()? {
            Tok::Num(n) => return Ok(Node::Const(num(&n))),
            Tok::Op(c) => return Err(self.err(&format!("unexpected {c:?}"))),
            Tok::Name(n) => n,
        };
        if self.peek() != Some(&Tok::Op('(')) {
            return Ok(Node::Field(Path::parse(&name)?));
        }
        self.take()?;
        let path_fn: Option<fn(Path) -> Node> = match name.as_str() {
            "field" => Some(Node::Field),
            "len" => Some(Node::Len),
            "count" => Some(Node::Count),
            "chars" => Some(Node::Chars),
            "area" => Some(Node::Area),
            _ => None,
        };
        if let Some(make) = path_fn {
            let Tok::Name(path) = self.take()? else {
                return Err(self.err(&format!("{name}() needs a path")));
            };
            let mut node = make(Path::parse(&path)?);
            if self.peek() == Some(&Tok::Op(',')) {
                node = Node::Coalesce(vec![node, self.keyword_default()?]);
            }
            self.expect(')')?;
            return Ok(node);
        }
        let combine: fn(Vec<Node>) -> Node = match name.as_str() {
            "product" => Node::Product,
            "sum" => Node::Sum,
            "min" => Node::Min,
            "max" => Node::Max,
            "coalesce" => Node::Coalesce,
            _ => return Err(self.err(&format!("unknown function {name:?}"))),
        };
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.err("nested too deeply"));
        }
        let mut args = vec![self.expr()?];
        loop {
            match self.take()? {
                Tok::Op(',') => args.push(self.expr()?),
                Tok::Op(')') => break,
                _ => return Err(self.err("expected ')'")),
            }
        }
        self.depth -= 1;
        Ok(combine(args))
    }
}

/// A compiled extractor expression.
#[derive(Clone, Debug, PartialEq)]
pub struct Extractor {
    root: Node,
    default: f64,
}

impl Extractor {
    /// Compile `expr`; bodies lacking what it reads give `default`.
    pub fn compile(expr: &str, default: f64) -> Result<Self, ExtractorError> {
        let mut p = Parser {
            text: expr,
            tokens: tokenize(expr),
            pos: 0,
            depth: 0,
        };
        let root = p.expr()?;
        if p.peek().is_some() {
            return Err(p.err("trailing input"));
        }
        Ok(Extractor { root, default })
    }

    /// The feature value for `body`, always finite.
    pub fn eval<D: Doc>(&self, body: &D) -> f64 {
        match self.root.eval(body) {
            Some(v) if v.is_finite() => v,
            _ => self.default,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn ev(expr: &str, body: Value) -> f64 {
        Extractor::compile(expr, 0.0).unwrap().eval(&&body)
    }

    #[test]
    fn grammar() {
        let body = json!({
            "messages": [{"content": "abc"}, {"content": [{"text": "de"}]}],
            "max_tokens": 7, "flag": true, "size": "640X480",
        });
        assert_eq!(ev("chars(messages[*].content)", body.clone()), 3.0);
        assert_eq!(ev("chars(messages[*].content[*].text)", body.clone()), 2.0);
        assert_eq!(ev("len(messages)", body.clone()), 2.0);
        assert_eq!(ev("count(messages[*].content)", body.clone()), 2.0);
        assert_eq!(ev("max_tokens", body.clone()), 7.0);
        assert_eq!(ev("field(flag, default=3)", body.clone()), 3.0);
        assert_eq!(ev("coalesce(missing, 1e3)", body.clone()), 1000.0);
        assert_eq!(ev("area(size)", body.clone()), 307200.0);
        assert_eq!(ev("min(4, max_tokens, 9)", body.clone()), 4.0);
        assert_eq!(ev("product(max_tokens, missing)", body.clone()), 0.0);
        assert_eq!(ev("sum(0.1, 0.2, 0.3)", body), 0.6);
    }

    #[test]
    fn limits_nesting() {
        let nested = |n: usize| format!("{}1{}", "sum(".repeat(n), ")".repeat(n));
        assert!(Extractor::compile(&nested(MAX_DEPTH), 0.0).is_ok());
        assert!(Extractor::compile(&nested(MAX_DEPTH + 1), 0.0).is_err());
        assert!(Extractor::compile(&nested(200_000), 0.0).is_err());
    }

    #[test]
    fn rejects_bad_expressions() {
        for bad in [
            "",
            "foo(",
            "nope(x)",
            "field(1)",
            "a..b",
            "sum(1,",
            "1 2",
            "field(a, b=1)",
        ] {
            assert!(Extractor::compile(bad, 0.0).is_err(), "{bad}");
        }
    }
}

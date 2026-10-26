"""Declarative feature extractors.

Engines configured in YAML cannot ship Python code for
:meth:`~swarmpilot.engines.base.EngineAdapter.features`, so they describe
each feature with a small expression over the request body instead:

========================  ==================================================
Expression                Value
========================  ==================================================
``field(max_tokens)``     The number at that path.
``len(messages)``         Length of the list or string at that path.
``count(items[*].id)``    How many values the path matches.
``chars(messages[*].content)``  Total characters over every match.
``product(width, height)``  Product of its arguments.
``sum(a, b)``             Sum of its arguments.
``min(a, b)``, ``max(a, b)``  Smallest or largest argument.
``42``                    A constant.
========================  ==================================================

Paths use dots for keys, ``[N]`` for a list index and ``[*]`` for every
element. A bare path as an argument means ``field(path)``. When a path is
missing or not a number, the feature takes its ``default``.

Expressions are parsed once into closures, never evaluated with ``eval``,
so a configuration file cannot run code, and the per-request cost is a few
dictionary lookups.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator, Mapping
from typing import Any

_MISSING = object()
_TOKEN = re.compile(r"\s*(?:([A-Za-z_][\w.\[\]*-]*)|(-?\d+(?:\.\d+)?)|(.))")

Getter = Callable[[Mapping[str, Any]], Any]
Extractor = Callable[[Mapping[str, Any]], float]


class ExtractorError(ValueError):
    """An extractor expression could not be parsed."""


def _parse_path(path: str) -> tuple[str | int | None, ...]:
    """Split ``a.b[*].c[0]`` into ``("a", "b", None, "c", 0)``."""
    steps: list[str | int | None] = []
    for part in path.split("."):
        m = re.fullmatch(r"([^\[\]]*)((?:\[(?:\*|\d+)\])*)", part)
        if not m or (not m.group(1) and not m.group(2)):
            raise ExtractorError(f"bad path {path!r}")
        if m.group(1):
            steps.append(m.group(1))
        for idx in re.findall(r"\[(\*|\d+)\]", m.group(2)):
            steps.append(None if idx == "*" else int(idx))
    return tuple(steps)


def _walk(value: Any, steps: tuple[str | int | None, ...]) -> Iterator[Any]:
    if not steps:
        yield value
        return
    step, rest = steps[0], steps[1:]
    if step is None:
        if isinstance(value, list):
            for item in value:
                yield from _walk(item, rest)
    elif isinstance(step, int):
        if isinstance(value, list) and -len(value) <= step < len(value):
            yield from _walk(value[step], rest)
    elif isinstance(value, Mapping) and step in value:
        yield from _walk(value[step], rest)


def _number(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _MISSING
    return float(value) if math.isfinite(value) else _MISSING


def _path_getter(path: str) -> Callable[[Any], Iterator[Any]]:
    steps = _parse_path(path)
    return lambda body: _walk(body, steps)


def _first(path: str) -> Getter:
    walk = _path_getter(path)

    def get(body: Mapping[str, Any]) -> Any:
        for v in walk(body):
            return _number(v)
        return _MISSING

    return get


def _len(path: str) -> Getter:
    walk = _path_getter(path)

    def get(body: Mapping[str, Any]) -> Any:
        for v in walk(body):
            if isinstance(v, (list, str, Mapping)):
                return float(len(v))
        return _MISSING

    return get


def _count(path: str) -> Getter:
    walk = _path_getter(path)
    return lambda body: float(sum(1 for _ in walk(body)))


def _chars(path: str) -> Getter:
    walk = _path_getter(path)
    return lambda body: float(
        sum(len(v) for v in walk(body) if isinstance(v, str))
    )


def _combine(fn: Callable[[list[float]], float]) -> Callable[..., Getter]:
    def build(*args: Getter) -> Getter:
        if not args:
            raise ExtractorError("needs at least one argument")

        def get(body: Mapping[str, Any]) -> Any:
            vals = [a(body) for a in args]
            if any(v is _MISSING for v in vals):
                return _MISSING
            return fn(vals)

        return get

    return build


_PATH_FUNCS: dict[str, Callable[[str], Getter]] = {
    "field": _first,
    "len": _len,
    "count": _count,
    "chars": _chars,
}
_COMBINERS: dict[str, Callable[..., Getter]] = {
    "product": _combine(math.prod),
    "sum": _combine(math.fsum),
    "min": _combine(min),
    "max": _combine(max),
}


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.tokens = [
            (name, num, op)
            for name, num, op in _TOKEN.findall(text)
            if name or num or op.strip()
        ]
        self.pos = 0

    def peek(self) -> tuple[str, str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> tuple[str, str, str]:
        tok = self.peek()
        if tok is None:
            raise ExtractorError(f"unexpected end of {self.text!r}")
        self.pos += 1
        return tok

    def expect(self, op: str) -> None:
        if self.take()[2] != op:
            raise ExtractorError(f"expected {op!r} in {self.text!r}")

    def expr(self) -> Getter:
        name, num, op = self.take()
        if num:
            value = float(num)
            return lambda body: value
        if not name:
            raise ExtractorError(f"unexpected {op!r} in {self.text!r}")
        nxt = self.peek()
        if nxt is None or nxt[2] != "(":
            return _first(name)
        self.take()
        if name in _PATH_FUNCS:
            path = self.take()[0]
            if not path:
                raise ExtractorError(f"{name}() needs a path in {self.text!r}")
            self.expect(")")
            return _PATH_FUNCS[name](path)
        if name in _COMBINERS:
            args = [self.expr()]
            while (tok := self.take())[2] == ",":
                args.append(self.expr())
            if tok[2] != ")":
                raise ExtractorError(f"expected ')' in {self.text!r}")
            return _COMBINERS[name](*args)
        raise ExtractorError(f"unknown function {name!r} in {self.text!r}")


def compile_extractor(expr: str, default: float = 0.0) -> Extractor:
    """Compile an extractor expression into a function of the body.

    Args:
        expr: Expression such as ``chars(messages[*].content)``.
        default: Value when the body lacks what the expression reads.

    Returns:
        A function that maps a request body to a finite float.

    Raises:
        ExtractorError: If the expression is malformed.
    """
    parser = _Parser(expr)
    getter = parser.expr()
    if parser.peek() is not None:
        raise ExtractorError(f"trailing input in {expr!r}")
    default = float(default)

    def extract(body: Mapping[str, Any]) -> float:
        value = getter(body)
        if value is _MISSING or not math.isfinite(value):
            return default
        return value

    return extract

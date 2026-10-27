"""Declarative feature extractors.

Engines configured in YAML cannot ship Python code for
:meth:`~swarmpilot.engines.base.EngineAdapter.features`, so they describe
each feature with a small expression over the request body instead:

========================  ==================================================
Expression                Value
========================  ==================================================
``field(max_tokens)``     The number at that path.
``field(max_tokens, default=256)``  The same, with a default of its own.
``len(messages)``         Length of the list or string at that path.
``count(items[*].id)``    How many values the path matches.
``chars(messages[*].content)``  Total characters over every match.
``product(width, height)``  Product of its arguments.
``sum(a, b)``             Sum of its arguments.
``min(a, b)``, ``max(a, b)``  Smallest or largest argument.
``coalesce(a, b, 1024)``  The first argument that has a value.
``area(size)``            ``W*H`` from a ``"WxH"`` string such as ``"1024x768"``.
``42``                    A constant.
========================  ==================================================

Paths use dots for keys, ``[N]`` for a list index and ``[*]`` for every
element. A bare path as an argument means ``field(path)``. When a path is
missing or not a number, the feature takes its ``default``.

The same grammar is part of the adapter manifest (see
:mod:`swarmpilot.engines.manifest`), so any implementation of it, such as
a future Rust gateway, extracts identical features. ``tokens(...)`` is
reserved for token counts and not implemented yet.

Expressions are parsed once into closures, never evaluated with ``eval``,
so a configuration file cannot run code, and the per-request cost is a few
dictionary lookups.

Names, paths and numbers are ASCII. ``area()`` reads 1 to 18 ASCII digits
on each side, and values too large for a float count as missing. With
``SCHEDULER_CORE=rust`` the Rust implementation in ``rust/crates/core``
evaluates the expressions; ``tests/core/`` checks both agree.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Any

_MISSING = object()
_TOKEN = re.compile(
    r"\s*(?:([A-Za-z_][\w.\[\]*-]*)|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|(.))",
    re.ASCII,
)
_AREA_SIDE = re.compile(r"[0-9]{1,18}")

#: Deepest nesting of combiner calls, so a configuration cannot exhaust
#: the stack (the Rust parser has the same limit).
MAX_DEPTH = 64

Getter = Callable[[Mapping[str, Any]], Any]
Extractor = Callable[[Mapping[str, Any]], float]


class ExtractorError(ValueError):
    """An extractor expression could not be parsed."""


def _parse_path(path: str) -> tuple[str | int | None, ...]:
    """Split ``a.b[*].c[0]`` into ``("a", "b", None, "c", 0)``."""
    steps: list[str | int | None] = []
    for part in path.split("."):
        m = re.fullmatch(r"([^\[\]]*)((?:\[(?:\*|[0-9]+)\])*)", part)
        if not m or (not m.group(1) and not m.group(2)):
            raise ExtractorError(f"bad path {path!r}")
        if m.group(1):
            steps.append(m.group(1))
        for idx in re.findall(r"\[(\*|[0-9]+)\]", m.group(2)):
            steps.append(None if idx == "*" else int(idx))
    return tuple(steps)


def _number(value: Any) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _MISSING
    try:
        value = float(value)
    except OverflowError:
        return _MISSING
    return value if math.isfinite(value) else _MISSING


def _is_map(v: Any) -> bool:
    return type(v) is dict or isinstance(v, Mapping)


def _path_getter(path: str) -> Callable[[Any], list[Any]]:
    """Compile a path into a function returning every value it matches.

    Paths without ``[*]`` take a fast single-value route; this runs on
    every request, so it avoids generators and recursion.
    """
    steps = _parse_path(path)

    if None not in steps:

        def single(body: Any) -> list[Any]:
            v = body
            for step in steps:
                if type(step) is int:
                    if type(v) is not list or not -len(v) <= step < len(v):
                        return []
                    v = v[step]
                elif _is_map(v):
                    v = v.get(step, _MISSING)
                    if v is _MISSING:
                        return []
                else:
                    return []
            return [v]

        return single

    def many(body: Any) -> list[Any]:
        cur = [body]
        for step in steps:
            nxt: list[Any] = []
            if step is None:
                for v in cur:
                    if type(v) is list:
                        nxt.extend(v)
            elif type(step) is int:
                for v in cur:
                    if type(v) is list and -len(v) <= step < len(v):
                        nxt.append(v[step])
            else:
                for v in cur:
                    if _is_map(v):
                        x = v.get(step, _MISSING)
                        if x is not _MISSING:
                            nxt.append(x)
            if not nxt:
                return nxt
            cur = nxt
        return cur

    return many


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
    return lambda body: float(len(walk(body)))


def _chars(path: str) -> Getter:
    walk = _path_getter(path)

    def get(body: Mapping[str, Any]) -> float:
        n = 0
        for v in walk(body):
            if type(v) is str:
                n += len(v)
        return float(n)

    return get


def _area(path: str) -> Getter:
    walk = _path_getter(path)

    def get(body: Mapping[str, Any]) -> Any:
        for v in walk(body):
            if isinstance(v, str):
                w, _, h = v.lower().partition("x")
                if _AREA_SIDE.fullmatch(w) and _AREA_SIDE.fullmatch(h):
                    return float(int(w) * int(h))
            return _MISSING
        return _MISSING

    return get


def _coalesce(*args: Getter) -> Getter:
    if not args:
        raise ExtractorError("needs at least one argument")

    def get(body: Mapping[str, Any]) -> Any:
        for a in args:
            v = a(body)
            if v is not _MISSING:
                return v
        return _MISSING

    return get


def _combine(fn: Callable[[list[float]], float]) -> Callable[..., Getter]:
    def build(*args: Getter) -> Getter:
        if not args:
            raise ExtractorError("needs at least one argument")

        def get(body: Mapping[str, Any]) -> Any:
            vals = []
            for a in args:
                v = a(body)
                if v is _MISSING:
                    return _MISSING
                vals.append(v)
            try:
                return fn(vals)
            except (OverflowError, ValueError):  # fsum overflow, inf - inf
                return _MISSING

        return get

    return build


_PATH_FUNCS: dict[str, Callable[[str], Getter]] = {
    "field": _first,
    "len": _len,
    "count": _count,
    "chars": _chars,
    "area": _area,
}
_COMBINERS: dict[str, Callable[..., Getter]] = {
    "product": _combine(math.prod),
    "sum": _combine(math.fsum),
    "min": _combine(min),
    "max": _combine(max),
    "coalesce": _coalesce,
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
        self.depth = 0

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

    def keyword_default(self) -> Getter:
        self.take()
        if self.take()[0] != "default":
            raise ExtractorError(f"expected default= in {self.text!r}")
        self.expect("=")
        num = self.take()[1]
        if not num:
            raise ExtractorError(f"default must be a number in {self.text!r}")
        value = float(num)
        return lambda body: value

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
            getter = _PATH_FUNCS[name](path)
            if self.peek() == ("", "", ","):
                getter = _coalesce(getter, self.keyword_default())
            self.expect(")")
            return getter
        if name in _COMBINERS:
            self.depth += 1
            if self.depth > MAX_DEPTH:
                raise ExtractorError(f"nested too deeply in {self.text!r}")
            args = [self.expr()]
            while (tok := self.take())[2] == ",":
                args.append(self.expr())
            if tok[2] != ")":
                raise ExtractorError(f"expected ')' in {self.text!r}")
            self.depth -= 1
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
    from swarmpilot.scheduler import core

    rc = core.rust()
    if rc is not None:
        try:
            return rc.Extractor(expr, float(default))
        except ValueError as e:
            raise ExtractorError(str(e)) from None
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

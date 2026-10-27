"""Differential tests: the Rust core against the Python code it replaces.

Each test feeds the same random inputs to the Python function and to its
Rust counterpart and asserts identical results, bit for bit. Seeds are
fixed, so a failure repeats.
"""

from __future__ import annotations

import asyncio
import json
import math
import random

import numpy as np
import pytest

from swarmpilot.engines.extractors import ExtractorError, compile_extractor
from swarmpilot.scheduler.algorithms.factory import get_strategy
from swarmpilot.scheduler.clients.models import Prediction
from swarmpilot.scheduler.models import (
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)
from swarmpilot.scheduler.registry.queue_sketch import (
    DEFAULT_QUANTILES,
    Charge,
    apply,
)

GRIDS = (
    list(DEFAULT_QUANTILES),
    [0.5],
    [0.1, 0.5, 0.9],
    [0.01, 0.25, 0.5, 0.75, 0.99],
)


def _quantiles(rng: random.Random) -> dict[float, float] | None:
    if rng.random() < 0.2:
        return None
    qs = sorted(
        rng.sample(
            [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99], rng.randint(1, 6)
        )
    )
    v = rng.uniform(0, 50)
    out = {}
    for q in qs:
        v += rng.choice([0.0, rng.uniform(0, 300)])
        out[q] = round(v, rng.choice([0, 3, 12]))
    return out


def _charge(rng: random.Random, iid: str = "i") -> Charge:
    t = rng.choice([0.0, rng.uniform(0, 1000), rng.expovariate(1e-3)])
    return Charge.of(
        iid, t, rng.choice([None, 0.0, rng.uniform(0, 200)]), _quantiles(rng)
    )


def _pairs(c: Charge):
    return list(c.quantiles) if c.quantiles else None


def test_charge_at_matches(rc):
    rng = random.Random(1)
    for _ in range(3000):
        c = _charge(rng)
        grid = rng.choice(GRIDS)
        assert rc.charge_at(c.time_ms, _pairs(c), grid) == c.at(grid)


def test_apply_matches(rc):
    rng = random.Random(2)
    for _ in range(3000):
        grid = rng.choice(GRIDS)
        values = sorted(rng.uniform(0, 5000) for _ in grid)
        c = _charge(rng)
        sign = rng.choice([1, -1])
        servers = rng.choice([1, 1, 2, 3, 7, 64])
        prob = InstanceQueueProbabilistic(
            instance_id="i", quantiles=grid, values=values
        )
        want = apply(prob, c, sign, servers)
        got = rc.apply_probabilistic(
            values, grid, c.time_ms, _pairs(c), sign, servers
        )
        assert got == want.values
        exp = InstanceQueueExpectError(
            instance_id="i",
            expected_time_ms=rng.uniform(0, 5000),
            error_margin_ms=rng.uniform(0, 500),
        )
        want = apply(exp, c, sign, servers)
        got = rc.apply_expect_error(
            exp.expected_time_ms,
            exp.error_margin_ms,
            c.time_ms,
            c.error_ms,
            sign,
            servers,
        )
        assert got == (want.expected_time_ms, want.error_margin_ms)


# ---------------------------------------------------------------------------
# Ledger against InstanceRegistry
# ---------------------------------------------------------------------------


def _sketch_tuple(q):
    if isinstance(q, InstanceQueueProbabilistic):
        return ("probabilistic", q.quantiles, q.values)
    return ("expect_error", q.expected_time_ms, q.error_margin_ms)


@pytest.mark.parametrize("kind", ["probabilistic", "expect_error"])
@pytest.mark.parametrize("seed", range(6))
def test_ledger_matches_registry(rc, kind, seed):
    from swarmpilot.scheduler.models import Instance
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry

    async def run() -> None:
        rng = random.Random(f"{kind}-{seed}")
        reg = InstanceRegistry(queue_info_type=kind)
        led = rc.Ledger(kind, list(DEFAULT_QUANTILES))
        ids = [f"i{k}" for k in range(4)]
        live: list[str] = []
        tasks: list[str] = []
        for n in range(400):
            roll = rng.random()
            if not live or roll < 0.05:
                iid = rng.choice(ids)
                if iid in live:
                    continue
                await reg.register(
                    Instance(
                        instance_id=iid,
                        model_id="m",
                        endpoint="http://x",
                        platform_info={},
                    )
                )
                led.register(iid)
                live.append(iid)
            elif roll < 0.08 and len(live) > 1:
                iid = live.pop(rng.randrange(len(live)))
                await reg.remove(iid)
                led.remove(iid)
            elif roll < 0.12:
                iid = rng.choice(live)
                servers = rng.choice([0, 1, 2, 4, 9])
                await reg.set_capacity(iid, servers)
                led.set_capacity(iid, servers)
            elif roll < 0.14:
                new_kind = rng.choice(["probabilistic", "expect_error"])
                grid = rng.choice([None, *GRIDS])
                await reg.reset_queues(new_kind, grid)
                led.reset(new_kind, grid)
            elif roll < 0.40 and tasks:
                t = tasks.pop(rng.randrange(len(tasks)))
                old = await reg.release_charge(t)
                assert led.release(t) == (old.instance_id if old else None)
            elif roll < 0.50 and tasks:
                t = rng.choice(tasks)
                iid = rng.choice(ids)
                assert led.move_charge(t, iid) == await reg.move_charge(t, iid)
            else:
                # Sometimes re-charge an open task, sometimes no task id.
                t = rng.choice([*tasks[-3:], f"t{n}", f"t{n}", None])
                c = _charge(rng, rng.choice(ids))
                await reg.charge(t, c)
                led.charge(t, c.instance_id, c.time_ms, c.error_ms, _pairs(c))
                if t is not None and t not in tasks:
                    tasks.append(t)
            for iid in live:
                want = _sketch_tuple(await reg.get_queue_info(iid))
                assert led.sketch(iid) == want, (n, iid)
                assert led.capacity(iid) == reg.get_capacity(iid)

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Strategy selection, same draws
# ---------------------------------------------------------------------------


def _predictions(rng: random.Random, n: int) -> list[Prediction]:
    out = []
    for k in range(n):
        t = round(rng.uniform(1, 500), rng.choice([0, 3]))
        q = _quantiles(rng)
        out.append(
            Prediction(
                instance_id=f"i{k if rng.random() > 0.05 else 0}",
                predicted_time_ms=t,
                error_margin_ms=rng.uniform(0, 50),
                quantiles=q,
            )
        )
    return out


def _queues(rng: random.Random, preds: list[Prediction], kind: str) -> dict:
    out = {}
    for p in preds:
        if rng.random() < 0.15:
            continue
        if kind == "probabilistic" and rng.random() < 0.9:
            vals = sorted(
                rng.choice([0.0, rng.uniform(0, 3000)])
                for _ in DEFAULT_QUANTILES
            )
            out[p.instance_id] = InstanceQueueProbabilistic(
                instance_id=p.instance_id,
                quantiles=list(DEFAULT_QUANTILES),
                values=vals,
            )
        else:
            out[p.instance_id] = InstanceQueueExpectError(
                instance_id=p.instance_id,
                expected_time_ms=rng.choice([0.0, rng.uniform(0, 3000)]),
                error_margin_ms=rng.uniform(0, 300),
            )
    return out


@pytest.mark.parametrize(
    "name", ["min_time", "serverless", "po2", "probabilistic"]
)
def test_select_instance_matches(use_core, name):
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry

    kind = "probabilistic" if name == "probabilistic" else "expect_error"
    strategies = {}
    for c in ("python", "rust"):
        use_core(c)
        strategies[c] = get_strategy(
            name, None, InstanceRegistry(kind), seed=11
        )
    rng = random.Random(name)
    for _ in range(2000):
        preds = _predictions(rng, rng.randint(0, 9))
        queues = _queues(rng, preds, kind)
        picks = {}
        for c, strat in strategies.items():
            use_core(c)
            # po2 overwrites prediction times; give each core its own copy.
            copy = [Prediction(**vars(p)) for p in preds]
            picks[c] = strat.select_instance(copy, dict(queues))
        assert picks["python"] == picks["rust"], (preds, queues)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

KEYS = ("prompt", "messages", "content", "text", "max_tokens", "size", "n", "a")
STRINGS = (
    "",
    "hello",
    "héllo wörld",
    "日本語のテキスト",
    "😀 emoji",
    "640x480",
    "12X34",
    " 1x1",
    "1x",
    "x9",
    "٣x٣",
    "²x2",
    "1x2x3",
    "9" * 18 + "x" + "9" * 18,
    "9" * 19 + "x1",
    "\ud800x1",
)
NUMBERS = (
    0,
    1,
    -3,
    7.5,
    -0.0,
    2**53 + 1,
    2**70,
    10**400,
    1e308,
    float("nan"),
    float("inf"),
)


def _value(rng: random.Random, depth: int):
    roll = rng.random()
    if depth > 2 or roll < 0.35:
        return rng.choice(
            [*NUMBERS, *STRINGS, True, False, None, rng.uniform(-1e3, 1e3)]
        )
    if roll < 0.65:
        return [_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {
        rng.choice(KEYS): _value(rng, depth + 1)
        for _ in range(rng.randint(0, 4))
    }


def _path(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(1, 3)):
        part = rng.choice(KEYS)
        part += "".join(
            rng.choice(["[*]", "[0]", "[1]", "[3]"])
            for _ in range(rng.randint(0, 2))
        )
        parts.append(part)
    return ".".join(parts)


def _expr(rng: random.Random, depth: int = 0) -> str:
    roll = rng.random()
    if depth > 2 or roll < 0.25:
        return rng.choice(
            [_path(rng), str(rng.choice([0, 1, 30, 1024, -2])), "1e3", "0.5"]
        )
    if roll < 0.6:
        fn = rng.choice(["field", "len", "count", "chars", "area"])
        dflt = (
            f", default={rng.choice(['7', '-1.5', '2e2'])}"
            if rng.random() < 0.2
            else ""
        )
        return f"{fn}({_path(rng)}{dflt})"
    fn = rng.choice(["product", "sum", "min", "max", "coalesce"])
    args = ", ".join(_expr(rng, depth + 1) for _ in range(rng.randint(1, 3)))
    return f"{fn}({args})"


def _same(a: float, b: float) -> bool:
    return a == b and math.copysign(1, a) == math.copysign(1, b)


def test_extractors_match_on_random_bodies(use_core):
    rng = random.Random(3)
    exprs = [_expr(rng) for _ in range(300)]
    bodies = [_value(rng, 0) for _ in range(150)]
    bodies += [{k: _value(rng, 1) for k in KEYS} for _ in range(150)]
    checked = 0
    for expr in exprs:
        use_core("python")
        py = compile_extractor(expr, 0.25)
        use_core("rust")
        rs = compile_extractor(expr, 0.25)
        for body in bodies:
            want, got = py(body), rs(body)
            assert _same(want, got), (expr, body, want, got)
            checked += 1
    assert checked == 300 * 300


def test_extractors_match_on_json_text(use_core):
    """The gateway path: Rust parses the raw JSON body itself."""
    rng = random.Random(4)
    use_core("rust")
    for _ in range(3000):
        expr = _expr(rng)
        body = {k: _value(rng, 1) for k in rng.sample(KEYS, 4)}
        try:
            text = json.dumps(body, allow_nan=False)
        except ValueError:
            continue
        # Numbers beyond a float are a parse error in Rust (the gateway
        # rejects such bodies); Python reads them as missing.
        if "\\ud800" in text or str(10**400) in text:
            continue
        use_core("python")
        want = compile_extractor(expr)(json.loads(text))
        use_core("rust")
        got = compile_extractor(expr).eval_json(text)
        assert _same(want, got), (expr, text)


GARBAGE = (
    "",
    " ",
    "(",
    ")",
    "a(",
    "sum(",
    "sum()",
    "sum(1,)",
    "sum(1 2)",
    "field()",
    "field(1)",
    "field(a, default=x)",
    "field(a, dflt=1)",
    "field(a, default=1",
    "a..b",
    "a[",
    "a[x]",
    "a[-1]",
    ".a",
    "a.",
    "nope(a)",
    "1 2",
    "a b",
    "-",
    "--1",
    "1.",
    ".5",
    "1e",
    "日本(a)",
    "fïeld(a)",
    "a\u00a0",
    "\u3000a",
    "a\x1c",
    "chars(messages[*].content) ",
    "\tcoalesce(\nmax_tokens ,1024 )",
    "sum(a,b,c,d,e,f)",
    "a-b",
    "a*",
    "a[*][*]",
    "a[0].b[*]",
    "field(a,default=-1e-3)",
)


@pytest.mark.parametrize("expr", GARBAGE)
def test_extractor_parse_errors_match(use_core, expr):
    results = {}
    for c in ("python", "rust"):
        use_core(c)
        try:
            fn = compile_extractor(expr)
            results[c] = fn({"a": {"b": [1, 2]}, "b": 3.5, "c": "xy"})
        except ExtractorError:
            results[c] = "error"
    assert results["python"] == results["rust"]


def test_random_strings_parse_the_same(use_core):
    rng = random.Random(5)
    alphabet = (
        "abcfieldsumproductcountchars()[]*.,=-+0123456789 e\t\u00e9\u3000"
    )
    for _ in range(5000):
        expr = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24)))
        out = {}
        for c in ("python", "rust"):
            use_core(c)
            try:
                out[c] = compile_extractor(expr)({"a": [1, "xy"], "sum": 2})
            except ExtractorError:
                out[c] = "error"
        assert out["python"] == out["rust"] or (
            isinstance(out["python"], float)
            and _same(out["python"], out["rust"])
        ), expr


def test_numpy_draws_reach_rust_unchanged(use_core):
    """Both cores draw the same 10 numbers per probabilistic decision."""
    from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry

    seen = {}
    for c in ("python", "rust"):
        use_core(c)
        strat = get_strategy("probabilistic", None, InstanceRegistry(), seed=5)
        preds = _predictions(random.Random(9), 5)
        for _ in range(20):
            strat.select_instance(preds, {})
        seen[c] = strat._rng.bit_generator.state["state"]["state"]
    assert seen["python"] == seen["rust"]
    assert (
        np.random.default_rng(5).bit_generator.state["state"]["state"]
        != seen["python"]
    )

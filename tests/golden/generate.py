"""Golden vectors: recorded inputs and outputs of the Python data plane.

A port of the scheduling core (the Rust gateway of the migration plan) must
reproduce these files exactly. ``test_golden.py`` replays them against the
Python code in CI, so a change in behaviour shows up as a diff here.

Three files, all plain JSON:

``sketches.json``
    Charge, release and move operations on queue sketches of both kinds,
    with every sketch after every operation.
``strategies.json``
    For each of the seven strategies, a run of routing and completion steps
    on four instances over two platforms. Each step lists the predictions
    the predictor returned, the random draws the strategy made (so a port
    can replay them instead of reproducing numpy's generator), the chosen
    instance and its sketch afterwards; a completion step lists the sketch
    it shrank.
``features.json``
    Predictor features for each built-in engine adapter on example and
    edge-case request bodies, plus each adapter's extractor expressions.

Regenerate after an intended behaviour change, and review the diff::

    uv run python tests/golden/generate.py
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from swarmpilot.engines import adapter_names, get_adapter
from swarmpilot.engines.manifest import manifest_features
from swarmpilot.scheduler.algorithms.factory import get_strategy
from swarmpilot.scheduler.clients.models import Prediction
from swarmpilot.scheduler.models import (
    Instance,
    InstanceQueueExpectError,
    InstanceQueueProbabilistic,
)
from swarmpilot.scheduler.registry.instance_registry import InstanceRegistry
from swarmpilot.scheduler.registry.queue_sketch import (
    DEFAULT_QUANTILES,
    Charge,
)

GOLDEN_DIR = Path(__file__).parent
VERSION = 1

STRATEGIES = (
    "adaptive_bootstrap",
    "min_time",
    "po2",
    "probabilistic",
    "random",
    "round_robin",
    "serverless",
)
EXPECT_ERROR_QUEUES = {"min_time", "po2", "serverless"}

PLATFORM_A = {
    "software_name": "vllm",
    "software_version": "0.11",
    "hardware_name": "gpu-a",
}
PLATFORM_B = {**PLATFORM_A, "hardware_name": "gpu-b"}
INSTANCES = (
    ("i0", PLATFORM_A),
    ("i1", PLATFORM_A),
    ("i2", PLATFORM_B),
    ("i3", PLATFORM_B),
)
# adaptive_bootstrap sees a trained model for platform A only.
WARM_HARDWARE = {"gpu-a"}


def _sketch(queue: Any) -> dict[str, Any]:
    if isinstance(queue, InstanceQueueProbabilistic):
        return {"quantiles": queue.quantiles, "values": queue.values}
    if isinstance(queue, InstanceQueueExpectError):
        return {
            "expected_time_ms": queue.expected_time_ms,
            "error_margin_ms": queue.error_margin_ms,
        }
    return {}


async def _sketches(reg: InstanceRegistry) -> dict[str, Any]:
    return {k: _sketch(v) for k, v in (await reg.get_all_queue_info()).items()}


def _instances() -> list[Instance]:
    return [
        Instance(
            instance_id=iid,
            model_id="m",
            endpoint=f"http://{iid}",
            platform_info=plat,
        )
        for iid, plat in INSTANCES
    ]


def _quantiles(rng: random.Random, median: float) -> dict[float, float]:
    """A monotone quantile set around ``median``, rounded to 3 places."""
    spread = rng.uniform(0.1, 0.8)
    out = {}
    for q in DEFAULT_QUANTILES:
        z = (q - 0.5) * 2
        out[q] = round(median * (1 + spread * z * (2 if z > 0 else 1)), 3)
    return out


# ---------------------------------------------------------------------------
# Sketches
# ---------------------------------------------------------------------------


async def build_sketches() -> dict[str, Any]:
    """Charge, release and move ops on both sketch kinds."""
    cases = []
    for kind in ("probabilistic", "expect_error"):
        rng = random.Random(f"sketch-{kind}")
        reg = InstanceRegistry(queue_info_type=kind)
        for inst in _instances()[:2]:
            await reg.register(inst)
        open_: list[str] = []
        steps = []
        for n in range(60):
            roll = rng.random()
            if open_ and roll < 0.3:
                task = open_.pop(rng.randrange(len(open_)))
                op = {"op": "release", "task_id": task}
                await reg.release_charge(task)
            elif open_ and roll < 0.4:
                task = rng.choice(open_)
                target = rng.choice(["i0", "i1"])
                op = {"op": "move", "task_id": task, "instance_id": target}
                await reg.move_charge(task, target)
            else:
                task = f"t{n}"
                median = round(rng.uniform(10, 500), 3)
                quantiles = (
                    _quantiles(rng, median) if rng.random() < 0.8 else None
                )
                charge = {
                    "instance_id": rng.choice(["i0", "i1"]),
                    "time_ms": median,
                    "error_ms": round(median * rng.uniform(0, 0.5), 3),
                    "quantiles": (
                        [[q, v] for q, v in quantiles.items()]
                        if quantiles
                        else None
                    ),
                }
                op = {"op": "charge", "task_id": task, "charge": charge}
                await reg.charge(
                    task,
                    Charge.of(
                        charge["instance_id"],
                        charge["time_ms"],
                        charge["error_ms"],
                        quantiles,
                    ),
                )
                open_.append(task)
            steps.append({**op, "sketches": await _sketches(reg)})
        for task in open_:
            await reg.release_charge(task)
        steps.append({"op": "release_all", "sketches": await _sketches(reg)})
        cases.append(
            {
                "kind": kind,
                "quantile_grid": list(DEFAULT_QUANTILES),
                "steps": steps,
            }
        )
    return {"version": VERSION, "cases": cases}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class RecordingRNG:
    """Wraps a numpy Generator and records every draw."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.draws: list[dict[str, Any]] = []

    def random(self, size: int | None = None) -> Any:
        out = self._rng.random(size)
        self.draws.append(
            {
                "op": "random",
                "size": size,
                "out": out.tolist() if size is not None else float(out),
            }
        )
        return out

    def integers(self, high: int) -> int:
        out = int(self._rng.integers(high))
        self.draws.append({"op": "integers", "high": high, "out": out})
        return out

    def choice(
        self, n: int, size: int | None = None, replace: bool = True
    ) -> Any:
        out = self._rng.choice(n, size=size, replace=replace)
        self.draws.append(
            {
                "op": "choice",
                "n": n,
                "size": size,
                "replace": replace,
                "out": out.tolist(),
            }
        )
        return out


class _Storage:
    def generate_model_key(
        self, model_id: str, platform_info: dict, prediction_type: str
    ) -> str:
        return platform_info["hardware_name"]

    def model_exists(self, key: str) -> bool:
        return key in WARM_HARDWARE


class _LowLevel:
    _storage = _Storage()


class ScriptedPredictor:
    """Returns the predictions scripted for the current step."""

    _low_level = _LowLevel()

    def __init__(self) -> None:
        self.table: dict[str, dict[str, Any]] = {}
        self.calls: list[str] = []

    async def predict(
        self,
        model_id: str,
        metadata: dict[str, Any],
        instances: list[Instance],
        prediction_type: str,
        quantiles: list[float] | None = None,
    ) -> list[Prediction]:
        self.calls.append(prediction_type)
        out = []
        for inst in instances:
            row = self.table[inst.instance_id]
            out.append(
                Prediction(
                    instance_id=inst.instance_id,
                    predicted_time_ms=row["predicted_time_ms"],
                    error_margin_ms=(
                        row["error_margin_ms"]
                        if prediction_type == "expect_error"
                        else None
                    ),
                    quantiles=(
                        dict(row["quantiles"])
                        if prediction_type == "quantile"
                        else None
                    ),
                )
            )
        return out


async def _run_strategy(name: str) -> dict[str, Any]:
    kind = "expect_error" if name in EXPECT_ERROR_QUEUES else "probabilistic"
    reg = InstanceRegistry(queue_info_type=kind)
    insts = _instances()
    for inst in insts:
        await reg.register(inst)
    predictor = ScriptedPredictor()
    strategy = get_strategy(name, predictor, reg, seed=7)
    recorders = [RecordingRNG(strategy._rng)]
    strategy._rng = recorders[0]
    inner = getattr(strategy, "_probabilistic", None)
    if inner is not None:
        recorders.append(RecordingRNG(inner._rng))
        inner._rng = recorders[1]

    rng = random.Random(f"strategy-{name}")
    open_: list[str] = []
    steps = []
    for n in range(30):
        if open_ and rng.random() < 0.35:
            task = open_.pop(rng.randrange(len(open_)))
            charge = await reg.release_charge(task)
            iid = charge.instance_id if charge else None
            steps.append(
                {
                    "op": "finish",
                    "task_id": task,
                    "instance_id": iid,
                    "sketch": (
                        _sketch(await reg.get_queue_info(iid)) if iid else None
                    ),
                }
            )
            continue
        table = {}
        for inst in insts:
            median = round(rng.uniform(20, 400), 3)
            table[inst.instance_id] = {
                "predicted_time_ms": median,
                "error_margin_ms": round(median * rng.uniform(0.05, 0.4), 3),
                "quantiles": [
                    [q, v] for q, v in _quantiles(rng, median).items()
                ],
            }
        predictor.table = {
            k: {**v, "quantiles": [tuple(p) for p in v["quantiles"]]}
            for k, v in table.items()
        }
        predictor.calls = []
        for r in recorders:
            r.draws = []
        task = f"t{n}"
        result = await strategy.schedule_task("m", {}, insts, task_id=task)
        open_.append(task)
        steps.append(
            {
                "op": "route",
                "task_id": task,
                "predictions": table,
                "prediction_types": predictor.calls,
                "draws": [d for r in recorders for d in r.draws],
                "selected": result.selected_instance_id,
                "charged": (
                    {
                        "time_ms": result.selected_prediction.predicted_time_ms,
                        "error_ms": result.selected_prediction.error_margin_ms,
                        "has_quantiles": bool(
                            result.selected_prediction.quantiles
                        ),
                    }
                    if result.selected_prediction is not None
                    else None
                ),
                "sketch": _sketch(
                    await reg.get_queue_info(result.selected_instance_id)
                ),
            }
        )
    return {
        "strategy": name,
        "queue_kind": kind,
        "instances": [
            {"instance_id": iid, "platform_info": plat}
            for iid, plat in INSTANCES
        ],
        "warm_hardware": sorted(WARM_HARDWARE),
        "steps": steps,
    }


async def build_strategies() -> dict[str, Any]:
    """Routing runs for all seven strategies."""
    return {
        "version": VERSION,
        "quantile_grid": list(DEFAULT_QUANTILES),
        "runs": [await _run_strategy(name) for name in STRATEGIES],
    }


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

LLM_PROBES: tuple[dict[str, Any], ...] = (
    {},
    {"prompt": "hello world", "max_tokens": 32},
    {"prompt": ["ab", "c", 7], "max_tokens": True},
    {
        "messages": [
            {"role": "user", "content": "abc"},
            {"role": "assistant", "content": [{"type": "text", "text": "de"}]},
            {"content": None},
            "junk",
        ],
        "max_completion_tokens": 7,
    },
    {"prompt": "x", "messages": "nope", "max_tokens": 2.5},
)
IMAGE_PROBES: tuple[dict[str, Any], ...] = (
    {},
    {"size": "640x480", "num_inference_steps": 8, "n": 2},
    {"width": 512, "height": 256.0, "size": "1x1"},
    {"width": 512, "size": "big", "n": "3"},
)


def build_features() -> dict[str, Any]:
    """Features per built-in adapter on example and edge-case bodies."""
    adapters = []
    for name in adapter_names():
        adapter = get_adapter(name)
        manifest = adapter.manifest()
        probes = (
            IMAGE_PROBES
            if str(adapter.workload.type).startswith("diffusion")
            else LLM_PROBES
        )
        path = adapter.routes()[0].path if adapter.routes() else ""
        cases = [
            {"path": p, "body": b} for p, b in adapter.example_requests()
        ] + [{"path": path, "body": b} for b in probes]
        for case in cases:
            case["features"] = adapter.features(case["path"], case["body"])
            case["manifest_features"] = manifest_features(
                manifest, case["body"]
            )
        adapters.append(
            {
                "adapter": name,
                "workload": str(adapter.workload.type),
                "feature_names": list(adapter.workload.feature_names),
                "extractors": manifest["features"],
                "cases": cases,
            }
        )
    return {"version": VERSION, "adapters": adapters}


# ---------------------------------------------------------------------------


def build_all() -> dict[str, dict[str, Any]]:
    """Every golden file's content, by file name."""
    from loguru import logger

    logger.disable("swarmpilot")
    # A private loop: asyncio.run would clear the thread's current loop,
    # which older synchronous tests still rely on.
    loop = asyncio.new_event_loop()
    try:
        return {
            "sketches.json": loop.run_until_complete(build_sketches()),
            "strategies.json": loop.run_until_complete(build_strategies()),
            "features.json": build_features(),
        }
    finally:
        loop.close()
        logger.enable("swarmpilot")


_NESTED = ("adapters", "cases", "runs", "steps")


def _fmt(value: Any, depth: int = 0) -> str:
    """JSON with one step, case or run record per line."""
    pad = "\n" + " " * (depth + 1)
    if isinstance(value, dict) and (
        depth == 0 or any(k in value for k in _NESTED)
    ):
        items = [
            f"{json.dumps(k)}: {_fmt(v, depth + 1)}"
            for k, v in sorted(value.items())
        ]
        return "{" + pad + ("," + pad).join(items) + "\n" + " " * depth + "}"
    if isinstance(value, list) and value and isinstance(value[0], dict):
        items = [_fmt(v, depth + 1) for v in value]
        return "[" + pad + ("," + pad).join(items) + "\n" + " " * depth + "]"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def dump(content: dict[str, Any]) -> str:
    """Serialise one golden file."""
    return _fmt(content) + "\n"


def main() -> None:
    """Write every golden file next to this script."""
    for name, content in build_all().items():
        (GOLDEN_DIR / name).write_text(dump(content))
        sys.stdout.write(f"wrote {GOLDEN_DIR / name}\n")


if __name__ == "__main__":
    main()

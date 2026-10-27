"""Unit tests for shadow mode's draw recording and comparison."""

import asyncio
import json

import httpx
import numpy as np
import respx

from swarmpilot.scheduler.shadow import RecordingRng, ShadowMirror


def test_draws_are_recorded_only_while_recording():
    rng = RecordingRng(np.random.default_rng(0))
    plain = np.random.default_rng(0)
    mirror = ShadowMirror("http://gw")
    rng.random(3)
    plain.random(3)
    with mirror.recording() as draws:
        xs = rng.random(4)
        k = rng.integers(5)
        pair = rng.choice(6, size=2, replace=False)
    # The wrapped generator draws exactly what a plain one would.
    assert xs.tolist() == plain.random(4).tolist()
    assert k == plain.integers(5)
    assert pair.tolist() == plain.choice(6, size=2, replace=False).tolist()
    assert draws == [
        {"random": xs.tolist()},
        {"integers": int(k)},
        {"choice2": pair.tolist()},
    ]
    rng.random(2)
    assert len(draws) == 3


def _decide(mirror: ShadowMirror, task_id: str, python: str) -> None:
    mirror.decided(
        task_id=task_id,
        path="v1/completions",
        body={"prompt": "x"},
        instance_id=python,
        draws=[],
        decide_us=10.0,
        strategy="MinimumExpectedTimeStrategy",
    )


@respx.mock
async def test_events_go_in_order_and_are_compared():
    sent = []

    def answer(request: httpx.Request) -> httpx.Response:
        events = json.loads(request.content)["events"]
        sent.extend(events)
        results = []
        for ev in events:
            if ev["op"] == "decide":
                results.append(
                    {"instance_id": "a" if ev["task_id"] != "t2" else "b"}
                )
            else:
                results.append({})
        return httpx.Response(200, json={"results": results})

    respx.post("http://gw/_gateway/shadow").mock(side_effect=answer)
    mirror = ShadowMirror("http://gw")
    _decide(mirror, "t1", "a")
    mirror.released("t1")
    mirror.released("never-mirrored")
    _decide(mirror, "t2", "a")
    mirror.start()
    for _ in range(100):
        if mirror.decisions == 2:
            break
        await asyncio.sleep(0.01)
    await mirror.stop()
    assert [(e["op"], e["task_id"]) for e in sent] == [
        ("decide", "t1"),
        ("release", "t1"),
        ("decide", "t2"),
    ]
    # What Python chose stays in the Scheduler.
    assert "python" not in sent[0]
    s = mirror.summary()
    assert (s["matched"], s["mismatched"], s["mismatch_rate"]) == (1, 1, 0.5)
    assert s["recent_mismatches"][0]["gateway"] == "b"


async def test_a_full_queue_drops_and_counts():
    mirror = ShadowMirror("http://gw", max_queue=1)
    _decide(mirror, "t1", "a")
    _decide(mirror, "t2", "a")
    mirror.released("t2")  # t2 was dropped, so it is not mirrored
    assert mirror.summary()["dropped"] == 1
    assert mirror.summary()["queued"] == 1

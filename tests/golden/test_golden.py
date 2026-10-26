"""Replay the golden vectors against the Python code (plan item 2.6).

If a behaviour change is intended, regenerate with
``uv run python tests/golden/generate.py`` and review the diff.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from generate import GOLDEN_DIR, build_all, dump

BUILT = build_all()


@pytest.mark.parametrize("name", sorted(BUILT))
def test_golden_file_matches_code(name):
    committed = (GOLDEN_DIR / name).read_text()
    if committed != dump(BUILT[name]):
        # Show the first differing record rather than the whole file.
        want, got = json.loads(committed), json.loads(dump(BUILT[name]))
        assert got == want, f"{name} differs; regenerate if intended"


def test_strategies_cover_all_seven():
    runs = BUILT["strategies.json"]["runs"]
    assert len({r["strategy"] for r in runs}) == 7


def test_every_sketch_ends_empty():
    for case in BUILT["sketches.json"]["cases"]:
        last = case["steps"][-1]["sketches"]
        for sketch in last.values():
            assert not any(sketch.get("values", [])), sketch
            assert not sketch.get("expected_time_ms"), sketch


def test_manifest_features_match_python():
    for adapter in BUILT["features.json"]["adapters"]:
        for case in adapter["cases"]:
            if adapter["extractors"] is not None:
                assert case["manifest_features"] == case["features"], (
                    adapter["adapter"],
                    case,
                )

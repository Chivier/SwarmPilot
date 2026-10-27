# Rust scheduling core

SwarmPilot is moving its per-request data plane to Rust (plan items 4.7,
5.0 and 8.6 of the production readiness plan). This page covers what is in
Rust today, how to turn it on, and how the project keeps the Rust and
Python implementations from drifting apart.

## What is in Rust

`rust/crates/core` (crate `swarmpilot-core`) holds the scheduling core:

| Rust module | Python original |
|-------------|-----------------|
| `sketch` | `scheduler/registry/queue_sketch.py` and the charge ledger of `InstanceRegistry` |
| `strategy` | the seven strategies in `scheduler/algorithms/` |
| `extract` | the feature extractor grammar in `engines/extractors.py` |
| `interp`, `fsum` | bit-exact ports of `numpy.interp` and `math.fsum` |
| `mlp` | inference of `predictor/predictor/expect_error.py` and `quantile.py` (the MLP, scaling, the quantile head), and a reader for the portable model file |

`rust/crates/core-py` wraps it as the Python module `swarmpilot_core`
(PyO3, abi3 for Python 3.11+).

## Turning it on

```bash
uv sync --extra core          # builds swarmpilot_core with maturin (needs cargo)
SCHEDULER_CORE=rust sscheduler start --port 8000
```

`SCHEDULER_CORE` is `python` (default) or `rust`. With `rust`, the queue
sketch updates, the selection step of min_time, serverless, po2 and
probabilistic (and so adaptive_bootstrap), and every compiled extractor
expression run in Rust, and so does the forward pass of every trained
runtime predictor. The scheduler logs `Scheduling core: rust` at
startup, and refuses to start if the module is missing.

The strategies still draw their random numbers from their own numpy
generators and pass them to Rust, so a seeded run makes the same
decisions on either core. Measured on a 4-vCPU container: probabilistic
selection over 8 instances 50 µs (Python) vs 14 µs (Rust), a sketch
update 4.7 vs 1.9 µs, the LLM prompt-length extractor 2.7 vs 0.8 µs.

### Predictor inference

Training, feature validation and feature preprocessing stay in Python.
With the Rust core, `ExpectErrorPredictor.predict()` and
`QuantilePredictor.predict()` hand the ordered feature vector to a
`swarmpilot_core.RuntimePredictor` built from the predictor's own state
(`predictor/predictor/native.py`); it is rebuilt whenever training or
loading replaces the network. `RuntimePredictor.load_portable()` reads
the portable model file that `ModelStorage` writes beside the joblib
file (`{key}.safetensors` plus the `{key}.json` sidecar), checks the
weights' SHA-256, and needs no Python, for the R3 gateway.

The port reproduces torch's arithmetic: float32 layers, softplus with
threshold 20, the quantile cumulative sum accumulated in float64 as
torch does on CPU, and target scaling in float32. Feature scaling is
float32 right after training and float64 after a reload, matching the
numpy dtypes the Python predictor holds in each case. Unlike the
scheduling core this is not bit-exact (the float32 dot products are
summed in a different order), so the bar is a relative error of 1e-5;
the worst seen on the recorded models is 2e-6.

One `predict()` call: ExpectError 45 µs (torch) vs 3.3 µs (Rust),
Quantile 74 vs 6.5 µs. The Rust call itself takes 1.0 and 1.5 µs; the
rest is Python feature validation and ordering.

## How drift is prevented

The Rust core must give the same answers as the Python code, to the last
bit, not approximately. Four layers check it, all in CI (job `rust-core`):

1. **Golden vectors** (`tests/golden/*.json`), recorded from the Python
   code: queue sketch ops, 30-step runs of all seven strategies with their
   recorded random draws, and extractor features per adapter. Python
   replays them in `tests/golden/test_golden.py`; Rust replays them in
   `rust/crates/core/tests/golden.rs`. The golden generator also runs
   with `SCHEDULER_CORE=rust` and must write identical files.
2. **Differential tests** (`tests/core/test_parity.py`): seeded random
   inputs fed to the Python function and its Rust counterpart side by
   side. Sketch updates, the full ledger against `InstanceRegistry` over
   random register/charge/release/move/capacity/reset sequences, the four
   selection functions, 90,000 extractor evaluations on random bodies
   (Unicode, bools, huge ints, NaN), raw-JSON evaluation, and parse
   errors on malformed expressions.
3. **Routing simulation A/B** (`tests/core/test_sim_parity.py`): the
   discrete-event sim in `tests/perf/sim_routing.py` routes 600 tasks with
   each strategy on both cores; every task must go to the same instance.
4. **Whole suites on the Rust core**: the scheduler, engine, API,
   conformance and regression suites, and the emulated cluster
   (`tests/sim/`, real Planner and Scheduler on simulated engines), all
   run with `SCHEDULER_CORE=rust`.

The predictor port has its own checks, also in the `rust-core` job:

- **Recorded torch outputs** (`tests/golden/predictor/`): an
  expect-error and a log-target quantile model in the portable format,
  with torch's answers on 43 inputs (zeros, negatives, values far
  outside the training range). Rust (`predictor_matches_torch`) and the
  Python predictor (`tests/golden/test_predictor_golden.py`) must both
  match within 1e-5; Rust also rejects a weights file whose SHA-256
  does not match the sidecar.
- **Differential tests** (`tests/core/test_predictor_parity.py`): five
  architectures trained on random data, compared on 150 random inputs
  each right after training, after a save and reload, and loaded from
  the portable file by Rust alone; retraining must rebuild the Rust copy.
- **Routing simulation with trained predictors**
  (`tests/core/test_sim_predictor_parity.py`): quantile and expect-error
  models trained in the sim drive probabilistic, adaptive_bootstrap,
  min_time and po2 on both cores. At least 98% of tasks must go to the
  same instance (a prediction can differ by an ulp and flip a near tie),
  P50, P95 and P99 latency must agree within 5%, and no task may be
  left over.

Changing core behaviour therefore means changing both implementations,
regenerating the golden files (`uv run python tests/golden/generate.py`)
and reviewing the diff.

### Python changes made for exact parity

Porting found places where the Python result depended on the platform or
crashed. Both cores now behave as follows:

- Expect-error sketches square with `x * x`, not `x ** 2`: libm's `pow`
  differs from the correctly rounded product in about 1 of 1,000 cases.
- Extractor expressions are ASCII: names, paths and numbers no longer
  match Unicode letters and digits.
- `area("WxH")` takes 1 to 18 ASCII digits per side; before, `"²x2"`
  raised inside the request path and `"٣x٣"` gave 9.
- A number too large for a float, and a `sum()` that overflows, count as
  missing instead of raising.

A raw JSON body with a number beyond the float range is a parse error in
Rust (`Extractor.eval_json`); the gateway will reject such bodies.
Rust parses JSON with serde_json's `float_roundtrip` feature, without
which some decimals parse one bit off.

## Phases

| Phase | Plan item | Content | Drift guard |
|-------|-----------|---------|-------------|
| R1 (done) | 4.7, part of 1.8 | Core crate, PyO3 module, `SCHEDULER_CORE` | Golden vectors, differential tests, sim A/B, suites on Rust |
| R2 (done) | 4.7 | Predictor inference in Rust: load the safetensors + JSON model files (1.7) and run the MLPs; preprocessing stays in Python | Recorded predictions within 1e-5 of torch on every model kind; differential tests; routing sim with trained predictors, P95 within 5% |
| R3 | 5.0, 1.8 | `swarmpilot-gateway` binary (proxy, routing with its own RNG, dispatch, SSE, API style translation) and the control-channel protos, after the 0.1 go/no-go benchmark | Conformance and API fixtures against both the Python Scheduler and the gateway; emulated-cluster A/B on the 256-instance spec; shadow mode (5.8) |
| R4 | 8.6 | Published `swarmpilot-core` wheels, static gateway binary, signatures and SBOM | Wheel and binary run the same CI suites |

`SCHEDULER_CORE` defaults to `python` until R2 lands and the Gate A
checks pass on the Rust core.

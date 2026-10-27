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
| `experiment` | experiment-mode predictions of `predictor/utils/experiment.py` (`exp_*` request fields), bit-exact |

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

## The gateway (R3)

`rust/crates/gateway` builds `swarmpilot-gateway`, the Scheduler's data
plane in Rust. It serves the public port in place of the Python
catch-all proxy: it reads the request, computes the adapter's features,
routes with the Rust core and its own predictor copy, and dispatches to
the engine with the same per-instance queues. The Python Scheduler keeps
everything else (registration, health, draining, training, the Planner,
the `/v1/...` API) and talks to the gateway over gRPC on a Unix socket
(`proto/swarmpilot/gateway/v1/control.proto`):

- `WatchState`: the Scheduler streams the routing state (instances and
  their status, limits and headers; strategy; adapter manifests; proxy
  timeouts; predictor model keys, hardware features and experiment
  draws). It rebuilds it every 0.2 s and sends it when it changed.
- `ReportCompletions`: every finished request goes back in batches
  (100 ms), with its features and prediction, so instance statistics,
  Planner throughput, arrival counts and training samples work as with
  the Python proxy.
- `ReportUnreachable`: a refused connection is handled by the
  Scheduler's `_handle_unreachable` (the instance becomes UNHEALTHY).

The gateway loads trained models from the predictor storage directory
(portable files, R2) and reloads them when a retrain rewrites them.

```bash
cargo build --release --manifest-path rust/Cargo.toml
SCHEDULER_GATEWAY_SOCKET=/run/swarmpilot/gw.sock sscheduler start --port 8000
rust/target/release/swarmpilot-gateway \
    --listen 0.0.0.0:8080 --control /run/swarmpilot/gw.sock
```

Clients then call port 8080; registration and the rest of the API stay
on 8000. `GET /_gateway/health` shows the state version, model,
strategy and instance count. `GATEWAY_SEED` seeds the routing RNG;
`GATEWAY_LOG` sets the log filter (`info` by default).

### Overhead

`tests/perf/bench_proxy.py --data-plane {python,gateway}`, one fake
engine, adaptive_bootstrap, 4-vCPU container (baselines in
`tests/perf/baselines/proxy-r3a-*.json`). Added latency over calling the
engine directly:

| Concurrency | Python P50 / P99 | Gateway P50 / P99 | Throughput vs direct (Python / gateway) |
|-------------|------------------|-------------------|------------------------------------------|
| 1 | 1.1 / 1.8 ms | 0.3 / 0.5-0.7 ms | 0.15 / 0.36 |
| 64 | 22 / 125 ms | 1.0 / 2.0 ms | 0.17 / 0.81 |
| 512 | 196 / 426 ms | 9 / 30 ms | 0.16 / 0.82 |

The single-process Python fake engine limits throughput at high
concurrency; the added P99 at concurrency 1 sits at the 0.5 ms budget
of plan item 0.1, within this machine's noise.

### Drift guards for the gateway

- **Dispatch conformance** (`tests/conformance/test_dispatch_conformance.py`):
  every black-box test runs against the Python proxy and the gateway.
- **Control channel end to end** (`tests/conformance/test_gateway_control.py`):
  503 without instances, registration and strategy switches reaching the
  gateway, completions counted on the instance, samples from gateway
  traffic training a model the gateway then loads, an unreachable
  instance marked UNHEALTHY in the Scheduler while requests succeed.
- **Routing A/B** (`tests/conformance/test_routing_ab.py`): 18 requests
  with experiment-mode predictions queue up on three engines; with
  min_time, serverless and round_robin the gateway must choose the same
  instance as the Python proxy for every request.
- **Fallbacks** (`strategy.rs` tests): a failed prediction gives the
  first instance uncharged; round_robin and random use 0 ms placeholders;
  adaptive_bootstrap keeps its round-robin pick.
- **Experiment mode** (`tests/core/test_experiment_parity.py`): 1,200
  random parameter sets and the error cases, equal to the last bit
  (CPython 3.12+ `sum()` compensation is emulated; 3.11 within 1e-15).
- **Scheduler side** (`tests/scheduler/test_gateway_control.py`): the
  state built from the registry and the report sinks.

### API style translation (R3b)

`swarmpilot_core::api` is a port of `swarmpilot/api` (canonical model,
SSE parser, the three built-in styles and `plan_request`), and the
gateway uses it as the Python proxy uses the original: a client may call
`/v1/chat/completions`, `/v1/messages` or `/v1/responses` whatever the
engine speaks, answers and streams included. Refused fields get the same
400 in the client's style, dropped ones the same `x-swarmpilot-dropped`
header; `SWARMPILOT_API_TRANSLATION=lenient` on the Scheduler reaches the
gateway in the state (`translation_lenient`), and each adapter's
`default_max_tokens` in its manifest.

The Python styles read loosely typed JSON, so the port reproduces Python
semantics rather than idiomatic Rust (`api/py.rs`): truthiness, `dict.get`
with and without defaults, `AttributeError` on `.get` of a non-object,
`TypeError` on an unhashable key, `str()` and `repr()` with Python's
float repr, `int()`, `json.dumps(ensure_ascii=False)`, and iteration
over strings and dicts. Ids and the clock come through an `Env`, called
in the same order as Python's generators call `new_id` and `now`. An
exception Python would raise becomes a 500 (`Internal Server Error`), a
failing engine stream an `Engine stream failed` error event, and a
failing client encoder a cut-off stream, as in Python.

Drift guards:

- **Differential fuzzing** (`tests/core/test_translation_parity.py`):
  random requests in each style, half of them with fields replaced by
  values of random types or removed, planned against every engine style;
  random engine answers and error bodies; random SSE streams, mutated,
  cut short anywhere (inside UTF-8 characters too), with keep-alives,
  bad JSON, CRLF and random chunk splits, ending normally or in an
  error. Both sides run with the same counting ids and fixed clock and
  must give the same plan, body, dropped list and answer bytes. About
  6,000 cases per CI run; `SWARMPILOT_PARITY_SCALE=40` ran 180,000 with
  no difference.
- **Translation matrix on both data planes**
  (`tests/api/test_conformance_matrix.py`): every client SDK against
  every engine style, text and tools, streamed and not, plus the
  tool-result round trip and the refusal and drop cases.
- **Byte A/B** (`test_gateway_answers_like_python` in the same file):
  the same requests through the Python proxy and the gateway in front of
  the same Scheduler; status, content type, dropped header and body must
  match once ids and timestamps are masked.

Known differences, all on input no client sends: JSON `NaN` and
`Infinity` literals and lone surrogates in strings are refused by the
gateway's parser (Python accepts them); integers beyond 64 bits are
refused too; `repr()` of strings with unusual non-printable characters
in error messages may differ. API styles added as Python plugins are not
in the gateway: requests that would be translated to or from one get 501.

Porting found one Python bug, fixed in both: a translated stream stops
reading the engine at its last event (`[DONE]`, `message_stop`, ...), so
the engine's body was never read to the end and the request counted as
failed (no throughput, no training sample). `Translation.stream_response`
now reads the rest of the engine's body after the client's stream ends.

### Not yet in the gateway (R3a)

- API styles that exist only as Python plugins (501, see above).
- Feature adapters written only in Python (no extractor manifest) give
  no features, so those requests route without predictions; the
  pass-through adapter is ported.
- A preprocessor chain configured for the model disables trained
  predictions in the gateway (routing falls back as on a cold model).
- Requests are not in the Scheduler's task registry (`/v1/task/...`).
- Strategies that draw random numbers (probabilistic, po2, random,
  adaptive_bootstrap with trained models) use the gateway's own RNG, so
  their decisions match Python in distribution, not one for one.
- State changes reach the gateway within 0.2 s.
- Like the Python proxy, the gateway calls the engine on the path alone
  and drops the query string; changing that means changing both.

## Phases

| Phase | Plan item | Content | Drift guard |
|-------|-----------|---------|-------------|
| R1 (done) | 4.7, part of 1.8 | Core crate, PyO3 module, `SCHEDULER_CORE` | Golden vectors, differential tests, sim A/B, suites on Rust |
| R2 (done) | 4.7 | Predictor inference in Rust: load the safetensors + JSON model files (1.7) and run the MLPs; preprocessing stays in Python | Recorded predictions within 1e-5 of torch on every model kind; differential tests; routing sim with trained predictors, P95 within 5% |
| R3a (done) | 5.0, 1.8 | `swarmpilot-gateway`: proxy, routing with its own RNG, predictor and experiment mode, dispatch, SSE; gRPC control channel. Go/no-go (0.1): the Python proxy added 2.4 ms P99 at concurrency 1 and 153 ms at 64 | Conformance on both data planes, control channel end to end, routing A/B identical on deterministic strategies, overhead benchmark |
| R3b (done) | 5.0 | API style translation in the gateway: canonical model, three styles, streams, `plan_request` in `swarmpilot_core::api` | Differential fuzzing against `swarmpilot/api`; translation matrix and byte A/B on both data planes |
| R3c | 5.8 | Shadow mode and the emulated-cluster A/B on the 256-instance spec | Decision and latency distributions within bars |
| R4 | 8.6 | Published `swarmpilot-core` wheels, static gateway binary, signatures and SBOM | Wheel and binary run the same CI suites |

`SCHEDULER_CORE` defaults to `python` until R2 lands and the Gate A
checks pass on the Rust core.

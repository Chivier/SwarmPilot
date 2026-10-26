# Cluster simulator

The simulator lets SwarmPilot be tested and benchmarked without GPUs. It is
built in phases (design doc: "SwarmPilot cluster simulator design"):

| Phase | What | Status |
|---|---|---|
| S0 | Engine performance models, catalogs, cluster specs, workloads, metrics | This change |
| S1 | Virtual engines over HTTP and PyLet sim mode (real PyLet head, simulated workers) | Next |
| S2 | Multi-instance discrete-event sim driving the real routing and autoscaling code | Planned |
| S3 | Scenario suite, CI gates, `calibrate` | Planned |

The core is Rust (`rust/crates/sim-core`, binary `swarmpilot-sim` in
`rust/crates/sim-cli`). `swarmpilot/sim/` is the Python face: it holds the
data files the binary reads and wraps the binary.

## Build and run

```bash
cargo build --release --manifest-path rust/Cargo.toml

# The default cluster: 9 hardware configurations, 288 GPUs, 256 instances.
rust/target/release/swarmpilot-sim plan
rust/target/release/swarmpilot-sim plan --instances

# One configuration's resolved profile.
rust/target/release/swarmpilot-sim profile \
  --engine vllm --model llama-3.1-8b --gpu H100-SXM-80GB

# Replay a request stream through one simulated engine: ShareGPT-like
# lengths at 4 QPS, fixed lengths, or an Azure LLM inference trace.
rust/target/release/swarmpilot-sim bench \
  --engine vllm --model llama-3.1-8b --gpu H100-SXM-80GB --qps 4
rust/target/release/swarmpilot-sim bench ... --prompt 32 --output 128
rust/target/release/swarmpilot-sim bench ... --trace AzureLLMInferenceTrace_conv.csv
```

From Python:

```python
from swarmpilot import sim

sim.plan()                    # default spec summary
sim.profile("vllm", "meta-llama/Llama-3.1-8B-Instruct", "H100-SXM-80GB")
sim.bench("diffusers", "sdxl", "L4-24GB", qps=0.5, slo_ms=15_000)
```

The binary finds the catalogs in `swarmpilot/sim/data/` through
`--catalog`, so editing them needs no rebuild. Set `SWARMPILOT_SIM_BIN` to
use a binary from elsewhere.

One engine replays about 100,000 requests (1.9 simulated hours at 15 QPS
on an 8B model) in a quarter of a second.

## Engine models

All engines implement one event-driven interface (`EngineModel`: submit,
next event time, advance), so the same model runs on a virtual clock (S2)
or the wall clock (S1).

- **LLM (vLLM, SGLang):** iteration-level continuous batching. Each step
  runs one decode token per running sequence, then fills the token budget
  (`max_num_batched_tokens`) with prefill chunks, oldest first, up to
  `max_num_seqs`. KV-cache blocks are taken as tokens are computed; when a
  decode needs a block and none is free, the newest sequence is preempted
  and recomputed. A step costs
  `overhead + max(weights + kv_read, compute)`, from datasheet bandwidth
  and FLOPS with the efficiency factors in `engines.yaml`. A per-request
  API server delay is added before the scheduler sees a request.
- **Diffusion:** batches of up to `max_batch`, the longest step count in
  the batch, cost scaled by pixel count and a batch factor, then VAE decode.
- **Generic (private libraries):** `concurrency` parallel slots, FIFO,
  lognormal service times.

## Where the numbers come from

Every profile records its `source` and an `uncalibrated` flag.

- GPUs: vendor datasheets (`gpus.yaml`).
- Models: published `config.json` shapes (`models.yaml`).
- vLLM factors: fit to the vLLM nightly performance benchmark on 8xH100
  (buildkite `vllm/performance-benchmark` build 9565), Llama-3.1-8B TP1 and
  Llama-3-70B TP4. Only those H100 profiles count as calibrated; other GPUs
  are extrapolated by bandwidth (decode) and FLOPS (prefill).
- SGLang: no usable public per-step numbers, so it reuses vLLM's factors
  (the vLLM v0.6.0 blog reports comparable TTFT and TPOT). Uncalibrated.
- SDXL: about 2.45 s for 30 steps at 1024x1024 on H100, derived from
  Baseten's TensorRT post (0.68 images/s with TensorRT, about 40% lower
  latency than the standard pipeline). Other GPUs scale by a damped FLOPS
  ratio. Uncalibrated.
- Generic: synthetic.

`cargo test --release --test calibration -- --nocapture` prints the
comparison. At the time of writing:

| Point (H100) | Measured | Simulated | Error |
|---|---|---|---|
| 8B latency test (32 in, 128 out, batch 8) | 997.5 ms | 991.7 ms | -0.6% |
| 8B ShareGPT QPS 1: TTFT / ITL | 24.7 / 7.63 ms | 25.3 / 7.66 ms | +2.5% / +0.5% |
| 8B QPS 16: TTFT / ITL | 38.2 / 10.3 ms | 28.4 / 8.4 ms | -26% / -19% |
| 8B QPS inf: throughput | 19.5 req/s | 16.7 req/s | -14% |
| 70B TP4 latency test | 2444 ms | 2530 ms | +3.5% |
| 70B QPS 1: TTFT / ITL | 62.4 / 19.4 ms | 64.2 / 19.7 ms | +2.9% / +1.5% |
| 70B QPS 16: TTFT / ITL | 130 / 28.3 ms | 87 / 23.2 ms | -33% / -18% |
| 70B QPS inf: throughput | 8.9 req/s | 6.7 req/s | -25% |

So: at low load the model is within 5%. Under heavy load it is optimistic
about latency (15 to 33% low) and pessimistic about saturated throughput
(15 to 25% low). Saturated throughput also swings with the length sample,
since a run lasts as long as its longest output (12 to 19 req/s across
seeds for 8B). Replace these factors with our own vLLM and SGLang
measurements when they exist.

## Cluster specs

```yaml
name: my-cluster
seed: 0
node_groups:
  - name: h100-llama8b
    gpu: H100-SXM-80GB
    gpus_per_node: 8
    nodes: 4
    deployments:
      - {engine: vllm, model: llama-3.1-8b, tp: 1, instances: 32}
```

`plan` places each instance's `tp` GPUs on one node, first fit, and fails
when a group runs out of GPUs, a name is unknown, or a model does not fit
in memory. Models can be named by catalog name or Hugging Face id.

## Workloads

`WorkloadSpec` in `sim-core` draws seeded request streams: burst, Poisson,
piecewise steps, two-state MMPP bursts, diurnal curves; fixed, uniform or
lognormal lengths; ShareGPT-like LLM lengths; image steps and sizes.
`load_azure_trace` replays the Azure LLM inference trace CSV format.

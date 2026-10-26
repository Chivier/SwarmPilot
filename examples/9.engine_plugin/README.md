# Bring your own engine

SwarmPilot launches, routes and predicts for engines through adapters. vLLM
and SGLang ship built in. This example adds a private inference server that
SwarmPilot has never heard of, in two ways, without touching `swarmpilot/`:

| Way | When to use it | Files |
|-----|----------------|-------|
| **YAML description** for the generic HTTP adapter | The server speaks plain HTTP and JSON, and its runtime depends on things you can point at in the request body. | [`toy-ranker-http.yaml`](toy-ranker-http.yaml) |
| **Plugin package** with an adapter class | You need code: a custom version probe, custom features, a non-JSON protocol, live metrics. | [`toy_ranker_plugin/`](toy_ranker_plugin/) |

The server itself is [`toy_ranker/server.py`](toy_ranker_plugin/toy_ranker/server.py),
a toy document ranker whose runtime grows with the number of documents and
the query length.

## Option 1: YAML, no code

```bash
cd examples/9.engine_plugin
export SWARMPILOT_ENGINE_CONFIGS=$PWD/toy-ranker-http.yaml   # files or dirs, ':'-separated
export PYTHONPATH=$PWD/toy_ranker_plugin                      # so the server module imports

# Start the server the way the Planner would, and a Scheduler in front of it
python -m toy_ranker.server --model toy-small --port 9100 &
SCHEDULER_MODEL_ID=toy-small sscheduler start --port 8000 &

curl -X POST localhost:8000/v1/instance/register -H 'content-type: application/json' -d '{
  "instance_id": "toy0", "model_id": "toy-small", "endpoint": "http://localhost:9100",
  "platform_info": {"software_name": "toy-ranker-http", "software_version": "1.4", "hardware_name": "cpu"}}'

curl -X POST localhost:8000/v1/rank -H 'content-type: application/json' \
  -d '{"query": "red fox", "documents": ["a red fox", "a blue whale"]}'
```

With the Planner, set `PYLET_BACKEND=toy-ranker-http` (and the same
`SWARMPILOT_ENGINE_CONFIGS`) and `splanner serve toy-small` launches it on the
cluster, reads its version from `/info` and scales it like any other engine.

The YAML fields:

- `launch.argv`: the command, as a list. `{model}`, `{gpus}` and `{port}` are
  filled in; the model id always stays one argument, so it cannot inject
  shell syntax. `gpu_args` is appended when an instance has more than one GPU.
- `version`: where a live instance reports its version (`path` and a dotted
  `field`). Predictor models are keyed on the major.minor version, so an
  upgrade starts a fresh model.
- `routes`: the public paths. `retry_safe: true` lets the Scheduler resend a
  request after a lost connection.
- `features`: predictor inputs, each a small expression over the request
  body: `field(x)`, `len(x)`, `count(a[*].b)`, `chars(a[*].b)`,
  `product(...)`, `sum(...)`, `min(...)`, `max(...)` and constants. Missing
  values take the feature's `default` (0). Expressions are parsed, never
  evaluated as code.
- `capacity.max_in_flight`: requests sent to one instance at once.

## Option 2: a plugin package

[`toy_ranker/adapter.py`](toy_ranker_plugin/toy_ranker/adapter.py) subclasses
`swarmpilot.engines.EngineAdapter`, and
[`pyproject.toml`](toy_ranker_plugin/pyproject.toml) registers it:

```toml
[project.entry-points."swarmpilot.engines"]
toy-ranker = "toy_ranker.adapter:ToyRankerAdapter"
```

```bash
pip install -e examples/9.engine_plugin/toy_ranker_plugin
python -c "from swarmpilot.engines import adapter_names; print(adapter_names())"
# [..., 'sglang', 'toy-ranker', 'vllm']
```

Check any adapter against the contract every engine must meet:

```python
from swarmpilot.engines.contract import check_adapter
from toy_ranker.adapter import ToyRankerAdapter
assert check_adapter(ToyRankerAdapter()) == []
```

`tests/engines/test_generic_http.py` runs both options end to end: it
launches the server from the YAML launch spec, registers it with a
Scheduler, routes a request through the proxy and checks the features the
scheduling strategy received.

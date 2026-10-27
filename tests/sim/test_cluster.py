"""End-to-end tests of PyLet sim mode: a real PyLet head, simulated workers.

These start real processes (the PyLet head, ``swarmpilot-sim up``, and for
the Planner test ``splanner`` and ``sscheduler``) and skip when the
simulator binary or the ``pylet`` package is missing.
"""

import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from swarmpilot import sim
from swarmpilot.engines.registry import get_adapter
from swarmpilot.planner.pylet.sim_cluster import SimPyLetCluster

pylet = pytest.importorskip("pylet")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sim.find_binary() is None, reason="swarmpilot-sim not built"
    ),
]

TIME_SCALE = 0.02


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _port_block(size: int) -> int:
    """A random start for ``size`` ports that are free right now.

    Stays below Linux's ephemeral range (32768+), where outgoing
    connections would take ports out from under the engines.
    """
    for _ in range(50):
        start = random.randrange(10000, 32768 - size)
        try:
            for p in range(start, start + size, max(1, size // 64)):
                with socket.socket() as s:
                    s.bind(("127.0.0.1", p))
        except OSError:
            continue
        return start
    raise RuntimeError("no free port block")


def _cluster(spec: Path | None, nodes: int, gap: int) -> SimPyLetCluster:
    return SimPyLetCluster(
        port=_free_port(),
        spec=spec,
        time_scale=TIME_SCALE,
        port_start=_port_block(nodes * gap),
        port_gap=gap,
        control_port=_free_port(),
        ready_timeout=120,
    )


def _wait(pred, timeout: float, what):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = pred()
        if last:
            return last
        time.sleep(0.5)
    what = what() if callable(what) else what
    raise AssertionError(f"timed out waiting for {what} (last: {last})")


@pytest.fixture(scope="module")
def cluster_256():
    cluster = _cluster(None, nodes=102, gap=20)
    assert cluster.start(), "simulated cluster did not start"
    pylet.init(cluster.head_url)
    yield cluster
    cluster.stop()


def test_default_spec_registers_every_node(cluster_256):
    workers = httpx.get(f"{cluster_256.head_url}/workers").json()
    online = [w for w in workers if w["status"] == "ONLINE"]
    assert len(online) == cluster_256.nodes == 102
    gpus = sum(w["total_resources"]["gpu_units"] for w in online)
    assert gpus == 288


def test_256_instances_on_9_configs_come_up(cluster_256):
    plan = sim.plan(instances=True)
    assert len(plan) == 256
    models = yaml.safe_load((sim.DATA_DIR / "models.yaml").read_text())
    start = time.monotonic()
    ids = []
    for p in plan:
        prof = p["profile"]
        engine, tp = prof["engine"], prof["tp"]
        hf_id = (models[prof["model"]].get("aliases") or [prof["model"]])[0]
        env = {}
        if engine == "generic":
            cmd = 'python -m private_engine --port "$PORT"'
            env = {"SWARMPILOT_SIM_ENGINE": "generic"}
        else:
            cmd = get_adapter(engine).launch_spec(hf_id, gpus=tp).render_shell()
        inst = pylet.submit(
            cmd, name=p["id"], gpu=tp, target_worker=p["node"], env=env
        )
        ids.append(inst.id)

    wanted = set(ids)
    seen: dict[str, int] = {}

    def all_running():
        insts = [i for i in pylet.instances() if i.id in wanted]
        seen.clear()
        for i in insts:
            seen[i.status] = seen.get(i.status, 0) + 1
        return insts if seen.get("RUNNING") == len(ids) else None

    insts = _wait(all_running, 120, lambda: f"256 RUNNING ({seen})")
    with httpx.Client(timeout=10) as c:

        def all_healthy():
            ok = sum(
                c.get(f"http://{i.endpoint}/health").status_code == 200
                for i in insts
            )
            return ok == len(insts)

        _wait(all_healthy, 60, "256 healthy engines")
        assert time.monotonic() - start < 120

        sim_view = {
            i["instance_id"]: i
            for i in c.get(f"{cluster_256.control_url}/instances").json()
        }
        configs = {
            (v["profile"]["gpu"], v["profile"]["engine"], v["profile"]["tp"])
            for v in sim_view.values()
        }
        assert len(configs) >= 9
        # One request per engine, shaped for its kind.
        for i in insts:
            kind = sim_view[i.id]["profile"]["kind"]["kind"]
            if kind == "llm":
                r = c.post(
                    f"http://{i.endpoint}/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 4,
                    },
                )
            elif kind == "diffusion":
                r = c.post(
                    f"http://{i.endpoint}/v1/images/generations",
                    json={"prompt": "a cat", "num_inference_steps": 2},
                )
            else:
                r = c.post(f"http://{i.endpoint}/infer", json={"x": 1})
            assert r.status_code == 200, (i.id, kind, r.text)


def test_crash_reports_failed_and_closes_port(cluster_256):
    # The default spec fills every GPU, so crash one of its engines.
    view = httpx.get(f"{cluster_256.control_url}/instances").json()
    victim = next(
        v
        for v in view
        if v["status"] == "RUNNING" and v["profile"]["engine"] == "vllm"
    )
    inst_id = victim["instance_id"]
    r = httpx.post(f"{cluster_256.control_url}/instances/{inst_id}/crash")
    assert r.status_code == 200

    def failed():
        r = httpx.get(f"{cluster_256.head_url}/instances/{inst_id}").json()
        return r if r["status"] == "FAILED" else None

    info = _wait(failed, 30, "FAILED status")
    assert info["exit_code"] == 137
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"http://127.0.0.1:{victim['port']}/health", timeout=2)


def test_faults_inject_errors(cluster_256):
    view = httpx.get(f"{cluster_256.control_url}/instances").json()
    target = next(
        v
        for v in view
        if v["status"] == "RUNNING" and v["profile"]["kind"]["kind"] == "llm"
    )
    url = f"{cluster_256.control_url}/instances/{target['instance_id']}"
    httpx.post(f"{url}/faults", json={"error_rate": 1.0, "error_status": 503})
    body = {"model": "m", "prompt": "hi", "max_tokens": 2}
    r = httpx.post(
        f"http://127.0.0.1:{target['port']}/v1/completions", json=body
    )
    assert r.status_code == 503
    httpx.post(f"{url}/faults", json={"error_rate": 0.0})
    r = httpx.post(
        f"http://127.0.0.1:{target['port']}/v1/completions", json=body
    )
    assert r.status_code == 200


# -- Planner + Scheduler on a small simulated cluster ----------------------

SMALL_SPEC = {
    "name": "small",
    "node_groups": [
        {
            "name": "h100",
            "gpu": "H100-SXM-80GB",
            "gpus_per_node": 8,
            "nodes": 1,
        },
        {"name": "l40s", "gpu": "L40S-48GB", "gpus_per_node": 4, "nodes": 1},
    ],
}
MODEL = "Qwen/Qwen3-8B"


def _service(argv, env, log: Path, health: str, timeout: float = 180):
    with open(log, "wb") as out:
        proc = subprocess.Popen(
            argv,
            env={**os.environ, **env},
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def up():
        if proc.poll() is not None:
            raise AssertionError(log.read_text()[-3000:])
        try:
            return httpx.get(health, timeout=2).status_code == 200
        except httpx.HTTPError:
            return False

    _wait(up, timeout, argv[1])
    return proc


def _bin(name: str) -> str:
    return str(Path(sys.executable).parent / name)


@pytest.fixture(scope="module")
def planner_stack(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("planner-sim")
    spec = tmp / "small.yaml"
    spec.write_text(yaml.safe_dump(SMALL_SPEC))
    planner_port, sched_port = _free_port(), _free_port()
    control = _free_port()
    planner_url = f"http://localhost:{planner_port}"
    procs = []
    try:
        procs.append(
            _service(
                [_bin("splanner"), "start", "--port", str(planner_port)],
                {
                    "PYLET_SIM_MODE": "true",
                    "PYLET_SIM_SPEC": str(spec),
                    "PYLET_SIM_TIME_SCALE": str(TIME_SCALE),
                    "PYLET_SIM_PORT_START": str(_port_block(200)),
                    "PYLET_SIM_CONTROL_PORT": str(control),
                    "PYLET_LOCAL_PORT": str(_free_port()),
                    "PLANNER_PORT": str(planner_port),
                },
                tmp / "planner.log",
                f"{planner_url}/v1/health",
            )
        )
        procs.append(
            _service(
                [_bin("sscheduler"), "start", "--port", str(sched_port)],
                {
                    "SCHEDULER_MODEL_ID": MODEL,
                    "SCHEDULER_PORT": str(sched_port),
                    "PLANNER_REGISTRATION_URL": planner_url,
                },
                tmp / "scheduler.log",
                f"http://localhost:{sched_port}/v1/health",
            )
        )
        yield {
            "planner": planner_url,
            "scheduler": f"http://localhost:{sched_port}",
            "control": f"http://127.0.0.1:{control}",
        }
    finally:
        for p in reversed(procs):
            p.terminate()
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()


def _chat(c: httpx.Client, scheduler: str) -> httpx.Response:
    return c.post(
        f"{scheduler}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        },
    )


def test_planner_serve_scale_terminate(planner_stack):
    planner, scheduler = planner_stack["planner"], planner_stack["scheduler"]
    with httpx.Client(timeout=120) as c:
        r = c.post(
            f"{planner}/v1/serve",
            json={"model_or_command": MODEL, "replicas": 4, "gpu_count": 1},
        ).json()
        assert r["success"], r
        assert len(r["instances"]) == 4

        listed = c.get(f"{scheduler}/v1/instance/list").json()
        assert listed["count"] == 4
        for _ in range(8):
            r = _chat(c, scheduler)
            assert r.status_code == 200, r.text
            assert r.json()["usage"]["completion_tokens"] == 16

        r = c.post(
            f"{planner}/v1/scale", json={"model": MODEL, "replicas": 6}
        ).json()
        assert r["success"] and r["current_count"] == 6, r
        sim_view = c.get(f"{planner_stack['control']}/instances").json()
        running = [v for v in sim_view if v["status"] == "RUNNING"]
        assert len(running) == 6
        assert {v["profile"]["model"] for v in running} == {"qwen3-8b"}

        r = c.post(f"{planner}/v1/terminate", json={"model": MODEL}).json()
        assert r["success"], r
        _wait(
            lambda: c.get(f"{scheduler}/v1/instance/list").json()["count"] == 0,
            30,
            "scheduler to drop instances",
        )


def test_instances_carry_their_node_gpu(planner_stack):
    """Each instance is registered with the GPU of the node it runs on."""
    planner, scheduler = planner_stack["planner"], planner_stack["scheduler"]
    with httpx.Client(timeout=120) as c:
        r = c.post(
            f"{planner}/v1/serve",
            json={"model_or_command": MODEL, "replicas": 10, "gpu_count": 1},
        ).json()
        assert r["success"], r
        sim_view = {
            v["instance_id"]: v
            for v in c.get(f"{planner_stack['control']}/instances").json()
        }
        listed = c.get(f"{scheduler}/v1/instance/list").json()["instances"]
        assert len(listed) == 10
        hardware = {}
        for inst in listed:
            gpu = sim_view[inst["instance_id"]]["profile"]["gpu"]
            assert inst["platform_info"]["hardware_name"] == gpu
            hardware[gpu] = hardware.get(gpu, 0) + 1
        # 8 H100 GPUs and 4 L40S GPUs: 10 replicas need both kinds.
        assert set(hardware) == {"H100-SXM-80GB", "L40S-48GB"}

        r = c.post(f"{planner}/v1/terminate", json={"model": MODEL}).json()
        assert r["success"], r
        _wait(
            lambda: c.get(f"{scheduler}/v1/instance/list").json()["count"] == 0,
            30,
            "scheduler to drop instances",
        )


def test_serve_on_one_kind_of_hardware(planner_stack):
    planner, scheduler = planner_stack["planner"], planner_stack["scheduler"]
    with httpx.Client(timeout=120) as c:
        r = c.post(
            f"{planner}/v1/serve",
            json={
                "model_or_command": MODEL,
                "replicas": 3,
                "gpu_count": 1,
                "hardware": "L40S*",
            },
        ).json()
        assert r["success"], r
        listed = c.get(f"{scheduler}/v1/instance/list").json()["instances"]
        assert len(listed) == 3
        assert {i["platform_info"]["hardware_name"] for i in listed} == {
            "L40S-48GB"
        }
        # Only 4 L40S GPUs exist: asking for 6 places 4 and reports the rest.
        r = c.post(
            f"{planner}/v1/scale", json={"model": MODEL, "replicas": 6}
        ).json()
        assert not r["success"]
        assert r["current_count"] == 4, r

        r = c.post(f"{planner}/v1/terminate", json={"model": MODEL}).json()
        assert r["success"], r
        _wait(
            lambda: c.get(f"{scheduler}/v1/instance/list").json()["count"] == 0,
            30,
            "scheduler to drop instances",
        )


def test_crashed_engine_stops_getting_traffic(planner_stack):
    """A crash costs no client errors, and the Planner replaces the engine."""
    planner, scheduler = planner_stack["planner"], planner_stack["scheduler"]
    control = planner_stack["control"]
    with httpx.Client(timeout=120) as c:
        r = c.post(
            f"{planner}/v1/serve",
            json={"model_or_command": MODEL, "replicas": 4, "gpu_count": 1},
        ).json()
        assert r["success"], r
        victim = r["instances"][0]
        assert c.post(f"{control}/instances/{victim}/crash").status_code == 200

        # Right after the crash, before PyLet or the Planner react.
        slow = 0
        for _ in range(20):
            start = time.monotonic()
            r = _chat(c, scheduler)
            assert r.status_code == 200, r.text
            slow += time.monotonic() - start > 1.0
        assert slow == 0

        ids = lambda: {  # noqa: E731
            i["instance_id"]
            for i in c.get(f"{scheduler}/v1/instance/list").json()["instances"]
        }
        _wait(lambda: victim not in ids(), 15, "crashed engine deregistered")
        # The Planner brings the model back to 4 replicas.
        _wait(lambda: len(ids()) == 4, 30, "replacement registered")
        planner_view = c.get(f"{planner}/v1/instances").json()
        assert victim not in {i["pylet_id"] for i in planner_view}
        for _ in range(8):
            assert _chat(c, scheduler).status_code == 200

        r = c.post(f"{planner}/v1/terminate", json={"model": MODEL}).json()
        assert r["success"], r
        _wait(lambda: not ids(), 30, "scheduler to drop instances")

"""Per-worker hardware names and hardware-constrained placement."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from swarmpilot.planner.pylet.client import InstanceInfo
from swarmpilot.planner.pylet.hardware import (
    HardwareResolver,
    load_hardware_map,
    matches,
)
from swarmpilot.planner.pylet.instance_manager import (
    InstanceManager,
    ManagedInstance,
    ManagedInstanceStatus,
)


class TestLoadHardwareMap:
    """PLANNER_HARDWARE_MAP parsing."""

    def test_empty(self):
        assert load_hardware_map(None) == {}
        assert load_hardware_map("  ") == {}

    def test_inline_json_keeps_order(self):
        m = load_hardware_map('{"gpu-h100-*": "H100", "10.0.*": "A100"}')
        assert list(m.items()) == [("gpu-h100-*", "H100"), ("10.0.*", "A100")]

    def test_yaml_file(self, tmp_path):
        f = tmp_path / "hw.yaml"
        f.write_text("node-a*: L40S-48GB\nnode-b*: A10G-24GB\n")
        assert load_hardware_map(str(f)) == {
            "node-a*": "L40S-48GB",
            "node-b*": "A10G-24GB",
        }

    def test_bad_values(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_hardware_map(str(tmp_path / "missing.yaml"))
        with pytest.raises(ValueError, match="must map"):
            load_hardware_map('{"a": 1}')


class TestResolver:
    """Worker to hardware name resolution."""

    def test_default_when_unmapped(self, monkeypatch):
        monkeypatch.delenv("PLANNER_HARDWARE_MAP", raising=False)
        monkeypatch.setenv("PLANNER_HARDWARE_NAME", "A100")
        r = HardwareResolver()
        assert not r.heterogeneous
        assert r.resolve("w1", "10.0.0.1") == "A100"

    def test_env_map(self, monkeypatch):
        monkeypatch.setenv("PLANNER_HARDWARE_MAP", '{"h100-*": "H100"}')
        monkeypatch.delenv("PLANNER_HARDWARE_NAME", raising=False)
        r = HardwareResolver()
        assert r.resolve("h100-3") == "H100"
        assert r.resolve("other") == "gpu"

    def test_worker_id_before_host(self):
        r = HardwareResolver({"10.0.0.*": "A100", "fast-*": "H100"})
        assert r.resolve("fast-1", "10.0.0.5") == "H100"
        assert r.resolve("slow-1", "10.0.0.5") == "A100"

    def test_host_from_observed_workers(self):
        r = HardwareResolver({"10.1.*": "L4"})
        r.observe_workers(
            [SimpleNamespace(id="w9", host="10.1.2.3"), {"worker_id": "x"}]
        )
        assert r.resolve("w9") == "L4"

    def test_update_wins_over_older_patterns(self):
        r = HardwareResolver({"*": "gpu-any"})
        r.update({"node-1": "H200"})
        assert r.resolve("node-1") == "H200"
        assert r.resolve("node-2") == "gpu-any"

    def test_matches_ignores_case(self):
        assert matches("H100-SXM-80GB", "h100*")
        assert not matches("A100-40GB", "H100*")


def _worker(wid, host, free):
    return SimpleNamespace(
        id=wid, host=host, status="ONLINE", gpu_indices_available=list(free)
    )


def _manager(resolver):
    mgr = InstanceManager("http://pylet:1", "http://sched:1", hardware=resolver)
    mgr._initialized = True
    mgr.pylet_client = MagicMock()
    return mgr


class TestPlacement:
    """Placement on one kind of hardware and registration."""

    def test_place_gives_each_replica_a_whole_free_node(self):
        mgr = _manager(HardwareResolver({"h-*": "H100", "l-*": "L40S"}))
        mgr.pylet_client.list_workers.return_value = [
            _worker("h-0", "a", range(8)),
            _worker("h-1", "b", range(2)),
            _worker("l-0", "c", range(4)),
        ]
        # A node is never split: two H100 nodes hold two replicas, not
        # the five that 8 + 2 free GPUs would fit.
        assert mgr._place("H100*", gpu_count=2, count=4) == {
            "h-0": 1,
            "h-1": 1,
        }
        # A node smaller than one replica does not fit.
        assert mgr._place("H100*", gpu_count=4, count=4) == {"h-0": 1}

    def test_partly_used_or_assigned_nodes_are_not_free(self):
        mgr = _manager(HardwareResolver({"h-*": "H100"}))
        busy = _worker("h-0", "a", range(2))
        busy.gpu = 4  # two of its four GPUs are held
        mgr.pylet_client.list_workers.return_value = [
            busy,
            _worker("h-1", "b", range(4)),
            _worker("h-2", "c", range(4)),
        ]
        mgr._instances["p1"] = ManagedInstance(
            pylet_id="p1",
            instance_id="p1",
            model_id="m",
            backend="vllm",
            worker_id="h-2",  # submitted there, not started yet
        )
        assert [w for w, _ in mgr.free_slots()] == ["h-1"]

    def test_unpinned_deploy_takes_whole_nodes_too(self):
        mgr = _manager(HardwareResolver({}))
        mgr.pylet_client.list_workers.return_value = [
            _worker("n-0", "a", range(4)),
            _worker("n-1", "b", range(1)),
        ]
        mgr.pylet_client.deploy_model.side_effect = lambda **kw: [
            InstanceInfo(f"p-{kw['target_worker']}", kw["model_id"], None, "x")
            for _ in range(kw["count"])
        ]
        result = mgr.deploy_instances("m", count=3)
        assert result.success_count == 2
        assert "no free node" in result.failed[0][1]
        workers = {
            c.kwargs["target_worker"]
            for c in mgr.pylet_client.deploy_model.call_args_list
        }
        assert workers == {"n-0", "n-1"}

    def test_deploy_reports_replicas_that_do_not_fit(self):
        mgr = _manager(HardwareResolver({"l-*": "L40S"}))
        mgr.pylet_client.list_workers.return_value = [
            _worker("l-0", "c", range(2)),
            _worker("h-0", "a", range(8)),
        ]
        mgr.pylet_client.deploy_model.side_effect = lambda **kw: [
            InstanceInfo(f"p{i}", kw["model_id"], None, "PENDING")
            for i in range(kw["count"])
        ]
        result = mgr.deploy_instances("m", count=3, hardware="l40s")
        # One L40S node holds one replica, however many GPUs it has.
        assert result.success_count == 1
        assert len(result.failed) == 2
        assert "hardware 'l40s'" in result.failed[0][1]
        call = mgr.pylet_client.deploy_model.call_args.kwargs
        assert call["target_worker"] == "l-0"
        assert call["count"] == 1

    def test_registers_hardware_of_assigned_worker(self):
        mgr = _manager(HardwareResolver({"l-*": "L40S-48GB"}))
        mgr._custom_command = "python fake.py"  # skip the live probe
        mgr.pylet_client.wait_instance_running.return_value = InstanceInfo(
            "p1", "m", "127.0.0.1:9000", "RUNNING", worker_id="l-3"
        )
        scheduler = MagicMock()
        scheduler.register_instance.return_value = MagicMock(success=True)
        mgr._get_scheduler_client = MagicMock(return_value=scheduler)
        mgr._instances["p1"] = ManagedInstance(
            pylet_id="p1", instance_id="p1", model_id="m", backend="vllm"
        )
        managed = mgr.wait_instance_ready("p1")
        assert managed.hardware == "L40S-48GB"
        assert managed.worker_id == "l-3"
        info = scheduler.register_instance.call_args.kwargs["platform_info"]
        assert info["hardware_name"] == "L40S-48GB"


class TestReapFailed:
    """Removal of instances whose engine died."""

    def _setup(self):
        mgr = _manager(HardwareResolver({}))
        scheduler = MagicMock()
        mgr._get_scheduler_client = MagicMock(return_value=scheduler)
        for pid, status in [
            ("dead", ManagedInstanceStatus.ACTIVE),
            ("gone", ManagedInstanceStatus.ACTIVE),
            ("ok", ManagedInstanceStatus.ACTIVE),
            ("starting", ManagedInstanceStatus.WAITING_HEALTH),
        ]:
            mgr._instances[pid] = ManagedInstance(
                pylet_id=pid, instance_id=pid, model_id="m", status=status
            )
        mgr.pylet_client.list_model_instances.return_value = [
            InstanceInfo("dead", "m", "h:1", "FAILED"),
            InstanceInfo("ok", "m", "h:2", "RUNNING"),
            InstanceInfo("starting", "m", None, "FAILED"),
        ]
        return mgr, scheduler

    def test_removes_failed_and_missing_active_instances(self):
        mgr, scheduler = self._setup()
        reaped = mgr.reap_failed()
        assert {m.pylet_id for m in reaped} == {"dead", "gone"}
        assert all(m.status == ManagedInstanceStatus.FAILED for m in reaped)
        assert set(mgr.instances) == {"ok", "starting"}
        removed = {c.args[0] for c in scheduler.remove_instance.call_args_list}
        assert removed == {"dead", "gone"}
        # Only the one PyLet still knows is deleted there.
        mgr.pylet_client.cancel_instance.assert_called_once_with(
            "dead", delete=True
        )

    def test_nothing_active_skips_pylet(self):
        mgr = _manager(HardwareResolver({}))
        assert mgr.reap_failed() == []
        mgr.pylet_client.list_model_instances.assert_not_called()


def test_reap_survives_pylet_delete_error():
    """A failed PyLet delete still drops the instance and goes on."""
    mgr = _manager(HardwareResolver({}))
    mgr._get_scheduler_client = MagicMock(return_value=MagicMock())
    for pid in ("a", "b"):
        mgr._instances[pid] = ManagedInstance(
            pylet_id=pid,
            instance_id=pid,
            model_id="m",
            status=ManagedInstanceStatus.ACTIVE,
        )
    mgr.pylet_client.list_model_instances.return_value = [
        InstanceInfo("a", "m", "h:1", "FAILED"),
        InstanceInfo("b", "m", "h:2", "FAILED"),
    ]
    mgr.pylet_client.cancel_instance.side_effect = ConnectionError("reset")
    assert {m.pylet_id for m in mgr.reap_failed()} == {"a", "b"}
    assert mgr.instances == {}

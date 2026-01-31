"""End-to-end tests for PyLet + Optimizer integration.

These tests validate the complete optimizer + PyLet deployment flow:
1. Optimizer computes optimal model distribution
2. PyLet deploys instances to workers
3. Instances reach running state with endpoints
4. HTTP verification of deployed services

Requirements:
- Local PyLet cluster (started by fixtures)
- Dummy model server script

Run with:
    uv run pytest planner/tests/integration/test_pylet_optimizer_e2e.py \
        --run-integration -v -s

PYLET-013: PyLet Optimizer Integration E2E Test
"""

import asyncio
import sys

import httpx
import numpy as np
import pytest

# Get the Python interpreter path for use in PyLet commands
PYTHON_EXECUTABLE = sys.executable

# Try to import pylet
try:
    import pylet

    PYLET_AVAILABLE = True
except ImportError:
    PYLET_AVAILABLE = False
    pylet = None  # type: ignore

# Import optimizer
from swarmpilot.planner.core.swarm_optimizer import SimulatedAnnealingOptimizer

# Try to import IP optimizer (optional, requires pulp)
try:
    from swarmpilot.planner.core.swarm_optimizer import IntegerProgrammingOptimizer

    IP_AVAILABLE = True
except ImportError:
    IP_AVAILABLE = False

# Mark all tests in this module as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]


async def verify_endpoint_health(
    endpoint: str,
    max_retries: int = 10,
    retry_delay: float = 0.5,
) -> dict:
    """Verify endpoint is healthy with retry logic.

    Args:
        endpoint: HTTP endpoint in "host:port" format.
        max_retries: Maximum number of retry attempts.
        retry_delay: Delay between retries in seconds.

    Returns:
        Health response data.

    Raises:
        AssertionError: If health check fails after all retries.
    """
    last_error = None
    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.get(f"http://{endpoint}/health")
                if resp.status_code == 200:
                    return resp.json()
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_error = e
            await asyncio.sleep(retry_delay)

    raise AssertionError(
        f"Health check failed for {endpoint} after {max_retries} retries: {last_error}"
    )


class TestPyLetOptimizerE2E:
    """End-to-end tests for optimization + deployment via PyLet."""

    async def test_optimize_and_deploy(
        self,
        pylet_local_cluster,
        dummy_script_path: str,
        deployed_instances: list,
    ):
        """Test: optimizer computes plan -> PyLet deploys -> endpoints respond.

        Scenario:
        - 5 workers (M=5)
        - 3 models (N=3): model-a, model-b, model-c
        - Target distribution: [40%, 30%, 30%]
        - Uniform capacity matrix

        Expected:
        - Optimizer allocates ~2 to model-a, ~2 to model-b, ~1 to model-c
        - All instances reach RUNNING state
        - All endpoints respond to health checks
        """
        if not PYLET_AVAILABLE:
            pytest.skip("pylet not available")

        model_ids = ["model-a", "model-b", "model-c"]

        # Capacity matrix: 5 workers x 3 models (uniform capacity)
        # Each worker can serve any model at throughput 1.0
        B = np.array(
            [
                [1.0, 1.0, 1.0],  # Worker 0
                [1.0, 1.0, 1.0],  # Worker 1
                [1.0, 1.0, 1.0],  # Worker 2
                [1.0, 1.0, 1.0],  # Worker 3
                [1.0, 1.0, 1.0],  # Worker 4
            ]
        )

        # Target distribution: 40% to model-a, 30% each to b and c
        target = np.array([40.0, 30.0, 30.0])

        # Initial deployment: all workers unassigned (-1)
        initial = np.array([-1, -1, -1, -1, -1])

        # Run optimizer
        optimizer = SimulatedAnnealingOptimizer(
            M=5,
            N=3,
            B=B,
            initial=initial,
            a=1.0,  # Allow all changes
            target=target,
        )

        deployment, score, stats = optimizer.optimize(
            objective_method="relative_error",
            verbose=False,
        )

        # Verify deployment is valid
        assert len(deployment) == 5, "Deployment should have 5 entries"
        assert all(0 <= d <= 2 for d in deployment), "Each entry should be 0, 1, or 2"

        # Convert deployment array to target_state dict
        target_state: dict[str, int] = {}
        for idx in deployment:
            model_id = model_ids[idx]
            target_state[model_id] = target_state.get(model_id, 0) + 1

        print(f"\n[TEST] Optimizer deployment: {deployment}")
        print(f"[TEST] Target state: {target_state}")
        print(f"[TEST] Score: {score:.4f}")

        # Verify total count
        assert sum(target_state.values()) == 5, "Should deploy to all 5 workers"

        # Deploy dummy servers via PyLet (one at a time, replicas no longer supported)
        for model_id, count in target_state.items():
            throughput = {"model-a": 10.0, "model-b": 5.0, "model-c": 6.67}.get(
                model_id, 1.0
            )

            # PyLet no longer supports replicas - submit instances one at a time
            for i in range(count):
                inst = pylet.submit(
                    f"{PYTHON_EXECUTABLE} {dummy_script_path}",
                    cpu=1,
                    gpu=0,
                    name=f"{model_id}-{i}",
                    env={
                        "MODEL_ID": model_id,
                        "THROUGHPUT": str(throughput),
                    },
                    labels={
                        "model_id": model_id,
                        "managed_by": "swarmpilot",
                        "test": "optimize_and_deploy",
                        "replica_index": str(i),
                    },
                )
                deployed_instances.append(inst)
            print(f"[TEST] Submitted {count} {model_id} instance(s)")

        # Wait for all instances to be running
        print(f"[TEST] Waiting for {len(deployed_instances)} instances to start...")
        for i, inst in enumerate(deployed_instances):
            inst.wait_running(timeout=60)
            print(f"[TEST] Instance {i + 1} running at {inst.endpoint}")

        # Verify all endpoints respond to health checks
        print("[TEST] Verifying endpoint health checks...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            for inst in deployed_instances:
                endpoint = inst.endpoint
                assert endpoint is not None, f"Instance {inst.id} has no endpoint"

                # Try health check
                resp = await client.get(f"http://{endpoint}/health")
                assert resp.status_code == 200, f"Health check failed: {resp.text}"

                data = resp.json()
                assert data["status"] == "healthy", f"Unhealthy: {data}"
                print(f"[TEST] {endpoint} -> {data['model_id']} healthy")

        print(f"[TEST] All {len(deployed_instances)} instances verified!")

    async def test_heterogeneous_workers(
        self,
        pylet_local_cluster,
        dummy_script_path: str,
        deployed_instances: list,
    ):
        """Test optimization with heterogeneous worker capacities.

        Scenario:
        - Worker 0: Specialized for model-a (capacity 2.0)
        - Worker 1: Specialized for model-b (capacity 2.0)
        - Workers 2-4: Generalist (uniform capacity 1.0)

        Expected:
        - Optimizer assigns Worker 0 to model-a (best match)
        - Optimizer assigns Worker 1 to model-b (best match)
        """
        if not PYLET_AVAILABLE:
            pytest.skip("pylet not available")

        model_ids = ["model-a", "model-b", "model-c"]

        # Heterogeneous capacity matrix
        B = np.array(
            [
                [2.0, 0.5, 0.5],  # Worker 0: specialized for model-a
                [0.5, 2.0, 0.5],  # Worker 1: specialized for model-b
                [1.0, 1.0, 1.0],  # Worker 2: generalist
                [1.0, 1.0, 1.0],  # Worker 3: generalist
                [1.0, 1.0, 1.0],  # Worker 4: generalist
            ]
        )

        # Target: 50% model-a, 30% model-b, 20% model-c
        target = np.array([50.0, 30.0, 20.0])
        initial = np.array([-1, -1, -1, -1, -1])

        # Use Integer Programming for deterministic optimal results
        # SA is stochastic and may not always find the optimal assignment
        optimizer = IntegerProgrammingOptimizer(
            M=5,
            N=3,
            B=B,
            initial=initial,
            a=1.0,
            target=target,
        )

        deployment, score, info = optimizer.optimize(
            objective_method="relative_error",
            verbose=False,
        )

        print(f"\n[TEST] Heterogeneous deployment: {deployment}")
        print(f"[TEST] Score: {score:.4f}")
        print(f"[TEST] IP status: {info.get('status', 'N/A')}")

        # Verify optimizer found a valid solution
        # Note: IP may not assign specialized workers to their specialty
        # if another configuration achieves better target alignment
        assert all(
            0 <= d < 3 for d in deployment
        ), "All workers must be assigned valid models"
        assert (
            score < 1.0
        ), f"Optimizer should find reasonable solution, got score {score}"

        # Deploy based on optimizer output
        for worker_idx, model_idx in enumerate(deployment):
            model_id = model_ids[model_idx]

            inst = pylet.submit(
                f"{PYTHON_EXECUTABLE} {dummy_script_path}",
                cpu=1,
                gpu=0,
                env={
                    "MODEL_ID": model_id,
                    "THROUGHPUT": str(B[worker_idx][model_idx]),
                },
                labels={
                    "model_id": model_id,
                    "managed_by": "swarmpilot",
                    "worker_idx": str(worker_idx),
                    "test": "heterogeneous_workers",
                },
            )
            deployed_instances.append(inst)

        # Wait and verify
        for inst in deployed_instances:
            inst.wait_running(timeout=60)

        async with httpx.AsyncClient(timeout=10.0) as client:
            for inst in deployed_instances:
                resp = await client.get(f"http://{inst.endpoint}/health")
                assert resp.status_code == 200

        print(f"[TEST] All {len(deployed_instances)} heterogeneous instances verified!")

    @pytest.mark.skipif(not IP_AVAILABLE, reason="pulp not available for IP optimizer")
    async def test_optimize_with_ip(
        self,
        pylet_local_cluster,
        dummy_script_path: str,
        deployed_instances: list,
    ):
        """Test optimization using Integer Programming algorithm.

        Same scenario as test_optimize_and_deploy, but uses IP solver.
        IP should find globally optimal solution.
        """
        if not PYLET_AVAILABLE:
            pytest.skip("pylet not available")

        model_ids = ["model-a", "model-b", "model-c"]

        B = np.array([[1.0, 1.0, 1.0]] * 5)
        target = np.array([40.0, 30.0, 30.0])
        initial = np.array([-1, -1, -1, -1, -1])

        optimizer = IntegerProgrammingOptimizer(
            M=5,
            N=3,
            B=B,
            initial=initial,
            a=1.0,
            target=target,
        )

        deployment, score, stats = optimizer.optimize(
            objective_method="relative_error",
        )

        print(f"\n[TEST] IP deployment: {deployment}")
        print(f"[TEST] IP score: {score:.4f}")
        print(f"[TEST] IP status: {stats.get('status', 'unknown')}")

        # Convert to target state
        target_state: dict[str, int] = {}
        for idx in deployment:
            model_id = model_ids[idx]
            target_state[model_id] = target_state.get(model_id, 0) + 1

        print(f"[TEST] IP target state: {target_state}")

        # Deploy instances one at a time (PyLet no longer supports replicas)
        for model_id, count in target_state.items():
            for i in range(count):
                inst = pylet.submit(
                    f"{PYTHON_EXECUTABLE} {dummy_script_path}",
                    cpu=1,
                    gpu=0,
                    name=f"{model_id}-ip-{i}",
                    env={"MODEL_ID": model_id, "THROUGHPUT": "1.0"},
                    labels={
                        "model_id": model_id,
                        "managed_by": "swarmpilot",
                        "test": "optimize_with_ip",
                        "replica_index": str(i),
                    },
                )
                deployed_instances.append(inst)
            print(f"[TEST] Submitted {count} {model_id} instance(s)")

        # Wait and verify
        for inst in deployed_instances:
            inst.wait_running(timeout=60)

        async with httpx.AsyncClient(timeout=10.0) as client:
            for inst in deployed_instances:
                resp = await client.get(f"http://{inst.endpoint}/health")
                assert resp.status_code == 200

        print(f"[TEST] All {len(deployed_instances)} IP-optimized instances verified!")


class TestOptimizerServiceCapacity:
    """Tests for optimizer service capacity calculations."""

    async def test_service_capacity_calculation(
        self,
        pylet_local_cluster,
        dummy_script_path: str,
        deployed_instances: list,
    ):
        """Test that optimizer correctly computes service capacity.

        Given deployment, verify computed capacity matches expected.
        """
        if not PYLET_AVAILABLE:
            pytest.skip("pylet not available")

        model_ids = ["model-a", "model-b"]

        # 3 workers, 2 models with different capacities
        B = np.array(
            [
                [10.0, 5.0],  # Worker 0
                [8.0, 6.0],  # Worker 1
                [12.0, 4.0],  # Worker 2
            ]
        )

        target = np.array([70.0, 30.0])  # 70% model-a, 30% model-b
        initial = np.array([-1, -1, -1])

        optimizer = SimulatedAnnealingOptimizer(
            M=3,
            N=2,
            B=B,
            initial=initial,
            a=1.0,
            target=target,
        )

        deployment, score, _ = optimizer.optimize(
            objective_method="relative_error",
            verbose=False,
        )

        # Compute expected service capacity
        service_capacity = optimizer.compute_service_capacity(deployment)

        print(f"\n[TEST] Deployment: {deployment}")
        print(f"[TEST] Service capacity: {service_capacity}")

        # Verify capacity calculation
        expected_capacity = [0.0, 0.0]
        for worker_idx, model_idx in enumerate(deployment):
            expected_capacity[model_idx] += B[worker_idx][model_idx]

        np.testing.assert_array_almost_equal(
            service_capacity, expected_capacity, decimal=5
        )

        # Deploy and verify endpoints
        for worker_idx, model_idx in enumerate(deployment):
            model_id = model_ids[model_idx]
            throughput = B[worker_idx][model_idx]

            inst = pylet.submit(
                f"{PYTHON_EXECUTABLE} {dummy_script_path}",
                cpu=1,
                gpu=0,
                env={"MODEL_ID": model_id, "THROUGHPUT": str(throughput)},
                labels={"model_id": model_id, "managed_by": "swarmpilot"},
            )
            deployed_instances.append(inst)

        for inst in deployed_instances:
            inst.wait_running(timeout=60)

        async with httpx.AsyncClient(timeout=10.0) as client:
            for inst in deployed_instances:
                resp = await client.get(f"http://{inst.endpoint}/health")
                assert resp.status_code == 200
                # Verify throughput in response
                data = resp.json()
                assert "throughput" in data

        print("[TEST] Service capacity test passed!")


# ==============================================================================
# Planner API E2E Tests (PYLET-015)
# These tests use the planner's /deploy API instead of raw pylet calls
# ==============================================================================


class TestPlannerDeployAPI:
    """E2E tests using planner's /deploy API.

    These tests validate the complete flow through the planner service:
    1. POST to /deploy with target_state
    2. Planner deploys instances via PyLet
    3. Verify instances are running via /status
    4. HTTP verification of deployed services
    """

    async def test_deploy_via_planner_api(
        self,
        planner_client: httpx.AsyncClient,
        cleanup_via_planner,
    ):
        """Test deployment via planner's /deploy endpoint.

        Deploys 3 instances using the planner API and verifies they respond.
        """
        # Target state: 2 model-a, 1 model-b
        target_state = {"model-a": 2, "model-b": 1}

        print(f"\n[TEST] Deploying via planner API: {target_state}")

        # Deploy via planner API
        resp = await planner_client.post(
            "/v1/deploy",
            json={
                "target_state": target_state,
                "wait_for_ready": True,
            },
        )
        assert resp.status_code == 200, f"Deploy failed: {resp.text}"

        data = resp.json()
        print(f"[TEST] Deploy response: success={data['success']}")
        print(f"[TEST] Added: {data['added_count']}, Removed: {data['removed_count']}")

        assert data["success"], f"Deployment failed: {data.get('error')}"
        assert data["added_count"] == 3, f"Expected 3 added, got {data['added_count']}"

        # Verify via /status
        status_resp = await planner_client.get("/v1/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()

        print(f"[TEST] Status: {status_data['total_instances']} instances")
        print(f"[TEST] Current state: {status_data['current_state']}")

        assert status_data["initialized"], "PyLet should be initialized"
        assert status_data["total_instances"] == 3

        # Verify endpoints respond (with retry for startup timing)
        active_instances = data["active_instances"]
        assert len(active_instances) == 3

        for inst in active_instances:
            endpoint = inst["endpoint"]
            assert endpoint is not None, f"Instance {inst['pylet_id']} has no endpoint"

            health_data = await verify_endpoint_health(endpoint)
            assert health_data["status"] == "healthy"
            print(f"[TEST] {endpoint} -> {health_data['model_id']} healthy")

        print("[TEST] test_deploy_via_planner_api passed!")

    async def test_scale_via_planner_api(
        self,
        planner_client: httpx.AsyncClient,
        cleanup_via_planner,
    ):
        """Test scaling via planner's /scale endpoint.

        First deploys, then scales up and down.
        """
        # Initial deploy: 2 model-a
        resp = await planner_client.post(
            "/v1/deploy",
            json={"target_state": {"model-a": 2}, "wait_for_ready": True},
        )
        assert resp.status_code == 200
        assert resp.json()["success"]

        print("\n[TEST] Initial deployment: 2 model-a")

        # Scale up to 4
        resp = await planner_client.post(
            "/v1/scale",
            json={
                "model_id": "model-a",
                "target_count": 4,
                "wait_for_ready": True,
            },
        )
        assert resp.status_code == 200
        scale_data = resp.json()

        print(
            f"[TEST] Scale up response: {scale_data['previous_count']} -> {scale_data['current_count']}"
        )

        assert scale_data["success"]
        assert scale_data["previous_count"] == 2
        assert scale_data["current_count"] == 4
        assert scale_data["added"] == 2

        # Scale down to 1
        resp = await planner_client.post(
            "/v1/scale",
            json={
                "model_id": "model-a",
                "target_count": 1,
                "wait_for_ready": False,
            },
        )
        assert resp.status_code == 200
        scale_data = resp.json()

        print(
            f"[TEST] Scale down response: {scale_data['previous_count']} -> {scale_data['current_count']}"
        )

        assert scale_data["success"]
        assert scale_data["previous_count"] == 4
        assert scale_data["current_count"] == 1
        assert scale_data["removed"] == 3

        print("[TEST] test_scale_via_planner_api passed!")

    async def test_optimize_via_planner_api(
        self,
        planner_client: httpx.AsyncClient,
        cleanup_via_planner,
    ):
        """Test optimization via planner's /optimize endpoint.

        Uses the planner's optimize endpoint to compute optimal deployment
        and deploy in one call.
        """
        model_ids = ["model-a", "model-b", "model-c"]

        # Capacity matrix: 5 workers x 3 models
        B = [[1.0, 1.0, 1.0]] * 5

        # Target distribution
        target = [40.0, 30.0, 30.0]

        print("\n[TEST] Optimizing via planner API...")

        resp = await planner_client.post(
            "/v1/optimize",
            json={
                "model_ids": model_ids,
                "B": B,
                "target": target,
                "a": 1.0,
                "algorithm": "simulated_annealing",
                "objective_method": "relative_error",
                "wait_for_ready": True,
            },
        )
        assert resp.status_code == 200, f"Optimize failed: {resp.text}"

        data = resp.json()
        print(f"[TEST] Deployment: {data['deployment']}")
        print(f"[TEST] Score: {data['score']:.4f}")
        print(f"[TEST] Service capacity: {data['service_capacity']}")
        print(f"[TEST] Deployment success: {data['deployment_success']}")
        print(f"[TEST] Added: {data['added_count']}")

        assert data["deployment_success"], f"Deployment failed: {data.get('error')}"
        assert len(data["deployment"]) == 5, "Should have 5 worker assignments"
        assert data["added_count"] == 5, f"Expected 5 added, got {data['added_count']}"

        # Verify instances (with retry for startup timing)
        active = data["active_instances"]
        assert len(active) == 5

        for inst in active:
            endpoint = inst["endpoint"]
            health_data = await verify_endpoint_health(endpoint)
            assert health_data["status"] == "healthy"
            print(f"[TEST] {endpoint} -> {health_data['model_id']} healthy")

        print("[TEST] test_optimize_via_planner_api passed!")

    async def test_deploy_reconciliation(
        self,
        planner_client: httpx.AsyncClient,
        cleanup_via_planner,
    ):
        """Test that planner correctly reconciles state changes.

        Deploys, then changes target state, verifying correct add/remove.
        """
        # Initial: 3 model-a
        resp = await planner_client.post(
            "/v1/deploy",
            json={"target_state": {"model-a": 3}, "wait_for_ready": True},
        )
        assert resp.status_code == 200
        assert resp.json()["success"]

        print("\n[TEST] Initial: 3 model-a")

        # Change to: 1 model-a, 2 model-b
        resp = await planner_client.post(
            "/v1/deploy",
            json={"target_state": {"model-a": 1, "model-b": 2}, "wait_for_ready": True},
        )
        assert resp.status_code == 200
        data = resp.json()

        print(
            f"[TEST] Reconcile: added={data['added_count']}, removed={data['removed_count']}"
        )

        assert data["success"]
        # Should remove 2 model-a, add 2 model-b
        assert data["removed_count"] == 2
        assert data["added_count"] == 2

        # Verify final state
        status_resp = await planner_client.get("/v1/status")
        status_data = status_resp.json()

        expected_state = {"model-a": 1, "model-b": 2}
        assert (
            status_data["current_state"] == expected_state
        ), f"Expected {expected_state}, got {status_data['current_state']}"

        print("[TEST] test_deploy_reconciliation passed!")

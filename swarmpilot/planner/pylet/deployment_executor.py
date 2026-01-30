"""Deployment strategy execution (PYLET-009).

This module provides the DeploymentExecutor class that translates optimizer
output into PyLet operations. It computes the diff between current and target
state, then executes add/remove operations via InstanceManager.

Example:
    from swarmpilot.planner.pylet.deployment_executor import DeploymentExecutor

    executor = DeploymentExecutor(instance_manager)

    # Execute optimizer output
    result = executor.execute(
        current_state={model_id: count for model_id, count in current.items()},
        target_state={model_id: count for model_id, count in target.items()},
    )
    print(f"Added: {result.added_count}, Removed: {result.removed_count}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from swarmpilot.planner.pylet.instance_manager import (
    InstanceManager,
    ManagedInstance,
    ManagedInstanceStatus,
)


@dataclass
class DeploymentAction:
    """A single deployment action to execute.

    Attributes:
        action_type: Either "add" or "remove".
        model_id: Model to add or remove.
        count: Number of instances to add/remove.
        target_worker: Target worker for add actions.
        pylet_ids: PyLet IDs of instances to remove (for remove actions).
    """

    action_type: str  # "add" or "remove"
    model_id: str
    count: int = 1
    target_worker: str | None = None
    pylet_ids: list[str] = field(default_factory=list)


@dataclass
class DeploymentPlan:
    """Plan of deployment actions to execute.

    Attributes:
        actions: List of actions to execute.
        total_adds: Total instances to add.
        total_removes: Total instances to remove.
    """

    actions: list[DeploymentAction]
    total_adds: int = 0
    total_removes: int = 0

    @classmethod
    def from_diff(
        cls,
        current_state: dict[str, int],
        target_state: dict[str, int],
        instances_by_model: dict[str, list[ManagedInstance]] | None = None,
    ) -> DeploymentPlan:
        """Create deployment plan from state diff.

        Args:
            current_state: Current model counts {model_id: count}.
            target_state: Target model counts {model_id: count}.
            instances_by_model: Optional map of model_id to instances for removals.

        Returns:
            DeploymentPlan with add/remove actions.
        """
        actions = []
        total_adds = 0
        total_removes = 0

        # Get all model IDs
        all_models = set(current_state.keys()) | set(target_state.keys())

        for model_id in all_models:
            current = current_state.get(model_id, 0)
            target = target_state.get(model_id, 0)
            diff = target - current

            if diff > 0:
                # Need to add instances
                actions.append(
                    DeploymentAction(
                        action_type="add",
                        model_id=model_id,
                        count=diff,
                    )
                )
                total_adds += diff

            elif diff < 0:
                # Need to remove instances
                remove_count = abs(diff)

                # Get PyLet IDs to remove if available
                pylet_ids = []
                if instances_by_model and model_id in instances_by_model:
                    # Select instances to remove (FIFO - oldest first)
                    candidates = sorted(
                        instances_by_model[model_id],
                        key=lambda i: i.created_at,
                    )
                    pylet_ids = [c.pylet_id for c in candidates[:remove_count]]

                actions.append(
                    DeploymentAction(
                        action_type="remove",
                        model_id=model_id,
                        count=remove_count,
                        pylet_ids=pylet_ids,
                    )
                )
                total_removes += remove_count

        return cls(
            actions=actions,
            total_adds=total_adds,
            total_removes=total_removes,
        )


@dataclass
class ExecutionResult:
    """Result of executing a deployment plan.

    Attributes:
        success: Whether all actions succeeded.
        added_instances: Instances successfully added.
        removed_instances: Instances successfully removed.
        failed_adds: List of (model_id, error) for failed adds.
        failed_removes: List of (pylet_id, error) for failed removes.
    """

    success: bool
    added_instances: list[ManagedInstance]
    removed_instances: list[str]  # pylet_ids
    failed_adds: list[tuple[str, str]]
    failed_removes: list[tuple[str, str]]

    @property
    def added_count(self) -> int:
        """Number of successfully added instances."""
        return len(self.added_instances)

    @property
    def removed_count(self) -> int:
        """Number of successfully removed instances."""
        return len(self.removed_instances)

    @property
    def failed_count(self) -> int:
        """Total number of failures."""
        return len(self.failed_adds) + len(self.failed_removes)


class DeploymentExecutor:
    """Executes deployment plans via InstanceManager.

    This class translates high-level deployment decisions (add N instances
    of model X, remove M instances of model Y) into PyLet operations.

    Attributes:
        instance_manager: InstanceManager for deployment operations.
        default_backend: Default model backend.
        default_gpu_count: Default GPU count per instance.
    """

    def __init__(
        self,
        instance_manager: InstanceManager,
        default_backend: str = "vllm",
        default_gpu_count: int = 1,
    ):
        """Initialize deployment executor.

        Args:
            instance_manager: InstanceManager for operations.
            default_backend: Default model backend.
            default_gpu_count: Default GPUs per instance.
        """
        self.instance_manager = instance_manager
        self.default_backend = default_backend
        self.default_gpu_count = default_gpu_count

    def plan(
        self,
        current_state: dict[str, int],
        target_state: dict[str, int],
    ) -> DeploymentPlan:
        """Create a deployment plan from state diff.

        Args:
            current_state: Current model counts {model_id: count}.
            target_state: Target model counts {model_id: count}.

        Returns:
            DeploymentPlan with actions to execute.
        """
        # Get current instances by model for removal selection
        instances_by_model: dict[str, list[ManagedInstance]] = {}
        for instance in self.instance_manager.instances.values():
            if instance.status == ManagedInstanceStatus.ACTIVE:
                if instance.model_id not in instances_by_model:
                    instances_by_model[instance.model_id] = []
                instances_by_model[instance.model_id].append(instance)

        return DeploymentPlan.from_diff(current_state, target_state, instances_by_model)

    def execute_plan(
        self,
        plan: DeploymentPlan,
        wait_for_ready: bool = True,
        ready_timeout: float = 300.0,
    ) -> ExecutionResult:
        """Execute a deployment plan.

        Deploys instances via PyLet's submit API.

        Args:
            plan: DeploymentPlan to execute.
            wait_for_ready: Whether to wait for added instances to be ready.
            ready_timeout: Timeout for instance readiness.

        Returns:
            ExecutionResult with success/failure details.
        """
        import time

        start_time = time.time()

        added_instances: list[ManagedInstance] = []
        removed_instances: list[str] = []
        failed_adds: list[tuple[str, str]] = []
        failed_removes: list[tuple[str, str]] = []

        # Log deployment start
        logger.info(
            f"[DEPLOY_START] plan_adds={plan.total_adds} plan_removes={plan.total_removes} "
            f"actions={len(plan.actions)}"
        )

        # Execute remove actions first (free up resources)
        for action in plan.actions:
            if action.action_type == "remove":
                # Log remove action
                logger.info(
                    f"[DEPLOY_ACTION] action=remove model_id={action.model_id} "
                    f"count={action.count} pylet_ids={action.pylet_ids}"
                )
                for pylet_id in action.pylet_ids:
                    try:
                        success = self.instance_manager.terminate_instance(pylet_id)
                        if success:
                            removed_instances.append(pylet_id)
                        else:
                            failed_removes.append((pylet_id, "Termination failed"))
                    except Exception as e:
                        failed_removes.append((pylet_id, str(e)))
                        logger.error(f"Failed to remove {pylet_id}: {e}")

        # Execute add actions
        pending_instances: list[ManagedInstance] = []
        for action in plan.actions:
            if action.action_type == "add" and action.count > 0:
                # Log add action
                logger.info(
                    f"[DEPLOY_ACTION] action=add model_id={action.model_id} "
                    f"count={action.count}"
                )
                # Deploy instances via PyLet
                result = self.instance_manager.deploy_instances(
                    model_id=action.model_id,
                    count=action.count,
                    backend=self.default_backend,
                    gpu_count=self.default_gpu_count,
                    target_worker=action.target_worker,
                )
                pending_instances.extend(result.deployed)
                failed_adds.extend(result.failed)

        # Wait for pending instances to be ready
        if wait_for_ready and pending_instances:
            ready_instances = self.instance_manager.wait_instances_ready(
                [i.pylet_id for i in pending_instances],
                timeout=ready_timeout,
            )
            for instance in ready_instances:
                if instance.status == ManagedInstanceStatus.ACTIVE:
                    added_instances.append(instance)
                else:
                    failed_adds.append((instance.model_id, instance.error or "Failed"))
        else:
            added_instances.extend(pending_instances)

        success = len(failed_adds) == 0 and len(failed_removes) == 0
        elapsed_time = time.time() - start_time

        # Log deployment end
        logger.info(
            f"[DEPLOY_END] success={success} "
            f"added={len(added_instances)} removed={len(removed_instances)} "
            f"failed={len(failed_adds) + len(failed_removes)} "
            f"elapsed_time={elapsed_time:.2f}s"
        )

        return ExecutionResult(
            success=success,
            added_instances=added_instances,
            removed_instances=removed_instances,
            failed_adds=failed_adds,
            failed_removes=failed_removes,
        )

    def execute(
        self,
        current_state: dict[str, int],
        target_state: dict[str, int],
        wait_for_ready: bool = True,
        ready_timeout: float = 300.0,
    ) -> ExecutionResult:
        """Plan and execute deployment changes.

        Convenience method that combines plan() and execute_plan().

        Args:
            current_state: Current model counts.
            target_state: Target model counts.
            wait_for_ready: Whether to wait for added instances.
            ready_timeout: Timeout for instance readiness.

        Returns:
            ExecutionResult with success/failure details.
        """
        plan = self.plan(current_state, target_state)
        return self.execute_plan(
            plan,
            wait_for_ready=wait_for_ready,
            ready_timeout=ready_timeout,
        )

    def reconcile(
        self,
        target_state: dict[str, int],
        wait_for_ready: bool = True,
    ) -> ExecutionResult:
        """Reconcile current state to target state.

        Computes current state from managed instances and executes diff.

        Args:
            target_state: Target model counts.
            wait_for_ready: Whether to wait for added instances.

        Returns:
            ExecutionResult with success/failure details.
        """
        # Compute current state from active instances
        current_state: dict[str, int] = {}
        for instance in self.instance_manager.get_active_instances():
            model_id = instance.model_id
            current_state[model_id] = current_state.get(model_id, 0) + 1

        logger.info(f"Reconciling: current={current_state}, target={target_state}")

        return self.execute(current_state, target_state, wait_for_ready=wait_for_ready)

    def scale_model(
        self,
        model_id: str,
        target_count: int,
        wait_for_ready: bool = True,
    ) -> ExecutionResult:
        """Scale a specific model to target count.

        Args:
            model_id: Model to scale.
            target_count: Target number of instances.
            wait_for_ready: Whether to wait for added instances.

        Returns:
            ExecutionResult with success/failure details.
        """
        # Get current count for this model
        current_count = len(self.instance_manager.get_instances_by_model(model_id))

        logger.info(f"Scaling {model_id}: {current_count} -> {target_count}")

        return self.execute(
            {model_id: current_count},
            {model_id: target_count},
            wait_for_ready=wait_for_ready,
        )

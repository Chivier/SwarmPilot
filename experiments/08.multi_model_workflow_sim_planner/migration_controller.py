"""
Instance Migration Controller for Dynamic Scheduler Rebalancing.

This module handles migrating instances between schedulers while maintaining
continuous task submission. The migration process uses instance's /model/restart API:
1. Call instance's /model/restart with new scheduler URL
2. Instance handles the full lifecycle internally:
   - Draining from current scheduler
   - Waiting for pending tasks to complete
   - Stopping current model
   - Deregistering from old scheduler
   - Starting new model
   - Registering to new scheduler
3. Monitor progress via /model/restart/status polling at 500ms intervals
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests
from loguru import logger


class MigrationStatus(str, Enum):
    """Status of an instance migration."""
    PENDING = "pending"           # Migration planned but not started
    DRAINING = "draining"         # Draining in progress
    WAITING_DELAY = "waiting"     # Waiting 50ms before re-registration
    REGISTERING = "registering"   # Re-registering to new scheduler
    COMPLETED = "completed"       # Successfully migrated
    FAILED = "failed"             # Migration failed


@dataclass
class InstanceMigration:
    """Tracks migration state for a single instance using /model/restart API."""
    instance_id: str
    source_scheduler_url: str
    target_scheduler_url: str
    model_id: str
    endpoint: str
    platform_info: Dict[str, str]

    status: MigrationStatus = MigrationStatus.PENDING
    operation_id: Optional[str] = None  # Restart operation ID from instance
    restart_initiated_at: Optional[float] = None
    restart_completed_at: Optional[float] = None
    error_message: Optional[str] = None

    pending_tasks_at_start: int = 0
    pending_tasks_completed: int = 0

    def mark_restart_initiated(self, operation_id: str) -> None:
        """Mark restart operation as initiated."""
        self.status = MigrationStatus.DRAINING
        self.operation_id = operation_id
        self.restart_initiated_at = time.time()

    def update_status_from_restart_api(self, api_status: str, pending_completed: int = 0) -> None:
        """
        Update migration status based on restart API status.

        Status mapping:
        - pending/draining/waiting_tasks → DRAINING
        - stopping_model/deregistering → WAITING_DELAY
        - starting_model/registering → REGISTERING
        - completed → COMPLETED
        - failed → FAILED
        """
        self.pending_tasks_completed = pending_completed

        if api_status in ["pending", "draining", "waiting_tasks"]:
            self.status = MigrationStatus.DRAINING
        elif api_status in ["stopping_model", "deregistering"]:
            self.status = MigrationStatus.WAITING_DELAY
        elif api_status in ["starting_model", "registering"]:
            self.status = MigrationStatus.REGISTERING
        elif api_status == "completed":
            self.status = MigrationStatus.COMPLETED
            self.restart_completed_at = time.time()
        elif api_status == "failed":
            self.status = MigrationStatus.FAILED

    def mark_failed(self, error: str) -> None:
        """Mark migration as failed."""
        self.status = MigrationStatus.FAILED
        self.error_message = error
        self.restart_completed_at = time.time()

    def get_total_duration_ms(self) -> Optional[float]:
        """Get total migration duration in milliseconds."""
        if self.restart_initiated_at and self.restart_completed_at:
            return (self.restart_completed_at - self.restart_initiated_at) * 1000
        return None


@dataclass
class PhaseConfig:
    """Configuration for a single experimental phase."""
    phase_id: int
    fanout: int  # Number of B tasks per A task (n)
    scheduler_a_instances: int  # Instances on Scheduler A
    scheduler_b_instances: int  # Instances on Scheduler B
    num_workflows: int  # Workflows to submit in this phase


class MigrationController:
    """
    Controls instance migrations between schedulers during phase transitions.

    Uses instance's /model/restart API to handle the complete migration lifecycle:
    - Initiating restart operation with new scheduler URL
    - Polling restart status at 500ms intervals
    - Tracking migration progress and statistics
    - Instance handles drain, task completion, stop, deregister, start, register internally
    """

    def __init__(
        self,
        scheduler_a_url: str,
        scheduler_b_url: str,
        instance_port_range: Tuple[int, int],
        model_id: str = "sleep_model",
        restart_poll_interval_ms: float = 500,
        max_restart_wait_ms: float = 60000,
    ):
        """
        Initialize migration controller.

        Args:
            scheduler_a_url: URL of Scheduler A
            scheduler_b_url: URL of Scheduler B
            instance_port_range: (start_port, end_port) for instances
            model_id: Model ID for instances
            restart_poll_interval_ms: How often to poll restart status (ms), default 500ms
            max_restart_wait_ms: Maximum time to wait for restart completion (ms)
        """
        self.scheduler_a_url = scheduler_a_url
        self.scheduler_b_url = scheduler_b_url
        self.instance_port_range = instance_port_range
        self.model_id = model_id
        self.restart_poll_interval_ms = restart_poll_interval_ms
        self.max_restart_wait_ms = max_restart_wait_ms

        # Track migrations
        self.migrations: Dict[str, InstanceMigration] = {}
        self.migration_history: List[InstanceMigration] = []

        # Platform info for instances
        self.platform_info = {
            "software_name": "Linux",
            "software_version": "5.15.0",
            "hardware_name": "x86_64"
        }

    def plan_migration(
        self,
        current_phase: PhaseConfig,
        next_phase: PhaseConfig,
        current_instances_a: List[str],
        current_instances_b: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Plan which instances to migrate between schedulers.

        Args:
            current_phase: Current phase configuration
            next_phase: Next phase configuration
            current_instances_a: Instance IDs currently on Scheduler A
            current_instances_b: Instance IDs currently on Scheduler B

        Returns:
            (instances_to_move_a_to_b, instances_to_move_b_to_a)
        """
        current_a_count = len(current_instances_a)
        current_b_count = len(current_instances_b)
        target_a_count = next_phase.scheduler_a_instances
        target_b_count = next_phase.scheduler_b_instances

        instances_a_to_b = []
        instances_b_to_a = []

        # Calculate how many to move from A to B
        if current_a_count > target_a_count:
            # Move excess from A to B
            num_to_move = current_a_count - target_a_count
            instances_a_to_b = current_instances_a[:num_to_move]

        # Calculate how many to move from B to A
        if current_b_count > target_b_count:
            # Move excess from B to A
            num_to_move = current_b_count - target_b_count
            instances_b_to_a = current_instances_b[:num_to_move]

        logger.info(
            f"Migration plan: Phase {current_phase.phase_id} → Phase {next_phase.phase_id}: "
            f"A→B: {len(instances_a_to_b)} instances, B→A: {len(instances_b_to_a)} instances"
        )

        return instances_a_to_b, instances_b_to_a

    def initiate_migration(
        self,
        instance_ids: List[str],
        source_scheduler_url: str,
        target_scheduler_url: str,
    ) -> None:
        """
        Initiate model restart for a list of instances using /model/restart API.

        Args:
            instance_ids: List of instance IDs to migrate
            source_scheduler_url: Source scheduler URL (for metadata only)
            target_scheduler_url: Target scheduler URL to register to
        """
        for instance_id in instance_ids:
            # Create migration record
            instance_endpoint = f"http://localhost:{self._get_port_from_id(instance_id)}"
            migration = InstanceMigration(
                instance_id=instance_id,
                source_scheduler_url=source_scheduler_url,
                target_scheduler_url=target_scheduler_url,
                model_id=self.model_id,
                endpoint=instance_endpoint,
                platform_info=self.platform_info,
            )
            self.migrations[instance_id] = migration

            # Initiate restart via instance API
            try:
                response = requests.post(
                    f"{instance_endpoint}/model/restart",
                    json={
                        "model_id": self.model_id,
                        "parameters": {},
                        "scheduler_url": target_scheduler_url,
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                result = response.json()

                if result.get("success"):
                    operation_id = result.get("operation_id")
                    migration.mark_restart_initiated(operation_id)
                    logger.info(
                        f"Initiated restart for {instance_id} (operation_id: {operation_id})"
                    )
                else:
                    error_msg = result.get("error", "Unknown error")
                    migration.mark_failed(error_msg)
                    logger.error(f"Failed to restart {instance_id}: {error_msg}")

            except Exception as e:
                error_msg = f"Failed to initiate restart: {e}"
                migration.mark_failed(error_msg)
                logger.error(f"Failed to initiate restart for {instance_id}: {e}")

    async def monitor_migrations_async(self) -> Dict[str, any]:
        """
        Monitor all active migrations asynchronously using /model/restart API.

        Polls restart status for each instance and tracks migration progress.
        This method runs in the background and returns statistics when all
        migrations are complete.

        Returns:
            Statistics about the migration process
        """
        start_time = time.time()
        total_migrations = len(self.migrations)
        completed_count = 0
        failed_count = 0

        logger.info(f"Starting migration monitor for {total_migrations} instances")

        # Use configured poll interval
        poll_interval_sec = self.restart_poll_interval_ms / 1000.0

        while self.migrations:
            # Check each active migration
            for instance_id in list(self.migrations.keys()):
                migration = self.migrations[instance_id]

                # Skip if already in terminal state
                if migration.status in [MigrationStatus.COMPLETED, MigrationStatus.FAILED]:
                    # Move to history and remove from active
                    self.migration_history.append(migration)
                    del self.migrations[instance_id]

                    if migration.status == MigrationStatus.COMPLETED:
                        completed_count += 1
                        logger.info(
                            f"Migration completed for {instance_id} "
                            f"(total: {migration.get_total_duration_ms():.1f}ms, "
                            f"tasks_completed: {migration.pending_tasks_completed})"
                        )
                    else:
                        failed_count += 1
                        logger.error(f"Migration failed for {instance_id}: {migration.error_message}")
                    continue

                # Poll restart status for active migrations
                if migration.operation_id:
                    await self._poll_restart_status_async(migration)

            # Wait before next poll
            await asyncio.sleep(poll_interval_sec)

            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.max_restart_wait_ms:
                logger.error(f"Migration timeout after {elapsed_ms:.1f}ms")
                for migration in list(self.migrations.values()):
                    migration.mark_failed("Timeout waiting for restart completion")
                    self.migration_history.append(migration)
                    failed_count += 1
                self.migrations.clear()
                break

        total_duration_ms = (time.time() - start_time) * 1000

        stats = {
            "total_migrations": total_migrations,
            "completed": completed_count,
            "failed": failed_count,
            "total_duration_ms": total_duration_ms,
        }

        logger.info(
            f"Migration batch completed: {completed_count}/{total_migrations} successful, "
            f"{failed_count} failed, {total_duration_ms:.1f}ms total"
        )

        return stats

    async def _poll_restart_status_async(self, migration: InstanceMigration) -> None:
        """
        Poll restart status and update migration state.

        Queries /model/restart/status endpoint and updates migration status
        based on the restart API's current state.
        """
        try:
            response = requests.get(
                f"{migration.endpoint}/model/restart/status",
                params={"operation_id": migration.operation_id},
                timeout=5.0
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("success"):
                # Operation not found or error
                error_msg = result.get("error", "Unknown error querying restart status")
                migration.mark_failed(error_msg)
                logger.error(f"Restart status error for {migration.instance_id}: {error_msg}")
                return

            # Extract status information
            api_status = result.get("status")
            pending_completed = result.get("pending_tasks_completed", 0)
            error = result.get("error")

            # Update migration status based on API status
            old_status = migration.status
            migration.update_status_from_restart_api(api_status, pending_completed)

            # Log status changes
            if migration.status != old_status:
                logger.info(
                    f"Instance {migration.instance_id} restart status: {old_status.value} → {migration.status.value} "
                    f"(API: {api_status}, completed_tasks: {pending_completed})"
                )

            # Handle failed status
            if api_status == "failed" and error:
                migration.mark_failed(error)
                logger.error(f"Restart failed for {migration.instance_id}: {error}")

        except Exception as e:
            error_msg = f"Failed to poll restart status: {e}"
            migration.mark_failed(error_msg)
            logger.error(f"Error polling restart status for {migration.instance_id}: {e}")

    def _get_port_from_id(self, instance_id: str) -> int:
        """Extract port number from instance ID."""
        # Expected format: instance-000, instance-001, etc.
        try:
            idx = int(instance_id.split('-')[-1])
            return self.instance_port_range[0] + idx
        except (ValueError, IndexError):
            raise ValueError(f"Invalid instance_id format: {instance_id}")

    def get_migration_stats(self) -> Dict[str, any]:
        """Get statistics about completed migrations."""
        if not self.migration_history:
            return {
                "total": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
            }

        successful = [m for m in self.migration_history if m.status == MigrationStatus.COMPLETED]
        failed = [m for m in self.migration_history if m.status == MigrationStatus.FAILED]

        drain_durations = [m.get_drain_duration_ms() for m in successful if m.get_drain_duration_ms()]
        total_durations = [m.get_total_duration_ms() for m in successful if m.get_total_duration_ms()]

        return {
            "total": len(self.migration_history),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(self.migration_history) * 100,
            "avg_drain_duration_ms": sum(drain_durations) / len(drain_durations) if drain_durations else 0,
            "avg_total_duration_ms": sum(total_durations) / len(total_durations) if total_durations else 0,
            "max_drain_duration_ms": max(drain_durations) if drain_durations else 0,
            "min_drain_duration_ms": min(drain_durations) if drain_durations else 0,
        }

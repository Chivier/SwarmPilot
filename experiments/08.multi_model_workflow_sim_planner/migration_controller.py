"""
Instance Migration Controller for Dynamic Scheduler Rebalancing.

This module handles migrating instances between schedulers while maintaining
continuous task submission. The migration process:
1. Drain instances from source scheduler (stop accepting new tasks)
2. Wait for pending tasks to complete
3. Wait 50ms delay
4. Re-register instances to target scheduler
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
    """Tracks migration state for a single instance."""
    instance_id: str
    source_scheduler_url: str
    target_scheduler_url: str
    model_id: str
    endpoint: str
    platform_info: Dict[str, str]

    status: MigrationStatus = MigrationStatus.PENDING
    drain_initiated_at: Optional[float] = None
    drain_completed_at: Optional[float] = None
    registration_completed_at: Optional[float] = None
    error_message: Optional[str] = None

    pending_tasks_at_drain: int = 0
    drain_poll_count: int = 0

    def mark_draining(self, pending_tasks: int) -> None:
        """Mark migration as draining phase."""
        self.status = MigrationStatus.DRAINING
        self.drain_initiated_at = time.time()
        self.pending_tasks_at_drain = pending_tasks

    def mark_waiting(self) -> None:
        """Mark migration as waiting for 50ms delay."""
        self.status = MigrationStatus.WAITING_DELAY
        self.drain_completed_at = time.time()

    def mark_registering(self) -> None:
        """Mark migration as re-registering."""
        self.status = MigrationStatus.REGISTERING

    def mark_completed(self) -> None:
        """Mark migration as successfully completed."""
        self.status = MigrationStatus.COMPLETED
        self.registration_completed_at = time.time()

    def mark_failed(self, error: str) -> None:
        """Mark migration as failed."""
        self.status = MigrationStatus.FAILED
        self.error_message = error

    def get_drain_duration_ms(self) -> Optional[float]:
        """Get drain duration in milliseconds."""
        if self.drain_initiated_at and self.drain_completed_at:
            return (self.drain_completed_at - self.drain_initiated_at) * 1000
        return None

    def get_total_duration_ms(self) -> Optional[float]:
        """Get total migration duration in milliseconds."""
        if self.drain_initiated_at and self.registration_completed_at:
            return (self.registration_completed_at - self.drain_initiated_at) * 1000
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

    Handles the complete migration lifecycle:
    - Initiating drain on source scheduler
    - Polling drain status until all tasks complete
    - Waiting 50ms delay
    - Re-registering to target scheduler
    """

    def __init__(
        self,
        scheduler_a_url: str,
        scheduler_b_url: str,
        instance_port_range: Tuple[int, int],
        model_id: str = "sleep_model",
        drain_poll_interval_ms: float = 100,
        max_drain_wait_ms: float = 60000,
    ):
        """
        Initialize migration controller.

        Args:
            scheduler_a_url: URL of Scheduler A
            scheduler_b_url: URL of Scheduler B
            instance_port_range: (start_port, end_port) for instances
            model_id: Model ID for instances
            drain_poll_interval_ms: How often to check drain status (ms)
            max_drain_wait_ms: Maximum time to wait for draining (ms)
        """
        self.scheduler_a_url = scheduler_a_url
        self.scheduler_b_url = scheduler_b_url
        self.instance_port_range = instance_port_range
        self.model_id = model_id
        self.drain_poll_interval_ms = drain_poll_interval_ms
        self.max_drain_wait_ms = max_drain_wait_ms

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
        Initiate draining for a list of instances.

        Args:
            instance_ids: List of instance IDs to migrate
            source_scheduler_url: Source scheduler URL
            target_scheduler_url: Target scheduler URL
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

            # Initiate drain
            try:
                response = requests.post(
                    f"{source_scheduler_url}/instance/drain",
                    json={"instance_id": instance_id},
                    timeout=5.0
                )
                response.raise_for_status()
                result = response.json()

                pending_tasks = result.get("pending_tasks", 0)
                migration.mark_draining(pending_tasks)

                logger.info(
                    f"Initiated drain for {instance_id} on {source_scheduler_url} "
                    f"({pending_tasks} pending tasks)"
                )

            except Exception as e:
                error_msg = f"Failed to initiate drain: {e}"
                migration.mark_failed(error_msg)
                logger.error(f"Failed to drain {instance_id}: {e}")

    async def monitor_migrations_async(self) -> Dict[str, any]:
        """
        Monitor all active migrations asynchronously.

        Polls drain status, waits for completion, applies 50ms delay,
        and re-registers instances.

        Returns:
            Statistics about the migration process
        """
        start_time = time.time()
        total_migrations = len(self.migrations)
        completed_count = 0
        failed_count = 0

        logger.info(f"Starting migration monitor for {total_migrations} instances")

        while self.migrations:
            # Check each active migration
            for instance_id in list(self.migrations.keys()):
                migration = self.migrations[instance_id]

                if migration.status == MigrationStatus.DRAINING:
                    # Poll drain status
                    await self._check_drain_status_async(migration)

                elif migration.status == MigrationStatus.WAITING_DELAY:
                    # Apply 50ms delay then re-register
                    await self._handle_reregistration_async(migration)

                elif migration.status == MigrationStatus.COMPLETED:
                    # Move to history and remove from active
                    self.migration_history.append(migration)
                    del self.migrations[instance_id]
                    completed_count += 1
                    logger.info(
                        f"Migration completed for {instance_id} "
                        f"(total: {migration.get_total_duration_ms():.1f}ms)"
                    )

                elif migration.status == MigrationStatus.FAILED:
                    # Move to history and remove from active
                    self.migration_history.append(migration)
                    del self.migrations[instance_id]
                    failed_count += 1
                    logger.error(f"Migration failed for {instance_id}: {migration.error_message}")

            # Wait before next poll
            await asyncio.sleep(self.drain_poll_interval_ms / 1000.0)

            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.max_drain_wait_ms:
                logger.error(f"Migration timeout after {elapsed_ms:.1f}ms")
                for migration in list(self.migrations.values()):
                    migration.mark_failed("Timeout waiting for drain completion")
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

    async def _check_drain_status_async(self, migration: InstanceMigration) -> None:
        """Check drain status and transition to waiting if complete."""
        try:
            response = requests.get(
                f"{migration.source_scheduler_url}/instance/drain/status",
                params={"instance_id": migration.instance_id},
                timeout=5.0
            )
            response.raise_for_status()
            status = response.json()

            migration.drain_poll_count += 1
            pending_tasks = status.get("pending_tasks", 0)
            can_remove = status.get("can_remove", False)

            if can_remove:
                migration.mark_waiting()
                logger.info(
                    f"Instance {migration.instance_id} drain complete "
                    f"({migration.drain_poll_count} polls, "
                    f"{migration.get_drain_duration_ms():.1f}ms)"
                )
            elif migration.drain_poll_count % 10 == 0:
                # Log progress every 10 polls
                logger.debug(
                    f"Draining {migration.instance_id}: {pending_tasks} tasks remaining "
                    f"(poll {migration.drain_poll_count})"
                )

        except Exception as e:
            error_msg = f"Failed to check drain status: {e}"
            migration.mark_failed(error_msg)
            logger.error(f"Error checking drain status for {migration.instance_id}: {e}")

    async def _handle_reregistration_async(self, migration: InstanceMigration) -> None:
        """
        Handle instance re-registration using the correct architecture.

        Instead of directly calling scheduler's /instance/register API,
        we use the instance's own APIs:
        1. Stop the model (which auto-deregisters from old scheduler)
        2. Wait 50ms delay
        3. Start the model with new scheduler_url (which auto-registers to new scheduler)
        """
        try:
            migration.mark_registering()

            # Step 1: Stop the model (auto-deregisters from old scheduler)
            logger.info(f"Stopping model on {migration.instance_id}")
            stop_response = requests.get(
                f"{migration.endpoint}/model/stop",
                timeout=5.0
            )
            stop_response.raise_for_status()

            # Step 2: Wait 50ms delay
            await asyncio.sleep(0.05)

            # Step 3: Start the model with new scheduler_url (auto-registers to new scheduler)
            logger.info(
                f"Starting model on {migration.instance_id} with new scheduler: {migration.target_scheduler_url}"
            )
            start_response = requests.post(
                f"{migration.endpoint}/model/start",
                json={
                    "model_id": migration.model_id,
                    "parameters": {},
                    "scheduler_url": migration.target_scheduler_url,
                },
                timeout=10.0
            )
            start_response.raise_for_status()

            migration.mark_completed()
            logger.info(
                f"Successfully migrated {migration.instance_id} to {migration.target_scheduler_url}"
            )

        except Exception as e:
            error_msg = f"Failed to migrate instance: {e}"
            migration.mark_failed(error_msg)
            logger.error(f"Error migrating {migration.instance_id}: {e}")

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

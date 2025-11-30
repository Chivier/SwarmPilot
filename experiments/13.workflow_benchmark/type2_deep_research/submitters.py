"""Deep Research workflow task submitters."""

import random
import threading
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from common import BaseTaskSubmitter, estimate_token_length
from .workflow_data import DeepResearchWorkflowData


# ============================================================================
# Prompt Templates
# ============================================================================

A_TEMPLATE = "Generate a comprehensive research plan for topic: {topic}"
B1_TEMPLATE = "Conduct detailed research on subtopic: {subtopic}"
B2_TEMPLATE = "Analyze and summarize research findings for: {subtopic}"
MERGE_TEMPLATE = "Synthesize all research findings into final report: {summary}"


class ATaskSubmitter(BaseTaskSubmitter):
    """
    Submits A tasks (initial query) with Poisson arrivals.

    Pre-generates all workflow data upfront, then submits A tasks
    following a Poisson process controlled by the rate limiter.
    """

    def __init__(self, config, workflow_states: Dict,
                 state_lock: threading.Lock,
                 pre_generated_workflows: Optional[List[DeepResearchWorkflowData]] = None,
                 **kwargs):
        """
        Initialize A task submitter.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            pre_generated_workflows: Optional pre-generated workflow data for reproducibility.
                                     If provided, uses deep copies with current strategy applied.
            **kwargs: Passed to BaseTaskSubmitter (name, scheduler_url, rate_limiter, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock

        # Set random seed for reproducibility (uniform sampling)
        random.seed(42)

        if pre_generated_workflows is not None:
            # Use pre-generated workflows (deep copy to avoid mutation)
            import copy
            self.workflows = []
            for w in pre_generated_workflows:
                workflow_copy = copy.deepcopy(w)
                # Update strategy for this run
                workflow_copy.strategy = getattr(config, 'strategy', 'probabilistic')
                # Reset timing fields for fresh run
                workflow_copy.a_submit_time = None
                workflow_copy.a_complete_time = None
                workflow_copy.b1_submit_times = []
                workflow_copy.b2_submit_times = []
                workflow_copy.b1_complete_times = {}
                workflow_copy.b2_complete_times = {}
                workflow_copy.merge_submit_time = None
                workflow_copy.merge_complete_time = None
                workflow_copy.workflow_complete_time = None
                workflow_copy.b1_task_ids = []
                workflow_copy.b2_task_ids = []
                workflow_copy.merge_task_id = None
                workflow_copy.a_result = None
                self.workflows.append(workflow_copy)
            self.logger.info(f"Using {len(self.workflows)} pre-generated workflows")
            fanout_values = [w.fanout_count for w in self.workflows]
        else:
            # Fallback: generate workflows on-the-fly (legacy behavior)
            # Create fanout sampler for distribution-based fanout
            fanout_sampler = config.create_fanout_sampler()

            # Pre-generate all workflow data with pre-generated sleep times for simulation
            self.workflows = []
            fanout_values = []  # Track fanout values for logging
            for i in range(config.num_workflows):
                # Sample fanout from distribution (or use static value)
                fanout_count = fanout_sampler.sample()
                fanout_values.append(fanout_count)

                workflow = DeepResearchWorkflowData(
                    workflow_id=f"workflow-{i:04d}",
                    fanout_count=fanout_count,
                    strategy=config.strategy,
                    is_warmup=(i < config.num_warmup)
                )

                # Pre-generate sleep times for simulation mode
                if config.mode == "simulation":
                    workflow.a_sleep_time = random.uniform(config.sleep_time_min, config.sleep_time_max)
                    workflow.b1_sleep_times = [
                        random.uniform(config.sleep_time_min, config.sleep_time_max)
                        for _ in range(fanout_count)
                    ]
                    workflow.b2_sleep_times = [
                        random.uniform(config.sleep_time_min, config.sleep_time_max)
                        for _ in range(fanout_count)
                    ]
                    workflow.merge_sleep_time = random.uniform(config.sleep_time_min, config.sleep_time_max)

                # Pre-generate topic for real mode
                if config.mode == "real":
                    workflow.topic = f"Deep research topic {i+1}: Advanced computing architectures"

                self.workflows.append(workflow)
            self.logger.info(f"Generated {len(self.workflows)} workflows on-the-fly (legacy mode)")

        # Populate shared workflow_states
        with state_lock:
            for workflow in self.workflows:
                workflow_states[workflow.workflow_id] = workflow

        self.index = 0

        # Calculate average sleep times for min_time strategy (simulation mode only)
        if config.mode == "simulation":
            a_times = [w.a_sleep_time for w in self.workflows if w.a_sleep_time is not None]
            b1_times = [t for w in self.workflows for t in w.b1_sleep_times]
            b2_times = [t for w in self.workflows for t in w.b2_sleep_times]
            merge_times = [w.merge_sleep_time for w in self.workflows if w.merge_sleep_time is not None]

            # Store averages in config for other submitters to use
            config.avg_a_sleep_time_ms = (sum(a_times) / len(a_times)) * 1000.0 if a_times else 10000.0
            config.avg_b1_sleep_time_ms = (sum(b1_times) / len(b1_times)) * 1000.0 if b1_times else 10000.0
            config.avg_b2_sleep_time_ms = (sum(b2_times) / len(b2_times)) * 1000.0 if b2_times else 10000.0
            config.avg_merge_sleep_time_ms = (sum(merge_times) / len(merge_times)) * 1000.0 if merge_times else 10000.0

            self.logger.info(
                f"Calculated average sleep times for min_time strategy: "
                f"A={config.avg_a_sleep_time_ms:.1f}ms, "
                f"B1={config.avg_b1_sleep_time_ms:.1f}ms, "
                f"B2={config.avg_b2_sleep_time_ms:.1f}ms, "
                f"merge={config.avg_merge_sleep_time_ms:.1f}ms"
            )

        # Log fanout distribution statistics
        if fanout_values:
            avg_fanout = sum(fanout_values) / len(fanout_values)
            min_fanout = min(fanout_values)
            max_fanout = max(fanout_values)
            fanout_config = config.get_fanout_config_info()
            self.logger.info(
                f"Using {len(self.workflows)} workflows with fanout "
                f"distribution: type={fanout_config['type']}, "
                f"avg={avg_fanout:.2f}, min={min_fanout}, max={max_fanout}"
            )

    def _prepare_task_payload(self, workflow_data: DeepResearchWorkflowData) -> Dict[str, Any]:
        """
        Prepare task submission payload for A.

        Args:
            workflow_data: Workflow data

        Returns:
            JSON payload for /task/submit
        """
        # Extract workflow number for task ID
        workflow_num = workflow_data.workflow_id.split('-')[-1]
        task_id = f"task-A-{workflow_data.strategy}-workflow-{workflow_num}"

        # Build task_input and metadata based on mode
        if self.config.mode == "simulation":
            # Use pre-generated sleep time
            task_input = {"sleep_time": workflow_data.a_sleep_time}

            # Calculate exp_runtime based on strategy
            # min_time strategy: use dataset average
            # other strategies: use actual sampled runtime
            if workflow_data.strategy == "min_time" and hasattr(self.config, 'avg_a_sleep_time_ms'):
                exp_runtime = self.config.avg_a_sleep_time_ms
            else:
                exp_runtime = workflow_data.a_sleep_time * 1000.0  # Convert to milliseconds

            # Simulation mode metadata (with scheduling parameters)
            metadata = {
                "workflow_id": workflow_data.workflow_id,
                "exp_runtime": exp_runtime,
                "exp_cv": 0.40,  # Match real LLM task CV (~39%)
                "exp_skewness": 0.0,  # LLM tasks are approximately symmetric
                "task_type": "A"
            }
        else:  # real mode
            # Use pre-generated topic
            sentence = A_TEMPLATE.format(topic=workflow_data.topic)
            task_input = {
                "sentence": sentence,
                "max_tokens": workflow_data.max_tokens
            }

            # Real mode metadata (LLM prediction features only)
            metadata = {
                "sentence": sentence,
                "token_length": estimate_token_length(sentence),
                "max_tokens": workflow_data.max_tokens
            }

        return {
            "task_id": task_id,
            "model_id": self.config.model_a_id,
            "task_input": task_input,
            "metadata": metadata
        }

    def _get_next_task_data(self) -> Optional[DeepResearchWorkflowData]:
        """
        Get next workflow to submit.

        Returns:
            Next workflow data, or None if all submitted
        """
        if self.index < len(self.workflows):
            workflow = self.workflows[self.index]
            self.index += 1
            return workflow
        return None

    def _submit_task(self, task_data: Any) -> bool:
        """
        Submit A task and record workflow start in metrics.

        Args:
            task_data: DeepResearchWorkflowData instance

        Returns:
            True if submission succeeded, False otherwise
        """
        # Record workflow start in metrics before submission
        if self.metrics and isinstance(task_data, DeepResearchWorkflowData):
            self.metrics.record_workflow_start(
                workflow_id=task_data.workflow_id,
                workflow_type="deep_research",
                metadata={"fanout_count": task_data.fanout_count, "strategy": task_data.strategy},
                is_warmup=task_data.is_warmup
            )

            # Record task submission in metrics
            workflow_num = task_data.workflow_id.split('-')[-1]
            task_id = f"task-A-{task_data.strategy}-workflow-{workflow_num}"
            self.metrics.record_task_submit(
                task_id=task_id,
                workflow_id=task_data.workflow_id,
                task_type="A"
            )

        # Log A task submission
        if isinstance(task_data, DeepResearchWorkflowData):
            workflow_num = task_data.workflow_id.split('-')[-1]
            self.logger.info(
                f"[A_SUBMIT] workflow={task_data.workflow_id}, "
                f"task_id=task-A-{task_data.strategy}-workflow-{workflow_num}, "
                f"scheduler_endpoint={self.scheduler_url}, "
                f"model_id={self.config.model_a_id}"
            )

        # Call parent implementation to actually submit the task
        return super()._submit_task(task_data)


class B1TaskSubmitter(BaseTaskSubmitter):
    """
    Submits B1 tasks (first level parallel processing).

    Receives A completion events from queue and submits fanout_count B1 tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 a_result_queue: Queue, **kwargs):
        """
        Initialize B1 task submitter.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            a_result_queue: Queue receiving (workflow_id, a_result) tuples from A receiver
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.a_result_queue = a_result_queue

    def _prepare_task_payload(self, task_data: tuple) -> Dict[str, Any]:
        """
        Prepare task submission payload for B1.

        Args:
            task_data: Tuple of (workflow_id, a_result, b1_index)

        Returns:
            JSON payload for /task/submit
        """
        workflow_id, a_result, b1_index = task_data

        # Get workflow data for pre-generated values
        with self.state_lock:
            workflow_data = self.workflow_states.get(workflow_id)
            if not workflow_data:
                raise ValueError(f"Workflow {workflow_id} not found in states")

        # Extract workflow number for task ID
        workflow_num = workflow_id.split('-')[-1]
        task_id = f"task-B1-{workflow_data.strategy}-workflow-{workflow_num}-{b1_index}"

        # Build task_input and metadata based on mode
        if self.config.mode == "simulation":
            # Use pre-generated sleep time for this B1 index
            sleep_time = workflow_data.b1_sleep_times[b1_index]
            task_input = {"sleep_time": sleep_time}

            # Calculate exp_runtime based on strategy
            # min_time strategy: use dataset average
            # other strategies: use actual sampled runtime
            if workflow_data.strategy == "min_time" and hasattr(self.config, 'avg_b1_sleep_time_ms'):
                exp_runtime = self.config.avg_b1_sleep_time_ms
            else:
                exp_runtime = sleep_time * 1000.0  # Convert to milliseconds

            # Simulation mode metadata (with scheduling parameters)
            metadata = {
                "workflow_id": workflow_id,
                "exp_runtime": exp_runtime,
                "exp_cv": 0.40,  # Match real LLM task CV (~39%)
                "exp_skewness": 0.0,  # LLM tasks are approximately symmetric
                "task_type": "B1",
                "b_index": b1_index  # B index for pairing with B2
            }
        else:  # real mode
            subtopic = f"Subtopic {b1_index}: {a_result}"
            sentence = B1_TEMPLATE.format(subtopic=subtopic)
            task_input = {
                "sentence": sentence,
                "max_tokens": workflow_data.max_tokens
            }

            # Real mode metadata (LLM prediction features only)
            metadata = {
                "sentence": sentence,
                "token_length": estimate_token_length(sentence),
                "max_tokens": workflow_data.max_tokens
            }

        return {
            "task_id": task_id,
            "model_id": self.config.model_b_id,
            "task_input": task_input,
            "metadata": metadata
        }

    def _get_next_task_data(self) -> Optional[tuple]:
        """
        Get next B1 task from queue (blocking until available).

        Returns:
            Tuple of (workflow_id, a_result, b1_index), or None if stopped
        """
        while not self.stop_event.is_set():
            try:
                return self.a_result_queue.get(timeout=0.1)
            except Empty:
                continue
        return None

    def _submit_task(self, task_data: Any) -> bool:
        """
        Submit B1 task and record in metrics.

        Args:
            task_data: Tuple of (workflow_id, a_result, b1_index)

        Returns:
            True if submission succeeded, False otherwise
        """
        # Record task submission in metrics before submission
        if self.metrics and isinstance(task_data, tuple):
            workflow_id, a_result, b1_index = task_data
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_num = workflow_id.split('-')[-1]
                    task_id = f"task-B1-{workflow_data.strategy}-workflow-{workflow_num}-{b1_index}"
                    self.metrics.record_task_submit(
                        task_id=task_id,
                        workflow_id=workflow_id,
                        task_type="B1"
                    )

        # Log B1 task submission
        if isinstance(task_data, tuple):
            workflow_id, a_result, b1_index = task_data
            self.logger.info(
                f"[B1_SUBMIT] workflow={workflow_id}, b1_index={b1_index}, "
                f"scheduler_endpoint={self.scheduler_url}"
            )

        # Call parent implementation to actually submit the task
        return super()._submit_task(task_data)


class B2TaskSubmitter(BaseTaskSubmitter):
    """
    Submits B2 tasks (second level parallel processing).

    Receives B1 completion events from queue and submits corresponding B2 tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 b1_result_queue: Queue, **kwargs):
        """
        Initialize B2 task submitter.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            b1_result_queue: Queue receiving (workflow_id, b1_result, b1_index) tuples
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.b1_result_queue = b1_result_queue

    def _prepare_task_payload(self, task_data: tuple) -> Dict[str, Any]:
        """
        Prepare task submission payload for B2.

        Args:
            task_data: Tuple of (workflow_id, b1_result, b1_index)

        Returns:
            JSON payload for /task/submit
        """
        workflow_id, b1_result, b1_index = task_data

        # Get workflow data for pre-generated values
        with self.state_lock:
            workflow_data = self.workflow_states.get(workflow_id)
            if not workflow_data:
                raise ValueError(f"Workflow {workflow_id} not found in states")

        # Extract workflow number for task ID
        workflow_num = workflow_id.split('-')[-1]
        task_id = f"task-B2-{workflow_data.strategy}-workflow-{workflow_num}-{b1_index}"

        # Build task_input and metadata based on mode
        if self.config.mode == "simulation":
            # Use pre-generated sleep time for this B2 index
            sleep_time = workflow_data.b2_sleep_times[b1_index]  # Use same index as B1
            task_input = {"sleep_time": sleep_time}

            # Calculate exp_runtime based on strategy
            # min_time strategy: use dataset average
            # other strategies: use actual sampled runtime
            if workflow_data.strategy == "min_time" and hasattr(self.config, 'avg_b2_sleep_time_ms'):
                exp_runtime = self.config.avg_b2_sleep_time_ms
            else:
                exp_runtime = sleep_time * 1000.0  # Convert to milliseconds

            # Simulation mode metadata (with scheduling parameters)
            metadata = {
                "workflow_id": workflow_id,
                "exp_runtime": exp_runtime,
                "exp_cv": 0.40,  # Match real LLM task CV (~39%)
                "exp_skewness": 0.0,  # LLM tasks are approximately symmetric
                "task_type": "B2",
                "b_index": b1_index  # B index for pairing with B1
            }
        else:  # real mode
            subtopic = f"Findings for subtopic {b1_index}: {b1_result}"
            sentence = B2_TEMPLATE.format(subtopic=subtopic)
            task_input = {
                "sentence": sentence,
                "max_tokens": workflow_data.max_tokens
            }

            # Real mode metadata (LLM prediction features only)
            metadata = {
                "sentence": sentence,
                "token_length": estimate_token_length(sentence),
                "max_tokens": workflow_data.max_tokens
            }

        return {
            "task_id": task_id,
            "model_id": self.config.model_b_id,
            "task_input": task_input,
            "metadata": metadata
        }

    def _get_next_task_data(self) -> Optional[tuple]:
        """
        Get next B2 task from queue (blocking until available).

        Returns:
            Tuple of (workflow_id, b1_result, b1_index), or None if stopped
        """
        while not self.stop_event.is_set():
            try:
                return self.b1_result_queue.get(timeout=0.1)
            except Empty:
                continue
        return None

    def _submit_task(self, task_data: Any) -> bool:
        """
        Submit B2 task and record in metrics.

        Args:
            task_data: Tuple of (workflow_id, b1_result, b1_index)

        Returns:
            True if submission succeeded, False otherwise
        """
        # Record task submission in metrics before submission
        if self.metrics and isinstance(task_data, tuple):
            workflow_id, b1_result, b1_index = task_data
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_num = workflow_id.split('-')[-1]
                    task_id = f"task-B2-{workflow_data.strategy}-workflow-{workflow_num}-{b1_index}"
                    self.metrics.record_task_submit(
                        task_id=task_id,
                        workflow_id=workflow_id,
                        task_type="B2"
                    )

        # Log B2 task submission
        if isinstance(task_data, tuple):
            workflow_id, b1_result, b1_index = task_data
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_num = workflow_id.split('-')[-1]
                    self.logger.info(
                        f"[B2_SUBMIT] workflow={workflow_id}, b2_index={b1_index}, "
                        f"task_id=task-B2-{workflow_data.strategy}-workflow-{workflow_num}-{b1_index}, "
                        f"scheduler_endpoint={self.scheduler_url}"
                    )

        # Call parent implementation to actually submit the task
        return super()._submit_task(task_data)


class MergeTaskSubmitter(BaseTaskSubmitter):
    """
    Submits Merge tasks (aggregation/finalization).

    Receives workflow_id from queue when all B2 tasks complete.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 merge_trigger_queue: Queue, **kwargs):
        """
        Initialize Merge task submitter.

        Args:
            config: DeepResearchConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            merge_trigger_queue: Queue receiving workflow_id when all B2 complete
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.merge_trigger_queue = merge_trigger_queue

    def _prepare_task_payload(self, workflow_id: str) -> Dict[str, Any]:
        """
        Prepare task submission payload for Merge.

        Args:
            workflow_id: Workflow identifier

        Returns:
            JSON payload for /task/submit
        """
        # Get workflow data for pre-generated values
        with self.state_lock:
            workflow_data = self.workflow_states.get(workflow_id)
            if not workflow_data:
                raise ValueError(f"Workflow {workflow_id} not found in states")
            b2_results = []
            if self.config.mode == "real":
                # Collect B2 results for real mode
                b2_results = [f"b2_result_{i}" for i in range(workflow_data.fanout_count)]

        # Extract workflow number for task ID
        workflow_num = workflow_id.split('-')[-1]
        task_id = f"task-merge-{workflow_data.strategy}-workflow-{workflow_num}"

        # Build task_input and metadata based on mode
        if self.config.mode == "simulation":
            # Use pre-generated merge sleep time
            sleep_time = workflow_data.merge_sleep_time
            task_input = {"sleep_time": sleep_time}

            # Calculate exp_runtime based on strategy
            # min_time strategy: use dataset average
            # other strategies: use actual sampled runtime
            if workflow_data.strategy == "min_time" and hasattr(self.config, 'avg_merge_sleep_time_ms'):
                exp_runtime = self.config.avg_merge_sleep_time_ms
            else:
                exp_runtime = sleep_time * 1000.0  # Convert to milliseconds

            # Simulation mode metadata (with scheduling parameters)
            metadata = {
                "workflow_id": workflow_id,
                "exp_runtime": exp_runtime,
                "exp_cv": 0.40,  # Match real LLM task CV (~39%)
                "exp_skewness": 0.0,  # LLM tasks are approximately symmetric
                "task_type": "merge"  # Use lowercase "merge" to match experiment 07
            }
        else:  # real mode
            summary = f"All research findings for {workflow_id}"
            sentence = MERGE_TEMPLATE.format(summary=summary)
            task_input = {
                "sentence": sentence,
                "max_tokens": workflow_data.max_tokens
            }

            # Real mode metadata (LLM prediction features only)
            metadata = {
                "sentence": sentence,
                "token_length": estimate_token_length(sentence),
                "max_tokens": workflow_data.max_tokens
            }

        return {
            "task_id": task_id,
            "model_id": self.config.model_merge_id,
            "task_input": task_input,
            "metadata": metadata
        }

    def _get_next_task_data(self) -> Optional[str]:
        """
        Get next Merge task from queue (blocking until available).

        Returns:
            workflow_id, or None if stopped
        """
        while not self.stop_event.is_set():
            try:
                return self.merge_trigger_queue.get(timeout=0.1)
            except Empty:
                continue
        return None

    def _submit_task(self, task_data: Any) -> bool:
        """
        Submit Merge task and record in metrics.

        Args:
            task_data: workflow_id string

        Returns:
            True if submission succeeded, False otherwise
        """
        # Record task submission in metrics before submission
        if self.metrics and isinstance(task_data, str):
            workflow_id = task_data
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_num = workflow_id.split('-')[-1]
                    task_id = f"task-merge-{workflow_data.strategy}-workflow-{workflow_num}"
                    self.metrics.record_task_submit(
                        task_id=task_id,
                        workflow_id=workflow_id,
                        task_type="merge"
                    )

        # Log merge task submission with scheduler endpoint
        if isinstance(task_data, str):
            workflow_id = task_data
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    # Determine the scheduler endpoint for merge (uses scheduler_a for merge)
                    merge_scheduler_url = getattr(self.config, 'scheduler_a_url', 'unknown')
                    self.logger.info(
                        f"[MERGE_SUBMIT] workflow={workflow_id}, "
                        f"scheduler_endpoint={merge_scheduler_url}, "
                        f"model_id={self.config.model_merge_id}, "
                        f"fanout_count={workflow_data.fanout_count}, "
                        f"b2_completed={len(workflow_data.b2_complete_times)}"
                    )

        # Call parent implementation to actually submit the task
        return super()._submit_task(task_data)

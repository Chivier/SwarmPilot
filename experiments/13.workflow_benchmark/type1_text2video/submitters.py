"""Text2Video workflow task submitters."""

import random
import threading
import time
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from common import BaseTaskSubmitter, estimate_token_length
from .workflow_data import Text2VideoWorkflowData


# ============================================================================
# Prompt Templates
# ============================================================================

A1_TEMPLATE = "Generate a detailed image generation prompt based on this caption: {caption}"
A2_TEMPLATE = "Generate a negative prompt for image generation to avoid artifacts, based on this positive prompt: Positive prompt: {positive_prompt}"


class A1TaskSubmitter(BaseTaskSubmitter):
    """
    Submits A1 tasks (caption → positive prompt) with Poisson arrivals.

    Uses pre-generated workflow data (from pre_generate_workflows) or generates
    workflows on-the-fly if not provided.
    """

    def __init__(self, captions: List[str], config, workflow_states: Dict,
                 state_lock: threading.Lock, pre_generated_workflows: Optional[List[Text2VideoWorkflowData]] = None,
                 **kwargs):
        """
        Initialize A1 task submitter.

        Args:
            captions: List of video captions (used if pre_generated_workflows is None)
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            pre_generated_workflows: Optional pre-generated workflow data for reproducibility.
                                    If provided, these workflows are used directly (with strategy updated).
                                    If None, workflows are generated on-the-fly.
            **kwargs: Passed to BaseTaskSubmitter (name, scheduler_url, rate_limiter, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock

        if pre_generated_workflows is not None:
            # Use pre-generated workflows (deep copy to avoid mutation across strategies)
            import copy
            self.workflows = []
            for w in pre_generated_workflows:
                workflow_copy = copy.deepcopy(w)
                # Update strategy for this run
                workflow_copy.strategy = getattr(config, 'strategy', 'probabilistic')
                # Reset timing fields for fresh run
                workflow_copy.a1_submit_time = None
                workflow_copy.a1_complete_time = None
                workflow_copy.a2_submit_time = None
                workflow_copy.a2_complete_time = None
                workflow_copy.b_loop_count = 0
                workflow_copy.b_submit_times = []
                workflow_copy.b_complete_times = []
                workflow_copy.workflow_complete_time = None
                workflow_copy.a1_result = None
                workflow_copy.a2_result = None
                self.workflows.append(workflow_copy)
            self.logger.info(f"Using {len(self.workflows)} pre-generated workflows")
        else:
            # Fallback: generate workflows on-the-fly (legacy behavior)
            # Set random seed for reproducibility
            random.seed(42)

            # Determine frame_count source:
            # - If frame_count_config is specified, use config.sample_frame_count()
            # - Otherwise, use data_loader.sample_frame_count() (from benchmark dataset)
            use_config_frame_count = config.frame_count_config is not None

            self.workflows = []
            for i in range(config.num_workflows):
                # Sample max_b_loops from distribution sampler (config-based)
                sampled_max_b_loops = config.sample_max_b_loops()

                # Sample frame_count based on configuration
                if use_config_frame_count:
                    sampled_frame_count = config.sample_frame_count()
                else:
                    sampled_frame_count = config.data_loader.sample_frame_count()

                workflow = Text2VideoWorkflowData(
                    workflow_id=f"workflow-{i:04d}",
                    caption=captions[i % len(captions)],
                    max_b_loops=sampled_max_b_loops,
                    strategy=getattr(config, 'strategy', 'default'),
                    frame_count=sampled_frame_count,
                    max_tokens=getattr(config, 'max_tokens', 512),
                    is_warmup=(i < getattr(config, 'num_warmup', 0))
                )

                # Pre-generate sleep times for simulation mode only
                if config.mode == "simulation":
                    workflow.a1_sleep_time = config.data_loader.sample_llm_runtime_ms() / 1000.0
                    workflow.a2_sleep_time = config.data_loader.sample_llm_runtime_ms() / 1000.0
                    workflow.b_sleep_time = config.data_loader.get_t2vid_runtime_ms(sampled_frame_count) / 1000.0

                self.workflows.append(workflow)
            self.logger.info(f"Generated {len(self.workflows)} workflows on-the-fly")

        # Populate shared workflow_states
        with state_lock:
            for workflow in self.workflows:
                workflow_states[workflow.workflow_id] = workflow

        self.index = 0
        self.task_ids = []  # Track all A1 task IDs

    def _prepare_task_payload(self, workflow_data: Text2VideoWorkflowData) -> Dict[str, Any]:
        """
        Prepare task submission payload for A1.

        Args:
            workflow_data: Workflow data with caption

        Returns:
            JSON payload for /task/submit
        """
        # Build task_input based on mode
        if self.config.mode == "simulation":
            # Use pre-generated sleep_time (always exists in simulation mode)
            task_input = {"sleep_time": workflow_data.a1_sleep_time}

            # Calculate exp_runtime based on strategy
            # min_time strategy: use dataset average
            # other strategies: use actual sampled runtime
            if workflow_data.strategy == "min_time":
                exp_runtime = self.config.data_loader.llm_avg_runtime_ms
            else:
                exp_runtime = workflow_data.a1_sleep_time * 1000.0  # Convert to milliseconds

            # Simulation mode metadata
            # A tasks (LLM) have moderate variability (CV ~39% in real data)
            metadata = {
                "workflow_id": workflow_data.workflow_id,
                "exp_runtime": exp_runtime,
                "exp_cv": 0.40,  # Match real LLM task CV (~39%)
                "exp_skewness": 0.0,  # LLM tasks are approximately symmetric
                "task_type": "A1"
            }
        else:  # real mode
            sentence = A1_TEMPLATE.format(caption=workflow_data.caption)
            task_input = {
                "sentence": sentence,
                "max_tokens": workflow_data.max_tokens
            }

            # Real mode metadata
            token_length = estimate_token_length(workflow_data.caption)
            metadata = {
                "sentence": sentence,
                "token_length": token_length,
                "max_tokens": workflow_data.max_tokens
            }

        # Generate task ID with strategy prefix
        workflow_num = workflow_data.workflow_id.split('-')[-1]
        task_id = f"task-A1-{workflow_data.strategy}-workflow-{workflow_num}"

        return {
            "task_id": task_id,
            "model_id": self.config.model_a_id,
            "task_input": task_input,
            "metadata": metadata
        }

    def _get_next_task_data(self) -> Optional[Text2VideoWorkflowData]:
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
        Submit a single task to the scheduler and track timing.

        Args:
            task_data: Task data to submit

        Returns:
            True if submission succeeded, False otherwise
        """
        # Record workflow start in metrics before submission
        if self.metrics and isinstance(task_data, Text2VideoWorkflowData):
            self.metrics.record_workflow_start(
                workflow_id=task_data.workflow_id,
                workflow_type="text2video",
                metadata={"max_b_loops": task_data.max_b_loops, "strategy": task_data.strategy},
                is_warmup=task_data.is_warmup
            )

            # Record task submission in metrics
            workflow_num = task_data.workflow_id.split('-')[-1]
            task_id = f"task-A1-{task_data.strategy}-workflow-{workflow_num}"
            self.metrics.record_task_submit(
                task_id=task_id,
                workflow_id=task_data.workflow_id,
                task_type="A1"
            )

        # Set submit time BEFORE submission
        submit_time = time.time()

        # Call parent implementation
        success = super()._submit_task(task_data)

        if success and isinstance(task_data, Text2VideoWorkflowData):
            # Update workflow state with submission time
            with self.state_lock:
                task_data.a1_submit_time = submit_time

            # Track task ID (must match the format used in _prepare_task_payload)
            workflow_num = task_data.workflow_id.split('-')[-1]
            strategy = getattr(self.config, 'strategy', 'default')
            task_id = f"task-A1-{strategy}-workflow-{workflow_num}"
            self.task_ids.append(task_id)

        return success


class A2TaskSubmitter(BaseTaskSubmitter):
    """
    Submits A2 tasks (positive prompt → negative prompt).

    Receives A1 completion events from queue and submits corresponding A2 tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 a1_result_queue: Queue, **kwargs):
        """
        Initialize A2 task submitter.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            a1_result_queue: Queue receiving (workflow_id, a1_result) tuples from A1 receiver
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.a1_result_queue = a1_result_queue
        self.task_ids = []  # Track all A2 task IDs

    def _prepare_task_payload(self, task_data: tuple) -> Dict[str, Any]:
        """
        Prepare task submission payload for A2.

        Args:
            task_data: Tuple of (workflow_id, a1_result)

        Returns:
            JSON payload for /task/submit
        """
        workflow_id, a1_result = task_data

        # Get workflow_data
        with self.state_lock:
            workflow_data = self.workflow_states.get(workflow_id)

        # Build task_input based on mode
        if self.config.mode == "simulation":
            # Use pre-generated sleep_time (always exists in simulation mode)
            sleep_time = workflow_data.a2_sleep_time
            task_input = {"sleep_time": sleep_time}

            # Calculate exp_runtime based on strategy
            # min_time strategy: use dataset average
            # other strategies: use actual sampled runtime
            strategy = workflow_data.strategy
            if strategy == "min_time":
                exp_runtime = self.config.data_loader.llm_avg_runtime_ms
            else:
                exp_runtime = sleep_time * 1000.0  # Convert to milliseconds

            # Simulation mode metadata
            # A tasks (LLM) have moderate variability (CV ~39% in real data)
            metadata = {
                "workflow_id": workflow_id,
                "exp_runtime": exp_runtime,
                "exp_cv": 0.40,  # Match real LLM task CV (~39%)
                "exp_skewness": 0.0,  # LLM tasks are approximately symmetric
                "task_type": "A2"
            }
        else:  # real mode
            sentence = A2_TEMPLATE.format(positive_prompt=a1_result)
            max_tokens = workflow_data.max_tokens if workflow_data else self.config.max_tokens
            task_input = {
                "sentence": sentence,
                "max_tokens": max_tokens
            }

            # Real mode metadata
            token_length = estimate_token_length(sentence)
            metadata = {
                "sentence": sentence,
                "token_length": token_length,
                "max_tokens": max_tokens
            }

        # Generate task ID with strategy prefix
        strategy = workflow_data.strategy if workflow_data else "probabilistic"
        workflow_num = workflow_id.split('-')[-1]
        task_id = f"task-A2-{strategy}-workflow-{workflow_num}"

        return {
            "task_id": task_id,
            "model_id": self.config.model_a_id,
            "task_input": task_input,
            "metadata": metadata
        }

    def _get_next_task_data(self) -> Optional[tuple]:
        """
        Get next A2 task from queue (blocking until available).

        Returns:
            Tuple of (workflow_id, a1_result), or None if stopped
        """
        while not self.stop_event.is_set():
            try:
                # Block for 0.1s to allow graceful shutdown
                return self.a1_result_queue.get(timeout=0.1)
            except Empty:
                # Queue empty, keep waiting
                continue
        return None

    def _submit_task(self, task_data: Any) -> bool:
        """
        Submit a single task to the scheduler and track timing.

        Args:
            task_data: Task data to submit

        Returns:
            True if submission succeeded, False otherwise
        """
        # Record task submission in metrics before submission
        if self.metrics and isinstance(task_data, tuple):
            workflow_id = task_data[0]
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_num = workflow_id.split('-')[-1]
                    task_id = f"task-A2-{workflow_data.strategy}-workflow-{workflow_num}"
                    self.metrics.record_task_submit(
                        task_id=task_id,
                        workflow_id=workflow_id,
                        task_type="A2"
                    )

        # Set submit time BEFORE submission
        submit_time = time.time()

        # Call parent implementation
        success = super()._submit_task(task_data)

        if success and isinstance(task_data, tuple):
            workflow_id = task_data[0]

            # Update workflow state with submission time
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_data.a2_submit_time = submit_time

            # Track task ID (must match the format used in _prepare_task_payload)
            workflow_num = workflow_id.split('-')[-1]
            strategy = workflow_data.strategy if workflow_data else "probabilistic"
            task_id = f"task-A2-{strategy}-workflow-{workflow_num}"
            self.task_ids.append(task_id)

        return success


class BTaskSubmitter(BaseTaskSubmitter):
    """
    Submits B tasks (video generation) with loop control.

    Receives A2 completion events from queue and submits B tasks.
    Implements B-loop logic: re-submits B task if loop count < max_b_loops.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 a2_result_queue: Queue, **kwargs):
        """
        Initialize B task submitter.

        Args:
            config: Text2VideoConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            a2_result_queue: Queue receiving (workflow_id, a2_result) tuples from A2 receiver
            **kwargs: Passed to BaseTaskSubmitter
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock
        self.a2_result_queue = a2_result_queue
        self.task_ids = []  # Track all B task IDs (all iterations)

    def _prepare_task_payload(self, task_data: tuple) -> Dict[str, Any]:
        """
        Prepare task submission payload for B.

        Args:
            task_data: Tuple of (workflow_id, a2_result, loop_iteration)

        Returns:
            JSON payload for /task/submit
        """
        workflow_id, a2_result, loop_iteration = task_data

        # Get workflow_data
        with self.state_lock:
            workflow_data = self.workflow_states.get(workflow_id)

        # Build task_input based on mode
        if self.config.mode == "simulation":
            # Use pre-generated sleep_time (always exists in simulation mode)
            sleep_time = workflow_data.b_sleep_time
            task_input = {"sleep_time": sleep_time}

            # Calculate exp_runtime based on strategy
            # min_time strategy: use dataset average
            # other strategies: use actual sampled runtime
            strategy = workflow_data.strategy
            if strategy == "min_time":
                exp_runtime = self.config.data_loader.t2vid_avg_runtime_ms
            else:
                exp_runtime = sleep_time * 1000.0  # Convert to milliseconds

            # Simulation mode metadata - includes all necessary fields
            # B tasks (T2VID) have high variability and long tail (CV ~112%, skewness ~2.7 in real data)
            metadata = {
                "workflow_id": workflow_id,
                "exp_runtime": exp_runtime,
                "exp_cv": 1.0,  # Match real T2VID task CV (~112%)
                "exp_skewness": 2.5,  # Match real T2VID skewness (~2.7)
                "frame_count": workflow_data.frame_count,
                "task_type": "B",
                "b_iteration": loop_iteration,
                "max_b_loops": workflow_data.max_b_loops
            }
        else:  # real mode - completely different structure
            # Real mode uses caption for both prompt and negative_prompt (simplified)
            caption = workflow_data.caption if workflow_data else ""
            frame_count = workflow_data.frame_count if workflow_data else 16

            task_input = {
                "prompt": caption,  # Use caption as positive prompt
                "negative_prompt": caption,  # Simplified: also use caption as negative prompt
                "frames": frame_count
            }

            # Real mode metadata - only includes prompt lengths and frames
            metadata = {
                "positive_prompt_length": estimate_token_length(caption),
                "negative_prompt_length": estimate_token_length(caption),
                "frames": frame_count
            }

        # Generate task ID with strategy prefix
        strategy = workflow_data.strategy if workflow_data else "probabilistic"
        workflow_num = workflow_id.split('-')[-1]
        task_id = f"task-B{loop_iteration}-{strategy}-workflow-{workflow_num}"

        return {
            "task_id": task_id,
            "model_id": self.config.model_b_id,
            "task_input": task_input,
            "metadata": metadata
        }

    def _get_next_task_data(self) -> Optional[tuple]:
        """
        Get next B task from queue (blocking until available).

        Returns:
            Tuple of (workflow_id, a2_result, loop_iteration), or None if stopped
        """
        while not self.stop_event.is_set():
            try:
                # Block for 0.1s to allow graceful shutdown
                return self.a2_result_queue.get(timeout=0.1)
            except Empty:
                continue
        return None

    def add_task(self, workflow_id: str, a2_result: str, loop_iteration: int):
        """
        Add a B task to the submission queue.

        Called by B receiver after completion to trigger next iteration.

        Args:
            workflow_id: Workflow identifier
            a2_result: Negative prompt from A2
            loop_iteration: Current loop iteration number (1-based)
        """
        self.a2_result_queue.put((workflow_id, a2_result, loop_iteration))

    def _submit_task(self, task_data: Any) -> bool:
        """
        Submit a single task to the scheduler and track timing.

        Args:
            task_data: Task data to submit

        Returns:
            True if submission succeeded, False otherwise
        """
        # Record task submission in metrics before submission
        if self.metrics and isinstance(task_data, tuple):
            workflow_id, a2_result, loop_iteration = task_data
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    workflow_num = workflow_id.split('-')[-1]
                    task_id = f"task-B{loop_iteration}-{workflow_data.strategy}-workflow-{workflow_num}"
                    self.metrics.record_task_submit(
                        task_id=task_id,
                        workflow_id=workflow_id,
                        task_type="B"
                    )

        # Log first and last B request parameters for debugging
        if isinstance(task_data, tuple) and self.config.mode == "simulation":
            workflow_id, a2_result, loop_iteration = task_data
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    is_first = (loop_iteration == 1)
                    is_last = (loop_iteration == workflow_data.max_b_loops)
                    if is_first or is_last:
                        position = "FIRST" if is_first else "LAST"
                        self.logger.info(
                            f"[{position} B] {workflow_id}: "
                            f"iteration={loop_iteration}/{workflow_data.max_b_loops}, "
                            f"sleep_time={workflow_data.b_sleep_time:.3f}s, "
                            f"frame_count={workflow_data.frame_count}, "
                            f"strategy={workflow_data.strategy}"
                        )

        # Set submit time BEFORE submission
        submit_time = time.time()

        # Call parent implementation
        success = super()._submit_task(task_data)

        if success and isinstance(task_data, tuple):
            workflow_id, a2_result, loop_iteration = task_data

            # Update workflow state with submission time
            with self.state_lock:
                workflow_data = self.workflow_states.get(workflow_id)
                if workflow_data:
                    # First B iteration, initialize the list
                    if loop_iteration == 1:
                        workflow_data.b_submit_times = [submit_time]
                    else:
                        # Subsequent iterations
                        if submit_time not in workflow_data.b_submit_times:
                            workflow_data.b_submit_times.append(submit_time)

            # Track task ID (must match the format used in _prepare_task_payload)
            workflow_num = workflow_id.split('-')[-1]
            strategy = workflow_data.strategy if workflow_data else "probabilistic"
            task_id = f"task-B{loop_iteration}-{strategy}-workflow-{workflow_num}"
            self.task_ids.append(task_id)

        return success

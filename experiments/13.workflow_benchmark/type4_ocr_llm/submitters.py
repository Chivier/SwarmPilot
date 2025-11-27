"""OCR+LLM workflow task submitters."""

import random
import threading
import time
from queue import Queue, Empty
from typing import Any, Dict, List, Optional

from common import BaseTaskSubmitter, estimate_token_length
from .workflow_data import OCRLLMWorkflowData, generate_dummy_images


# ============================================================================
# Prompt Templates
# ============================================================================

LLM_TEMPLATE = "Please analyze and summarize the following OCR-extracted text:\n\n{ocr_text}"


class ATaskSubmitter(BaseTaskSubmitter):
    """
    Submits A tasks (OCR) with Poisson arrivals.

    Pre-generates all workflow data upfront, then submits A tasks
    following a Poisson process controlled by the rate limiter.
    """

    def __init__(self, images: List[str], config, workflow_states: Dict,
                 state_lock: threading.Lock, **kwargs):
        """
        Initialize A task submitter (OCR).

        Args:
            images: List of base64-encoded images
            config: OCRLLMConfig instance
            workflow_states: Shared workflow state dictionary
            state_lock: Lock for workflow_states access
            **kwargs: Passed to BaseTaskSubmitter (name, scheduler_url, rate_limiter, etc.)
        """
        super().__init__(**kwargs)
        self.config = config
        self.workflow_states = workflow_states
        self.state_lock = state_lock

        # Set random seed for reproducibility (distribution sampling)
        random.seed(42)

        # Use provided images or generate dummy ones
        if not images:
            self.logger.info("No images provided, generating dummy images")
            images = generate_dummy_images(config.num_workflows)

        # Pre-generate all workflow data
        self.workflows = []
        for i in range(config.num_workflows):
            workflow = OCRLLMWorkflowData(
                workflow_id=f"workflow-{i:04d}",
                image_data=images[i % len(images)],
                strategy=getattr(config, 'strategy', 'probabilistic'),
                ocr_languages=config.ocr_languages.split(",") if isinstance(config.ocr_languages, str) else config.ocr_languages,
                ocr_detail_level=getattr(config, 'ocr_detail_level', 'standard'),
                max_tokens=getattr(config, 'max_tokens', 512),
                is_warmup=(i < getattr(config, 'num_warmup', 0))
            )

            # Pre-generate sleep times for simulation mode
            if config.mode == "simulation":
                workflow.a_sleep_time = config.sample_sleep_time_a()
                workflow.b_sleep_time = config.sample_sleep_time_b()

            self.workflows.append(workflow)

        # Populate shared workflow_states
        with state_lock:
            for workflow in self.workflows:
                workflow_states[workflow.workflow_id] = workflow

        self.index = 0
        self.task_ids = []  # Track all A task IDs

        # Calculate average sleep times for min_time strategy (simulation mode only)
        if config.mode == "simulation":
            a_times = [w.a_sleep_time for w in self.workflows if w.a_sleep_time is not None]
            b_times = [w.b_sleep_time for w in self.workflows if w.b_sleep_time is not None]

            # Store averages in config for other submitters to use
            config.avg_a_sleep_time_ms = (sum(a_times) / len(a_times)) * 1000.0 if a_times else 3000.0
            config.avg_b_sleep_time_ms = (sum(b_times) / len(b_times)) * 1000.0 if b_times else 10000.0

            self.logger.info(
                f"Calculated average sleep times for min_time strategy: "
                f"A={config.avg_a_sleep_time_ms:.1f}ms, "
                f"B={config.avg_b_sleep_time_ms:.1f}ms"
            )

        self.logger.info(f"Pre-generated {len(self.workflows)} workflows")

    def _prepare_task_payload(self, workflow_data: OCRLLMWorkflowData) -> Dict[str, Any]:
        """
        Prepare task submission payload for A (OCR).

        Args:
            workflow_data: Workflow data with image

        Returns:
            JSON payload for /task/submit
        """
        # Build task_input based on mode
        if self.config.mode == "simulation":
            # Use pre-generated sleep_time
            if workflow_data.a_sleep_time is None:
                workflow_data.a_sleep_time = self.config.sample_sleep_time_a()
            task_input = {"sleep_time": workflow_data.a_sleep_time}

            # Calculate exp_runtime based on strategy
            # min_time strategy: use dataset average
            # other strategies: use actual sampled runtime
            if workflow_data.strategy == "min_time" and hasattr(self.config, 'avg_a_sleep_time_ms'):
                exp_runtime = self.config.avg_a_sleep_time_ms
            else:
                exp_runtime = workflow_data.a_sleep_time * 1000.0  # Convert to milliseconds

            # Simulation mode metadata
            metadata = {
                "workflow_id": workflow_data.workflow_id,
                "exp_runtime": exp_runtime,
                "task_type": "A"
            }
        else:  # real mode
            # Real OCR task input
            task_input = {
                "image_data": workflow_data.image_data,
                "languages": workflow_data.ocr_languages,
                "detail_level": workflow_data.ocr_detail_level,
                "confidence_threshold": 0.0,
                "paragraph_mode": True
            }

            # Real mode metadata
            metadata = {
                "workflow_id": workflow_data.workflow_id,
                "task_type": "A",
                "image_size": len(workflow_data.image_data)  # Base64 size as proxy
            }

        # Generate task ID with strategy prefix
        workflow_num = workflow_data.workflow_id.split('-')[-1]
        task_id = f"task-A-{workflow_data.strategy}-workflow-{workflow_num}"

        return {
            "task_id": task_id,
            "model_id": self.config.model_a_id,
            "task_input": task_input,
            "metadata": metadata
        }

    def _get_next_task_data(self) -> Optional[OCRLLMWorkflowData]:
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
        if self.metrics and isinstance(task_data, OCRLLMWorkflowData):
            self.metrics.record_workflow_start(
                workflow_id=task_data.workflow_id,
                workflow_type="ocr_llm",
                metadata={"strategy": task_data.strategy},
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

        # Set submit time BEFORE submission
        submit_time = time.time()

        # Call parent implementation
        success = super()._submit_task(task_data)

        if success and isinstance(task_data, OCRLLMWorkflowData):
            # Update workflow state with submission time
            with self.state_lock:
                task_data.a_submit_time = submit_time

            # Track task ID
            workflow_num = task_data.workflow_id.split('-')[-1]
            strategy = getattr(self.config, 'strategy', 'probabilistic')
            task_id = f"task-A-{strategy}-workflow-{workflow_num}"
            self.task_ids.append(task_id)

        return success


class BTaskSubmitter(BaseTaskSubmitter):
    """
    Submits B tasks (LLM) after receiving OCR results.

    Receives A completion events from queue and submits corresponding B tasks.
    """

    def __init__(self, config, workflow_states: Dict, state_lock: threading.Lock,
                 a_result_queue: Queue, **kwargs):
        """
        Initialize B task submitter (LLM).

        Args:
            config: OCRLLMConfig instance
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
        self.task_ids = []  # Track all B task IDs

    def _prepare_task_payload(self, task_data: tuple) -> Dict[str, Any]:
        """
        Prepare task submission payload for B (LLM).

        Args:
            task_data: Tuple of (workflow_id, a_result)

        Returns:
            JSON payload for /task/submit
        """
        workflow_id, a_result = task_data

        # Get workflow_data
        with self.state_lock:
            workflow_data = self.workflow_states.get(workflow_id)

        # Build task_input based on mode
        if self.config.mode == "simulation":
            # Use pre-generated sleep_time or generate if not available
            if workflow_data and workflow_data.b_sleep_time is None:
                workflow_data.b_sleep_time = self.config.sample_sleep_time_b()
            sleep_time = workflow_data.b_sleep_time if workflow_data else self.config.sample_sleep_time_b()
            task_input = {"sleep_time": sleep_time}

            # Calculate exp_runtime based on strategy
            # min_time strategy: use dataset average
            # other strategies: use actual sampled runtime
            strategy = workflow_data.strategy if workflow_data else "probabilistic"
            if strategy == "min_time" and hasattr(self.config, 'avg_b_sleep_time_ms'):
                exp_runtime = self.config.avg_b_sleep_time_ms
            else:
                exp_runtime = sleep_time * 1000.0  # Convert to milliseconds

            # Simulation mode metadata
            metadata = {
                "workflow_id": workflow_id,
                "exp_runtime": exp_runtime,
                "task_type": "B"
            }
        else:  # real mode
            # Format prompt with OCR result
            sentence = LLM_TEMPLATE.format(ocr_text=a_result or "")
            max_tokens = workflow_data.max_tokens if workflow_data else self.config.max_tokens

            task_input = {
                "sentence": sentence,
                "max_tokens": max_tokens
            }

            # Real mode metadata
            token_length = estimate_token_length(sentence)
            metadata = {
                "workflow_id": workflow_id,
                "task_type": "B",
                "sentence": sentence,
                "token_length": token_length,
                "max_tokens": max_tokens
            }

        # Generate task ID with strategy prefix
        strategy = workflow_data.strategy if workflow_data else "probabilistic"
        workflow_num = workflow_id.split('-')[-1]
        task_id = f"task-B-{strategy}-workflow-{workflow_num}"

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
            Tuple of (workflow_id, a_result), or None if stopped
        """
        while not self.stop_event.is_set():
            try:
                # Block for 0.1s to allow graceful shutdown
                return self.a_result_queue.get(timeout=0.1)
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
                    task_id = f"task-B-{workflow_data.strategy}-workflow-{workflow_num}"
                    self.metrics.record_task_submit(
                        task_id=task_id,
                        workflow_id=workflow_id,
                        task_type="B"
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
                    workflow_data.b_submit_time = submit_time

            # Track task ID
            workflow_num = workflow_id.split('-')[-1]
            strategy = workflow_data.strategy if workflow_data else "probabilistic"
            task_id = f"task-B-{strategy}-workflow-{workflow_num}"
            self.task_ids.append(task_id)

        return success

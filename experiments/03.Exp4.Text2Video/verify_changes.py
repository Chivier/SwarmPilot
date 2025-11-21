import sys
import os
from unittest.mock import MagicMock

# Mock dependencies before importing collect_training_data
sys.modules["httpx"] = MagicMock()
sys.modules["tiktoken"] = MagicMock()
sys.modules["pynvml"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["tqdm"] = MagicMock()

# Add the experiment directory to the python path
sys.path.append("/home/yanweiye/Projects/swarmpilot-refresh/experiments/03.Exp4.Text2Video")

from collect_training_data import extract_tasks_from_dataset, A1_TEMPLATE, A2_TEMPLATE

# Mock dataset
dataset = [
    {"id": "test-001", "caption": "A beautiful sunset over the ocean"}
]

# Test LLM task extraction
print("Testing LLM task extraction...")
tasks = extract_tasks_from_dataset(dataset, "llm_service_small_model")

assert len(tasks) == 2, f"Expected 2 tasks, got {len(tasks)}"

a1_task = next(t for t in tasks if t["task_type"] == "A1")
a2_task = next(t for t in tasks if t["task_type"] == "A2")

expected_a1_sentence = A1_TEMPLATE.format(caption="A beautiful sunset over the ocean")
expected_a2_sentence = A2_TEMPLATE.format(positive_prompt="A beautiful sunset over the ocean")

print(f"A1 Sentence: {a1_task['sentence']}")
assert a1_task['sentence'] == expected_a1_sentence, "A1 sentence mismatch"

print(f"A2 Sentence: {a2_task['sentence']}")
assert a2_task['sentence'] == expected_a2_sentence, "A2 sentence mismatch"

print("Verification successful!")

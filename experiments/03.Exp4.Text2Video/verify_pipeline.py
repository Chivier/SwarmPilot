import asyncio
import json
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

# Mock httpx before importing the script
mock_httpx = MagicMock()
mock_httpx.AsyncClient.return_value.post = AsyncMock()
mock_httpx.AsyncClient.return_value.get = AsyncMock()
mock_httpx.AsyncClient.return_value.aclose = AsyncMock()
sys.modules["httpx"] = mock_httpx

# Add parent directory to path to import the script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collect_training_data import main, ServiceClient, PredictorClient

async def mock_execute_task(self, task_input):
    # Mock response based on task type
    if "sentence" in task_input:
        # LLM Task
        if "caption" in task_input.get("sentence", ""):
            # A1
            return {
                "execution_time_ms": 100.0,
                "success": True,
                "result": {"output": "Positive prompt: " + task_input["sentence"]},
                "error": None,
                "instance_id": self.instance_id
            }
        else:
            # A2
            return {
                "execution_time_ms": 100.0,
                "success": True,
                "result": {"output": "Negative prompt for: " + task_input["sentence"]},
                "error": None,
                "instance_id": self.instance_id
            }
    elif "prompt" in task_input:
        # T2Vid Task
        return {
            "execution_time_ms": 500.0,
            "success": True,
            "result": {"output": "video_url"},
            "error": None,
            "instance_id": self.instance_id
        }
    return {"success": False, "error": "Unknown task"}

async def mock_submit_training_data(self, *args, **kwargs):
    return {"status": "success"}

async def mock_predict(self, *args, **kwargs):
    return {
        "result": {
            "expected_runtime_ms": 150.0,
            "error_margin_ms": 10.0,
            "quantiles": {"0.1": 140.0, "0.5": 150.0, "0.9": 160.0}
        }
    }

async def run_test():
    print("Starting verification test...")
    
    # Mock ServiceClient.execute_task
    with patch('collect_training_data.ServiceClient.execute_task', side_effect=mock_execute_task, autospec=True):
        with patch('collect_training_data.PredictorClient.submit_training_data', side_effect=mock_submit_training_data, autospec=True):
            with patch('collect_training_data.PredictorClient.predict', side_effect=mock_predict, autospec=True):
                
                # Run main with test args
                test_args = [
                    "collect_training_data.py",
                    "--config", "config_pipeline_test.json",
                    "--output_file", "test_output.json",
                    "--llm_limit", "2",
                    "--t2vid_limit", "2"
                ]
                
                with patch.object(sys, 'argv', test_args):
                    await main()

    # Verify output
    if os.path.exists("test_output.json"):
        with open("test_output.json", "r") as f:
            data = json.load(f)
            
        llm_samples = data.get("llm_samples", [])
        t2vid_samples = data.get("t2vid_samples", [])
        
        print(f"LLM Samples: {len(llm_samples)}")
        print(f"T2Vid Samples: {len(t2vid_samples)}")
        
        # Verify LLM samples (A1 + A2 for 2 captions = 4 samples)
        if len(llm_samples) >= 4:
            print("PASS: LLM sample count looks correct (>= 4)")
        else:
            print(f"FAIL: Expected >= 4 LLM samples, got {len(llm_samples)}")

        # Verify T2Vid samples (2 prompts * 4 frame counts = 8 samples)
        if len(t2vid_samples) == 8:
            print("PASS: T2Vid sample count is correct (8)")
        else:
            print(f"FAIL: Expected 8 T2Vid samples, got {len(t2vid_samples)}")
            
        # Verify frame counts
        frames = set(s["frames"] for s in t2vid_samples)
        if frames == {30.0, 60.0, 90.0, 120.0}:
            print("PASS: Frame counts are correct")
        else:
            print(f"FAIL: Frame counts mismatch: {frames}")

    else:
        print("FAIL: Output file not created")

if __name__ == "__main__":
    asyncio.run(run_test())

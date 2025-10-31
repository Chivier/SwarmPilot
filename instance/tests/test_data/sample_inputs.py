"""
Sample test data for task inputs and outputs
"""

# Sample task inputs for different types of models
SAMPLE_TASK_INPUTS = {
    "simple_text": {
        "prompt": "Hello, world!",
    },
    "with_parameters": {
        "prompt": "Translate this text",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 100,
        }
    },
    "complex_nested": {
        "prompt": "Complex query",
        "context": {
            "history": ["Previous message 1", "Previous message 2"],
            "metadata": {
                "user_id": "user-123",
                "session_id": "session-456"
            }
        },
        "options": {
            "stream": False,
            "stop_sequences": ["\n", "END"],
        }
    },
    "sleep_model": {
        "duration": 5,
        "message": "Sleep for 5 seconds"
    }
}

# Sample expected outputs
SAMPLE_TASK_OUTPUTS = {
    "simple_text": {
        "output": "Processed: Hello, world!",
        "status": "success"
    },
    "with_parameters": {
        "output": "Translation result",
        "tokens_used": 42,
        "finish_reason": "stop"
    },
    "sleep_model": {
        "slept_for": 5,
        "message": "Sleep completed"
    }
}

# Sample error scenarios
SAMPLE_ERRORS = {
    "timeout": "Request timeout after 300 seconds",
    "invalid_input": "Invalid input format: missing required field 'prompt'",
    "model_error": "Model inference failed: internal error",
    "no_model": "No model is currently running",
}

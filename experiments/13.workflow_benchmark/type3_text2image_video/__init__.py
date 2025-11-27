"""Type 3: Text2Image+Video Workflow.

Workflow pattern: LLM (A) -> FLUX (C) -> T2VID (B loops)

This workflow type modifies type1 by replacing the second LLM step (A2)
with a FLUX text-to-image model, creating a chain:
- A: LLM generates positive prompt from caption
- C: FLUX generates image from positive prompt (with configurable resolution)
- B: T2VID generates video (1-N iterations per workflow)

Key differences from type1:
- Uses separate scheduler C for FLUX tasks (port 8300)
- Resolution is configurable (512x512 or 1024x1024)
- T2VID negative prompt is fixed as "blur"
"""

from .config import Text2ImageVideoConfig
from .workflow_data import Text2ImageVideoWorkflowData, load_captions
from .data_loader import Type3DataLoader, FLUX_TIMING_MS
from .submitters import ATaskSubmitter, CTaskSubmitter, BTaskSubmitter
from .receivers import ATaskReceiver, CTaskReceiver, BTaskReceiver

__all__ = [
    # Config
    "Text2ImageVideoConfig",
    # Data structures
    "Text2ImageVideoWorkflowData",
    "load_captions",
    # Data loader
    "Type3DataLoader",
    "FLUX_TIMING_MS",
    # Submitters
    "ATaskSubmitter",
    "CTaskSubmitter",
    "BTaskSubmitter",
    # Receivers
    "ATaskReceiver",
    "CTaskReceiver",
    "BTaskReceiver",
]

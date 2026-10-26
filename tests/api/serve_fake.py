"""Run one fake engine: ``python serve_fake.py <style> <port>``."""

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).parent))
from fake_engines import ENGINES

if __name__ == "__main__":
    style, port = sys.argv[1], int(sys.argv[2])
    uvicorn.run(
        ENGINES[style](), host="127.0.0.1", port=port, log_level="warning"
    )

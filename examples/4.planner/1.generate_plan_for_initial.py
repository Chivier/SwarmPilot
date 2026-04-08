# Before you run this example, you should start server first
#
# Start the planner:
#   cd planner && uv run python -m src.api
#
# Then run this script:
#   uv run python examples/1.generate_plan_for_initial.py

import httpx
from pprint import pprint

PLANNER_URL = "http://localhost:8000"

response = httpx.get(PLANNER_URL + "/v1/health")

if not response.status_code == 200:
    print("Server is not started, please check")
    exit(1)

M = 30
N = 2
B = [[7, 7]] * M
initial = [-1] * M

payload = {
    "M": M,
    "N": N,
    "B": B,
    "initial": initial,
    "a": 1,
    "target": [10, 200],
    "algorithm": "simulated_annealing",
    "objective_method": "ratio_difference",
}

response = httpx.post(PLANNER_URL + "/v1/plan", json=payload)

if response.status_code == 200:
    pprint(response.json())
else:
    print("Request Failed")
    print(response.text)
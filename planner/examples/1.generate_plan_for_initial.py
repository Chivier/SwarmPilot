# Before you run this example, you should start server first

import requests
from pprint import pprint

PLANNER_URL = "http://localhost:8000"

response = requests.get(PLANNER_URL + '/health')

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
    "objective_method": "ratio_difference"
  }

response = requests.post(PLANNER_URL + "/plan", json=payload)

if response.status_code == 200:
  pprint(response.json())
else:
  print("Request Failed")
  print(response.text)
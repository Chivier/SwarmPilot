# Before you run this example, you should start server first

import requests
from pprint import pprint

PLANNER_URL = "http://localhost:8000"

response = requests.get(PLANNER_URL + '/health')

if not response.status_code == 200:
  print("Server is not started, please check")
  exit(1)

payload = {
    "M": 4,
    "N": 3,
    "B": [[10, 5, 0], [8, 6, 4], [0, 10, 8], [6, 0, 12]],
    "initial": [0, 1, 2, 2],
    "a": 0.5,
    "target": [20, 30, 25],
    "algorithm": "simulated_annealing"
  }

response = requests.post(PLANNER_URL + "/plan", json=payload)

if response.status_code == 200:
  pprint(response.json())
else:
  print("Request Failed")
  print(response.text)
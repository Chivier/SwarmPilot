import json
import os
from pathlib import Path
import random

DATA_DIR = Path(__file__).parent / "data"

with open(DATA_DIR / "query.jsonl", "r") as f:
  query_data = [json.loads(line) for line in f]

with open(DATA_DIR / "query_gen.jsonl", "r") as f:
  query_gen_data = [json.loads(line) for line in f]

with open(DATA_DIR / "summary.jsonl", "r") as f:
  summary_data = [json.loads(line) for line in f]

dataset = []

for q, qg, s in zip(query_data, query_gen_data, summary_data):
  query_num = random.randint(5, 20)
  dataset.append({
    "boot": qg["input"],
    "queries": [random.choice(query_data) for _ in range(query_num)],
    "summary": s["input"]
  })

with open(DATA_DIR / "dataset.jsonl", "w", encoding="utf-8") as f:
  for item in dataset:
    f.write(json.dumps(item, ensure_ascii=False) + "\n")
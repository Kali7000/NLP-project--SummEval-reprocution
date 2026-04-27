
# explore_data.py
import json
import pandas as pd
from collections import defaultdict

def load_summeval(path="data/model_annotations.aligned.jsonl"):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

records = load_summeval()

print(f"Total records: {len(records)}")
print(f"\nKeys in each record: {list(records[0].keys())}")

# Inspect a single record
r = records[0]
print(f"\n--- Example Record ---")
print(f"Model: {r.get('model_id', 'N/A')}")
print(f"Article (first 200 chars): {r['text'][:200]}...")
print(f"Summary: {r['decoded'][:200]}...")
print(f"Reference summaries (count): {len(r.get('references', []))}")

# Human scores are averaged across annotators
# expert_annotations contains per-annotator scores
expert = r.get('expert_annotations', [])
print(f"\nExpert annotations ({len(expert)} annotators):")
for ann in expert:
    print(f"  coherence={ann['coherence']}, consistency={ann['consistency']}, "
          f"fluency={ann['fluency']}, relevance={ann['relevance']}")

# Check which models are in the dataset
models = set(r.get('model_id', 'unknown') for r in records)
print(f"\nModels in dataset ({len(models)}): {sorted(models)}")
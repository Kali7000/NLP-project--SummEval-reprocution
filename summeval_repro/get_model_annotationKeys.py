

import json

with open("data/model_annotations.aligned.jsonl") as f:
    r = json.loads(f.readline())

print("All keys:", list(r.keys()))
print("\nSample values:")
for k, v in r.items():
    val = str(v)[:80]
    print(f"  {k!r}: {val}")
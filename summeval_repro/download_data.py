

# download_data.py
import requests
import os

url = "https://raw.githubusercontent.com/Yale-LILY/SummEval/master/data/model_annotations.aligned.jsonl"

os.makedirs("data", exist_ok=True)
output_path = "data/model_annotations.aligned.jsonl"

print("Downloading SummEval data...")
response = requests.get(url)
response.raise_for_status()

with open(output_path, "wb") as f:
    f.write(response.content)

print(f"Saved to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
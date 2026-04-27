

# run_all.py  ← Run this file to execute the full pipeline
import subprocess, sys

steps = [
    ("Download data",         "download_data.py"),
    ("Build DataFrame",       "build_dataframe.py"),
    ("Compute ROUGE",         "compute_rouge.py"),
    ("Compute BERTScore",     "compute_bertscore.py"),
    ("Correlation analysis",  "correlation_analysis.py"),
    ("Generate figures",      "visualize.py"),
]

for name, script in steps:
    print(f"\n{'='*50}")
    print(f"  STEP: {name}")
    print(f"{'='*50}")
    result = subprocess.run([sys.executable, script], check=True)
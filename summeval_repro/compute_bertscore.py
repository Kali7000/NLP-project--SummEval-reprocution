

# compute_bertscore.py
import pandas as pd
import numpy as np
import ast
from bert_score import score as bert_score
from tqdm import tqdm

import torch
from bert_score import BERTScorer  # use the class, not the function

def compute_bertscore_fast(df):
    """
    Loads roberta-large ONCE, then scores all rows.
    Should take ~5 min on CPU, ~1 min on GPU.
    """
    print("Loading roberta-large model (one time only)...")
    scorer = BERTScorer(
        model_type="roberta-large",
        lang="en",
        rescale_with_baseline=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Model loaded. Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    all_f1 = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f"  Row {i}/{len(df)}")

        summary = row['summary']
        refs = row['references']
        if isinstance(refs, str):
            refs = ast.literal_eval(refs)

        if not refs:
            all_f1.append(np.nan)
            continue

        # Score summary against all references, take max
        cands = [summary] * len(refs)
        P, R, F1 = scorer.score(cands, refs, verbose=False)
        all_f1.append(F1.max().item())

    df['bertscore_f1'] = all_f1
    print(f"\nDone! BERTScore F1 mean: {df['bertscore_f1'].mean():.4f}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/summeval_with_metrics.csv")
    df = compute_bertscore_fast(df)
    df.to_csv("data/summeval_with_metrics.csv", index=False)
    print("Saved.")






def compute_bertscore_batched(df, batch_size=32):
    """
    Compute BERTScore (F1) for each summary vs. its references.
    Uses roberta-large (default, matches paper).
    """
    summaries = df['summary'].tolist()
    
    # For multi-reference: score vs each ref, take max
    # We'll flatten and track indices
    all_bert_f1 = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="BERTScore"):
        batch = df.iloc[i:i+batch_size]
        batch_scores = []
        
        for _, row in batch.iterrows():
            summary = row['summary']
            refs = row['references']
            if isinstance(refs, str):
                refs = ast.literal_eval(refs)
            
            if not refs:
                batch_scores.append(np.nan)
                continue
            
            # Score against each reference
            cands = [summary] * len(refs)
            P, R, F1 = bert_score(
                cands, refs,
                lang="en",
                model_type="roberta-large",
                verbose=False
            )
            batch_scores.append(F1.max().item())  # take max over references
        
        all_bert_f1.extend(batch_scores)
    
    df['bertscore_f1'] = all_bert_f1
    print(f"\nBERTScore F1 average: {df['bertscore_f1'].mean():.4f}")
    return df
"""
if __name__ == "__main__":
    df = pd.read_csv("data/summeval_with_metrics.csv")
    df = compute_bertscore_batched(df)
    df.to_csv("data/summeval_with_metrics.csv", index=False)
    print("Updated data/summeval_with_metrics.csv")

"""
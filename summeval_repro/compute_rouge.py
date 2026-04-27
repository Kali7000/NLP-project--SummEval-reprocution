

# compute_rouge.py
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

def compute_rouge(df):
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    Each summary is scored against ALL its reference summaries;
    we take the MAX (as in the original paper).
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    
    r1_scores, r2_scores, rL_scores = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing ROUGE"):
        summary = row['summary']
        refs = row['references']
    
        # ---- FIX: it's already a list, but pd.read_csv stringifies it ----
        # Only parse if it came from CSV (string); skip if still a list
        if isinstance(refs, str):
            import ast
            refs = ast.literal_eval(refs)
    
        if not refs:
            r1_scores.append(np.nan)
            r2_scores.append(np.nan)
            rL_scores.append(np.nan)
            continue
    
        best_r1, best_r2, best_rL = 0, 0, 0
        for ref in refs:
            scores = scorer.score(ref, summary)
            best_r1 = max(best_r1, scores['rouge1'].fmeasure)
            best_r2 = max(best_r2, scores['rouge2'].fmeasure)
            best_rL = max(best_rL, scores['rougeL'].fmeasure)
    
        r1_scores.append(best_r1)
        r2_scores.append(best_r2)
        rL_scores.append(best_rL)
    
    df['rouge1'] = r1_scores
    df['rouge2'] = r2_scores
    df['rougeL'] = rL_scores
    
    print(f"\nROUGE score averages:")
    for col in ['rouge1', 'rouge2', 'rougeL']:
        print(f"  {col}: {df[col].mean():.4f}")
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/summeval_flat.csv")
    df = compute_rouge(df)
    df.to_csv("data/summeval_with_metrics.csv", index=False)
    print("Saved to data/summeval_with_metrics.csv")
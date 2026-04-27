

# correlation_analysis.py
"""
This is the KEY reproduction step.
We compute Spearman and Pearson correlations between each
automatic metric and each human evaluation dimension.
The paper's main claim: BERTScore correlates better than ROUGE.
"""
import pandas as pd
import numpy as np
from scipy import stats

def compute_correlations(df):
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    dimensions = ['human_coherence', 'human_consistency',
                  'human_fluency', 'human_relevance']
    
    # ---- Summary-level Spearman correlation ----
    print("=" * 60)
    print("SUMMARY-LEVEL SPEARMAN CORRELATIONS")
    print("=" * 60)
    
    spearman_results = {}
    for metric in metrics:
        spearman_results[metric] = {}
        for dim in dimensions:
            # Drop NaN pairs
            valid = df[[metric, dim]].dropna()
            r, p = stats.spearmanr(valid[metric], valid[dim])
            spearman_results[metric][dim] = {'r': r, 'p': p}
    
    # Pretty print table
    dim_labels = ['Coherence', 'Consistency', 'Fluency', 'Relevance']
    print(f"\n{'Metric':<20}", end="")
    for label in dim_labels:
        print(f"{label:>14}", end="")
    print()
    print("-" * 76)
    
    for metric in metrics:
        print(f"{metric:<20}", end="")
        for dim in dimensions:
            r = spearman_results[metric][dim]['r']
            print(f"{r:>14.4f}", end="")
        print()
    
    # ---- System-level correlations (average per model) ----
    print("\n" + "=" * 60)
    print("SYSTEM-LEVEL SPEARMAN CORRELATIONS (averaged per model)")
    print("=" * 60)
    
    system_df = df.groupby('model_id').mean(numeric_only=True).reset_index()
    
    sys_spearman = {}
    for metric in metrics:
        sys_spearman[metric] = {}
        for dim in dimensions:
            valid = system_df[[metric, dim]].dropna()
            if len(valid) < 3:
                sys_spearman[metric][dim] = {'r': np.nan, 'p': np.nan}
                continue
            r, p = stats.spearmanr(valid[metric], valid[dim])
            sys_spearman[metric][dim] = {'r': r, 'p': p}
    
    print(f"\n{'Metric':<20}", end="")
    for label in dim_labels:
        print(f"{label:>14}", end="")
    print()
    print("-" * 76)
    
    for metric in metrics:
        print(f"{metric:<20}", end="")
        for dim in dimensions:
            r = sys_spearman[metric][dim]['r']
            print(f"{r:>14.4f}", end="")
        print()
    
    return spearman_results, sys_spearman

def check_paper_claims(spearman_results):
    """
    The paper's core claims to verify:
    1. BERTScore correlates better with human judgments than ROUGE
    2. No single metric correlates well with Consistency
    3. System-level correlations are higher than summary-level
    """
    print("\n" + "=" * 60)
    print("VERIFYING PAPER CLAIMS")
    print("=" * 60)
    
    dims = ['human_coherence', 'human_consistency', 'human_fluency', 'human_relevance']
    
    for dim in dims:
        bert_r = spearman_results['bertscore_f1'][dim]['r']
        rouge1_r = spearman_results['rouge1'][dim]['r']
        rouge2_r = spearman_results['rouge2'][dim]['r']
        
        best_rouge = max(rouge1_r, rouge2_r, key=abs)
        best_rouge_name = 'rouge1' if abs(rouge1_r) >= abs(rouge2_r) else 'rouge2'
        
        dim_name = dim.replace('human_', '').capitalize()
        claim = "✓ CONFIRMED" if abs(bert_r) > abs(best_rouge) else "✗ NOT CONFIRMED"
        print(f"\n{dim_name}: BERTScore ({bert_r:.4f}) vs best ROUGE "
              f"({best_rouge_name}: {best_rouge:.4f}) → {claim}")

if __name__ == "__main__":
    df = pd.read_csv("data/summeval_with_metrics.csv")
    spearman_summary, spearman_system = compute_correlations(df)
    check_paper_claims(spearman_summary)
    
    
    

print("\n" + "=" * 60)
print("KEY FINDING: SYSTEM vs SUMMARY LEVEL COMPARISON")
print("=" * 60)
metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
dims    = ['human_coherence', 'human_consistency', 'human_fluency', 'human_relevance']

for metric in metrics:
    print(f"\n{metric}:")
    for dim in dims:
        s = spearman_summary[metric][dim]['r']
        sy = spearman_system[metric][dim]['r']
        direction = "↑ HIGHER" if sy > s else "↓ LOWER"
        print(f"  {dim:25s}  summary={s:.3f}  system={sy:.3f}  {direction}")
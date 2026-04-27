# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:18:05 2026

@author: Kali
"""

# visualize.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import ast

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']



def plot_correlation_heatmap(df, save_path="figures/correlation_heatmap.png"):
    import os; os.makedirs("figures", exist_ok=True)
    
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bertscore_f1']
    dimensions = ['human_coherence', 'human_consistency',
                  'human_fluency', 'human_relevance']
    dim_labels = ['Coherence', 'Consistency', 'Fluency', 'Relevance']
    metric_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']

    paper_values = {
        (0,0): 0.22, (0,1): 0.14, (0,2): 0.18, (0,3): 0.33,
        (1,0): 0.18, (1,1): 0.12, (1,2): 0.16, (1,3): 0.28,
        (3,0): 0.25, (3,1): 0.16, (3,2): 0.23, (3,3): 0.39,
    }

    corr_matrix = np.zeros((len(metrics), len(dimensions)))
    for i, metric in enumerate(metrics):
        for j, dim in enumerate(dimensions):
            valid = df[[metric, dim]].dropna()
            r, _ = stats.spearmanr(valid[metric], valid[dim])
            corr_matrix[i, j] = r

    fig, ax = plt.subplots(figsize=(9, 4))

    # ---- Draw heatmap FIRST ----
    sns.heatmap(
        corr_matrix,
        xticklabels=dim_labels,
        yticklabels=metric_labels,
        annot=True, fmt='.3f',
        cmap='RdYlGn', center=0,
        vmin=-0.3, vmax=0.7,
        ax=ax, linewidths=0.5
    )

    # ---- THEN update text labels with paper values ----
    for i in range(len(metrics)):
        for j in range(len(dimensions)):
            paper = paper_values.get((i, j), None)
            label = f"{corr_matrix[i, j]:.3f}"
            if paper:
                label += f"\n(paper: {paper:.2f})"
            ax.texts[i * len(dimensions) + j].set_text(label)

    ax.set_title('Spearman Correlation: Automatic Metrics vs Human Judgments\n'
                 '(Summary-Level)', fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")
    

def plot_score_distributions(df, save_path="figures/score_distributions.png"):
    """Distribution of human scores across all 4 dimensions."""
    import os; os.makedirs("figures", exist_ok=True)
    
    dims = ['human_coherence', 'human_consistency',
            'human_fluency', 'human_relevance']
    labels = ['Coherence', 'Consistency', 'Fluency', 'Relevance']
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, dim, label, color in zip(axes, dims, labels, COLORS):
        ax.hist(df[dim], bins=20, color=color, alpha=0.8, edgecolor='white')
        ax.axvline(df[dim].mean(), color='black', linestyle='--', linewidth=1.5,
                   label=f'Mean={df[dim].mean():.2f}')
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel('Score (1–5)')
        ax.set_ylabel('Count' if ax == axes[0] else '')
        ax.legend(fontsize=8)
    
    fig.suptitle('Distribution of Human Evaluation Scores', fontsize=13,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def plot_metric_vs_human(df, metric='bertscore_f1',
                         dim='human_relevance',
                         save_path="figures/scatter_bert_relevance.png"):
    """Scatter plot: one metric vs one human dimension."""
    import os; os.makedirs("figures", exist_ok=True)
    
    valid = df[[metric, dim, 'model_id']].dropna()
    r, p = stats.spearmanr(valid[metric], valid[dim])
    
    models = valid['model_id'].unique()
    palette = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, color in zip(models, palette):
        subset = valid[valid['model_id'] == model]
        ax.scatter(subset[metric], subset[dim], alpha=0.5,
                   color=color, s=20, label=model)
    
    ax.set_xlabel(metric.replace('_', ' ').upper(), fontsize=11)
    ax.set_ylabel(dim.replace('human_', '').capitalize() + ' (Human)', fontsize=11)
    ax.set_title(f'ρ = {r:.3f} (p={p:.3e})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, ncol=2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

def plot_system_level(df, save_path="figures/system_level_bars.png"):
    """System-level (per-model-average) scores."""
    import os; os.makedirs("figures", exist_ok=True)
    
    system = df.groupby('model_id')[
        ['human_coherence', 'human_consistency',
         'human_fluency', 'human_relevance']
    ].mean().reset_index()
    
    system = system.sort_values('human_relevance', ascending=False)
    
    x = np.arange(len(system))
    width = 0.2
    dims = ['human_coherence', 'human_consistency',
            'human_fluency', 'human_relevance']
    labels = ['Coherence', 'Consistency', 'Fluency', 'Relevance']
    
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (dim, label, color) in enumerate(zip(dims, labels, COLORS)):
        ax.bar(x + i * width, system[dim], width, label=label, color=color, alpha=0.85)
    
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(system['model_id'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Average Human Score (1–5)')
    ax.set_title('System-Level Human Evaluation Scores by Model',
                 fontweight='bold', fontsize=12)
    ax.legend()
    ax.set_ylim(1, 5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    df = pd.read_csv("data/summeval_with_metrics.csv")
    plot_correlation_heatmap(df)
    plot_score_distributions(df)
    plot_metric_vs_human(df)
    plot_system_level(df)
    print("\nAll figures saved to figures/")
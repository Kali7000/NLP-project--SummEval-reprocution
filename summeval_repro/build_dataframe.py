

# build_dataframe.py
import json
import numpy as np
import pandas as pd

def load_summeval(path="data/model_annotations.aligned.jsonl"):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records

def build_dataframe(records):
    rows = []
    for r in records:
        expert_anns = r.get('expert_annotations', [])
        if not expert_anns:
            continue

        coherence   = np.mean([a['coherence']   for a in expert_anns])
        consistency = np.mean([a['consistency'] for a in expert_anns])
        fluency     = np.mean([a['fluency']     for a in expert_anns])
        relevance   = np.mean([a['relevance']   for a in expert_anns])

        rows.append({
            'id':                r['id'],
            'model_id':          r['model_id'],
            'summary':           r['decoded'],        # the generated summary
            'references':        r['references'],     # list of human reference summaries
            'human_coherence':   coherence,
            'human_consistency': consistency,
            'human_fluency':     fluency,
            'human_relevance':   relevance,
        })

    df = pd.DataFrame(rows)
    print(f"DataFrame shape: {df.shape}")
    print(f"\nModels found: {sorted(df['model_id'].unique())}")
    print(f"\nHuman score distributions:")
    for dim in ['human_coherence', 'human_consistency', 'human_fluency', 'human_relevance']:
        print(f"  {dim}: mean={df[dim].mean():.2f}, std={df[dim].std():.2f}")
    return df

if __name__ == "__main__":
    records = load_summeval()
    df = build_dataframe(records)
    df.to_csv("data/summeval_flat.csv", index=False)
    print("\nSaved to data/summeval_flat.csv")
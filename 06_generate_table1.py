"""
Generate Table 1: Characteristics of the study test sets.
"""
import json
import os
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_DIR = Path(os.environ.get(
    'RLM_DATA_DIR',
    str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
))
TABLE_DIR = Path(__file__).parent / 'output' / 'tables'
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def analyze_dataset(cases, name):
    """Compute descriptive statistics for a test set."""
    n = len(cases)
    det_truth = [c.get('detection_truth', 0) for c in cases]
    n_hazard = sum(det_truth)
    n_benign = n - n_hazard

    # Action distribution.
    actions = [c.get('action_truth', 'None') for c in cases]
    action_counts = Counter(actions)

    # Hazard categories.
    categories = [c.get('hazard_category', c.get('hazard_type', 'unknown')) for c in cases]
    cat_counts = Counter(categories)

    # Message length statistics.
    msgs = [c.get('prompt', c.get('message', '')) for c in cases]
    msg_lens = [len(m) for m in msgs]
    msg_words = [len(m.split()) for m in msgs]

    return {
        'Dataset': name,
        'N': n,
        'Hazard cases, n (%)': f"{n_hazard} ({100*n_hazard/n:.1f}%)",
        'Benign cases, n (%)': f"{n_benign} ({100*n_benign/n:.1f}%)",
        'Action: None, n (%)': f"{action_counts.get('None', 0)} ({100*action_counts.get('None', 0)/n:.1f}%)",
        'Action: Routine Follow-up, n (%)': f"{action_counts.get('Routine Follow-up', 0)} ({100*action_counts.get('Routine Follow-up', 0)/n:.1f}%)",
        'Action: Contact Doctor, n (%)': f"{action_counts.get('Contact Doctor', 0)} ({100*action_counts.get('Contact Doctor', 0)/n:.1f}%)",
        'Action: Call 911/988, n (%)': f"{action_counts.get('Call 911/988', 0)} ({100*action_counts.get('Call 911/988', 0)/n:.1f}%)",
        'Message length, characters, median (IQR)': f"{pd.Series(msg_lens).median():.0f} ({pd.Series(msg_lens).quantile(0.25):.0f}-{pd.Series(msg_lens).quantile(0.75):.0f})",
        'Message length, words, median (IQR)': f"{pd.Series(msg_words).median():.0f} ({pd.Series(msg_words).quantile(0.25):.0f}-{pd.Series(msg_words).quantile(0.75):.0f})",
        'Unique hazard categories': len([c for c, v in cat_counts.items() if c.lower() not in ('benign', 'unknown', 'harm_candidate', 'hazard_candidate')]),
    }


def main():
    # Load datasets.
    with open(DATA_DIR / 'physician_full.json') as f:
        phys = json.load(f)
    with open(DATA_DIR / 'realworld_full.json') as f:
        rw = json.load(f)

    stats_phys = analyze_dataset(phys, 'Physician-created')
    stats_rw = analyze_dataset(rw, 'Real-world validation')

    df = pd.DataFrame([stats_phys, stats_rw]).T
    df.columns = ['Physician-Created (N=' + str(len(phys)) + ')',
                   'Real-World Validation (N=' + str(len(rw)) + ')']
    df.index.name = 'Characteristic'

    df.to_csv(TABLE_DIR / 'table1_characteristics.csv')
    print(df.to_string())

    # Also generate hazard category breakdown for physician set.
    categories = Counter([c.get('hazard_category', c.get('hazard_type', 'unknown'))
                         for c in phys if c.get('detection_truth', 0) == 1])
    cat_df = pd.DataFrame(
        [(k, v, f"{100*v/sum(categories.values()):.1f}%")
         for k, v in categories.most_common()],
        columns=['Hazard Category', 'N', '%']
    )
    cat_df.to_csv(TABLE_DIR / 'etable_hazard_categories.csv', index=False)
    print("\nHazard categories (physician set):")
    print(cat_df.to_string())


if __name__ == '__main__':
    main()

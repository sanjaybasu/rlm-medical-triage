"""
CFS threshold sensitivity analysis.
Computes PSR and CFS at thresholds 0.5, 0.6, 0.7, 0.8, 0.9.
"""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils import compute_phantom_symptom_rate

RAW_DIR = Path(__file__).parent / 'output' / 'raw'
DATA_DIR = Path(os.environ.get(
    'RLM_DATA_DIR',
    str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
))
METRICS_DIR = Path(__file__).parent / 'output' / 'metrics'
METRICS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def load_cases(ds_name, filename_stem=''):
    is_rlm = filename_stem.startswith(('C_', 'D_'))
    if ds_name == 'realworld' and is_rlm:
        path = DATA_DIR / 'realworld_rlm_subsample.json'
    elif ds_name == 'realworld':
        path = DATA_DIR / 'realworld_full.json'
    else:
        path = DATA_DIR / 'physician_full.json'
    with open(path) as f:
        return json.load(f)


def main():
    rows = []
    for jf in sorted(RAW_DIR.glob('*.jsonl')):
        ds_name = 'physician' if 'physician' in jf.stem else 'realworld'
        cases = load_cases(ds_name, jf.stem)

        results_by_idx = {}
        with open(jf) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    results_by_idx[r['case_idx']] = r
                except Exception:
                    pass

        for thresh in THRESHOLDS:
            psr_vals = []
            cfs_vals = []
            for i, case in enumerate(cases):
                result = results_by_idx.get(i)
                if result is None:
                    continue
                parsed = result.get('parsed')
                if parsed is None:
                    continue
                evidence = parsed.get('evidence', [])
                if not isinstance(evidence, list) or len(evidence) == 0:
                    continue
                message = case.get('prompt', case.get('message', ''))
                psr, cfs, n_claims = compute_phantom_symptom_rate(evidence, message, threshold=thresh)
                if n_claims > 0:
                    psr_vals.append(psr)
                    cfs_vals.append(cfs)

            if psr_vals:
                rows.append({
                    'file': jf.stem,
                    'threshold': thresh,
                    'n_cases': len(psr_vals),
                    'mean_psr': round(np.mean(psr_vals), 4),
                    'mean_cfs': round(np.mean(cfs_vals), 4),
                })

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / 'cfs_threshold_sensitivity.csv', index=False)
    print("CFS threshold sensitivity analysis:")
    for f in df['file'].unique():
        sub = df[df['file'] == f]
        print(f"\n  {f}:")
        for _, row in sub.iterrows():
            print(f"    thresh={row['threshold']}: PSR={row['mean_psr']:.4f}, CFS={row['mean_cfs']:.4f}")


if __name__ == '__main__':
    main()

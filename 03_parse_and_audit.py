"""
Parse all JSONL outputs and compute hallucination audit metrics:
  - Phantom Symptom Rate (PSR)
  - Citation Fidelity Score (CFS)
  - Parse failure rate
Saves per-case audit as CSV.
"""
import json
import os
import pandas as pd
from pathlib import Path

from utils import parse_json_response, extract_detection_action, compute_phantom_symptom_rate

DATA_DIR = Path(os.environ.get(
    'RLM_DATA_DIR',
    str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
))
RAW_DIR = Path(__file__).parent / 'output' / 'raw'
PARSED_DIR = Path(__file__).parent / 'output' / 'parsed'
PARSED_DIR.mkdir(parents=True, exist_ok=True)

TEST_SETS = {
    'physician': DATA_DIR / 'physician_full.json',
    'realworld': DATA_DIR / 'realworld_full.json',
    'realworld_rlm': DATA_DIR / 'realworld_rlm_subsample.json',
}


def load_cases(ds_name, filename_stem=''):
    """Load cases, using subsample for RLM arms on realworld data."""
    is_rlm = filename_stem.startswith(('C_', 'D_'))
    if ds_name == 'realworld' and is_rlm:
        with open(TEST_SETS['realworld_rlm']) as f:
            return json.load(f)
    with open(TEST_SETS[ds_name]) as f:
        return json.load(f)


def process_jsonl(jsonl_path, cases):
    """Parse a JSONL file and compute per-case audit metrics."""
    rows = []
    parse_failures = 0
    total = 0

    results_by_idx = {}
    with open(jsonl_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                results_by_idx[row['case_idx']] = row
            except Exception:
                pass

    for i, case in enumerate(cases):
        result = results_by_idx.get(i)
        if result is None:
            # Case not yet run (incomplete experiment) — skip entirely
            continue
        total += 1

        raw = result.get('raw_response', '')
        parsed = result.get('parsed')
        if parsed is None:
            parsed = parse_json_response(raw)

        if parsed is None:
            parse_failures += 1
            rows.append(_empty_row(i, case))
            continue

        det, act = extract_detection_action(parsed)
        evidence = parsed.get('evidence', [])
        if not isinstance(evidence, list):
            evidence = []

        message = case.get('prompt', case.get('message', ''))
        psr, cfs, n_claims = compute_phantom_symptom_rate(evidence, message)

        # PSR and CFS are only meaningful when the model produced evidence claims.
        # Set to NaN when n_claims == 0 so they're excluded from averages.
        rows.append({
            'case_idx': i,
            'case_name': case.get('name', case.get('case_label', f'case_{i}')),
            'detection_pred': det,
            'action_pred': act,
            'detection_truth': case.get('detection_truth', 0),
            'action_truth': case.get('action_truth', 'None'),
            'n_evidence_claims': n_claims,
            'phantom_symptom_rate': round(psr, 4) if n_claims > 0 else None,
            'citation_fidelity_score': round(cfs, 4) if n_claims > 0 else None,
            'elapsed_sec': result.get('elapsed_sec', None),
            'parse_success': True,
        })

    return pd.DataFrame(rows), parse_failures, total


def _empty_row(i, case):
    return {
        'case_idx': i,
        'case_name': case.get('name', case.get('case_label', f'case_{i}')),
        'detection_pred': 0,
        'action_pred': 0,
        'detection_truth': case.get('detection_truth', 0),
        'action_truth': case.get('action_truth', 'None'),
        'n_evidence_claims': 0,
        'phantom_symptom_rate': None,
        'citation_fidelity_score': None,
        'elapsed_sec': None,
        'parse_success': False,
    }


def main():
    summary_rows = []
    jsonl_files = sorted(RAW_DIR.glob('*.jsonl'))
    print(f"Found {len(jsonl_files)} JSONL files in {RAW_DIR}")

    for jf in jsonl_files:
        # Determine dataset and arm from filename.
        ds_name = 'physician' if 'physician' in jf.stem else 'realworld'
        arm_name = jf.stem.split('_')[0] + '_' if jf.stem[0] in 'ABCD' else ''
        cases = load_cases(ds_name, jf.stem)

        df, n_fail, n_total = process_jsonl(jf, cases)
        out_csv = PARSED_DIR / f'{jf.stem}_audit.csv'
        df.to_csv(out_csv, index=False)

        psr_mean = df['phantom_symptom_rate'].dropna().mean()
        cfs_mean = df['citation_fidelity_score'].dropna().mean()

        summary_rows.append({
            'file': jf.stem,
            'dataset': ds_name,
            'n_cases': n_total,
            'parse_failures': n_fail,
            'parse_failure_rate': round(n_fail / max(n_total, 1), 4),
            'mean_psr': round(psr_mean, 4) if pd.notna(psr_mean) else None,
            'mean_cfs': round(cfs_mean, 4) if pd.notna(cfs_mean) else None,
        })
        print(f"  {jf.stem}: {n_total} cases, {n_fail} parse failures, "
              f"PSR={psr_mean:.3f}, CFS={cfs_mean:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(PARSED_DIR / 'parse_summary.csv', index=False)
    print(f"\nSummary saved to {PARSED_DIR / 'parse_summary.csv'}")


if __name__ == '__main__':
    main()

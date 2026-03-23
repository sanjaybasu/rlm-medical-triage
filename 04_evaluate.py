"""
Compute detection and action metrics for all conditions.
Outputs: metrics CSV with 95% CIs for all arms x models x datasets.

Addresses peer review requirements:
  - Parse failure sensitivity analysis (with/without parse failures)
  - Action confusion matrices
  - Per-hazard-category sensitivity (physician set)
  - Evidence claim counts per arm
"""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score

from utils import wilson_ci, bootstrap_ci, ACTION_MAP, ACTION_LABELS

PARSED_DIR = Path(__file__).parent / 'output' / 'parsed'
METRICS_DIR = Path(__file__).parent / 'output' / 'metrics'
DATA_DIR = Path(os.environ.get(
    'RLM_DATA_DIR',
    str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
))
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(df, label=''):
    """Compute all metrics from a parsed audit DataFrame."""
    y_det_true = df['detection_truth'].values.astype(int)
    y_det_pred = df['detection_pred'].values.astype(int)
    y_act_true = df['action_truth'].map(ACTION_MAP).fillna(0).values.astype(int)
    y_act_pred = df['action_pred'].values.astype(int)

    # Detection metrics.
    try:
        tn, fp, fn, tp = confusion_matrix(y_det_true, y_det_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn, fp, fn, tp = 0, 0, 0, 0

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)

    try:
        mcc = matthews_corrcoef(y_det_true, y_det_pred)
    except Exception:
        mcc = 0.0
    mcc_ci = bootstrap_ci(y_det_true, y_det_pred, matthews_corrcoef)

    # Action metrics.
    act_acc = accuracy_score(y_act_true, y_act_pred)
    act_ci = bootstrap_ci(y_act_true, y_act_pred, accuracy_score)

    # Action confusion matrix.
    act_labels = [0, 1, 2, 3]
    try:
        act_cm = confusion_matrix(y_act_true, y_act_pred, labels=act_labels)
    except Exception:
        act_cm = np.zeros((4, 4), dtype=int)

    # Critical under-triage: among urgent/emergent cases, fraction assigned lower action.
    critical_mask = y_act_true >= 2
    n_critical = critical_mask.sum()
    if n_critical > 0:
        under_triage = ((y_act_pred[critical_mask] < y_act_true[critical_mask]).sum())
        cut_rate = under_triage / n_critical
        cut_ci = wilson_ci(int(under_triage), int(n_critical))
    else:
        cut_rate = 0.0
        cut_ci = (0.0, 0.0)

    # Hallucination metrics.
    psr_vals = df['phantom_symptom_rate'].dropna().values
    cfs_vals = df['citation_fidelity_score'].dropna().values
    n_claims = int(df['n_evidence_claims'].sum())
    n_cases_with_claims = int((df['n_evidence_claims'] > 0).sum())
    mean_claims = df['n_evidence_claims'].mean()
    psr_mean = psr_vals.mean() if len(psr_vals) > 0 else np.nan
    cfs_mean = cfs_vals.mean() if len(cfs_vals) > 0 else np.nan

    # Bootstrap CIs for PSR and CFS.
    if len(psr_vals) > 1:
        psr_boot = np.percentile(
            [np.random.RandomState(s).choice(psr_vals, len(psr_vals), replace=True).mean()
             for s in range(10000)],
            [2.5, 97.5]
        )
        cfs_boot = np.percentile(
            [np.random.RandomState(s).choice(cfs_vals, len(cfs_vals), replace=True).mean()
             for s in range(10000)],
            [2.5, 97.5]
        )
    else:
        psr_boot = (np.nan, np.nan)
        cfs_boot = (np.nan, np.nan)

    # Parse success rate.
    parse_rate = df['parse_success'].mean()
    n_parse_success = int(df['parse_success'].sum())
    n_parse_fail = len(df) - n_parse_success

    # Latency.
    latency_vals = df['elapsed_sec'].dropna().values
    latency_median = np.median(latency_vals) if len(latency_vals) > 0 else np.nan
    latency_p95 = np.percentile(latency_vals, 95) if len(latency_vals) > 0 else np.nan

    return {
        'n_cases': len(df),
        'n_parse_success': n_parse_success,
        'n_parse_fail': n_parse_fail,
        'sensitivity': round(sens, 4),
        'sensitivity_ci_lo': round(sens_ci[0], 4),
        'sensitivity_ci_hi': round(sens_ci[1], 4),
        'specificity': round(spec, 4),
        'specificity_ci_lo': round(spec_ci[0], 4),
        'specificity_ci_hi': round(spec_ci[1], 4),
        'mcc': round(mcc, 4),
        'mcc_ci_lo': round(mcc_ci[0], 4),
        'mcc_ci_hi': round(mcc_ci[1], 4),
        'action_accuracy': round(act_acc, 4),
        'action_accuracy_ci_lo': round(act_ci[0], 4),
        'action_accuracy_ci_hi': round(act_ci[1], 4),
        'critical_under_triage': round(cut_rate, 4),
        'cut_ci_lo': round(cut_ci[0], 4),
        'cut_ci_hi': round(cut_ci[1], 4),
        'phantom_symptom_rate': round(psr_mean, 4) if not np.isnan(psr_mean) else None,
        'psr_ci_lo': round(psr_boot[0], 4) if not np.isnan(psr_boot[0]) else None,
        'psr_ci_hi': round(psr_boot[1], 4) if not np.isnan(psr_boot[1]) else None,
        'citation_fidelity': round(cfs_mean, 4) if not np.isnan(cfs_mean) else None,
        'cfs_ci_lo': round(cfs_boot[0], 4) if not np.isnan(cfs_boot[0]) else None,
        'cfs_ci_hi': round(cfs_boot[1], 4) if not np.isnan(cfs_boot[1]) else None,
        'total_evidence_claims': n_claims,
        'n_cases_with_claims': n_cases_with_claims,
        'mean_claims_per_case': round(mean_claims, 2),
        'parse_success_rate': round(parse_rate, 4),
        'latency_median_sec': round(latency_median, 2) if not np.isnan(latency_median) else None,
        'latency_p95_sec': round(latency_p95, 2) if not np.isnan(latency_p95) else None,
        'action_confusion_matrix': act_cm.tolist(),
    }


def compute_category_sensitivity(df, cases, arm_str, model_str):
    """Compute per-hazard-category sensitivity (physician set only)."""
    rows = []
    for i, case in enumerate(cases):
        cat = case.get('hazard_category', case.get('hazard_type', 'unknown'))
        det_truth = case.get('detection_truth', 0)
        if det_truth == 0:
            continue
        audit_row = df[df['case_idx'] == i]
        if len(audit_row) == 0:
            det_pred = 0
        else:
            det_pred = int(audit_row.iloc[0]['detection_pred'])
        rows.append({'category': cat, 'truth': det_truth, 'pred': det_pred})

    if not rows:
        return pd.DataFrame()

    cat_df = pd.DataFrame(rows)
    result = []
    for cat, grp in cat_df.groupby('category'):
        n = len(grp)
        tp = (grp['pred'] == 1).sum()
        sens = tp / n if n > 0 else 0.0
        ci = wilson_ci(int(tp), int(n))
        result.append({
            'category': cat,
            'n': n,
            'detected': int(tp),
            'sensitivity': round(sens, 4),
            'sens_ci_lo': round(ci[0], 4),
            'sens_ci_hi': round(ci[1], 4),
            'arm': arm_str,
            'model': model_str,
        })

    return pd.DataFrame(result)


def _is_rlm_arm(arm_str):
    """Check if arm uses REPL (can have parse failures)."""
    return arm_str.startswith(('C_', 'D_', 'Cp_', 'Dp_'))


def main():
    audit_files = sorted(PARSED_DIR.glob('*_audit.csv'))
    print(f"Found {len(audit_files)} audit CSVs")

    # Load physician cases for category analysis.
    phys_cases = None
    phys_path = DATA_DIR / 'physician_full.json'
    if phys_path.exists():
        with open(phys_path) as f:
            phys_cases = json.load(f)

    all_metrics = []
    all_category_sens = []
    all_action_cms = []
    parse_failure_summary = []

    for af in audit_files:
        df = pd.read_csv(af)
        stem = af.stem.replace('_audit', '')
        ds = stem.split('_')[-1]  # physician or realworld

        # Find model and arm.
        model_str = None
        arm_str = None
        for mname in ['deepseek-r1_70b', 'qwen3_32b', 'qwen3_8b', 'llama3.1_8b']:
            if mname in stem:
                model_str = mname
                arm_str = stem.split(f'_{mname}')[0]
                break
        if model_str is None:
            model_str = 'unknown'
            arm_str = stem

        # ITT analysis (all cases, parse failures count as detection=0).
        metrics = compute_metrics(df, f'{arm_str}/{model_str}/{ds}')
        metrics['arm'] = arm_str
        metrics['model'] = model_str
        metrics['dataset'] = ds
        metrics['analysis_type'] = 'itt'

        # Save action confusion matrix separately.
        act_cm = metrics.pop('action_confusion_matrix')
        all_action_cms.append({
            'arm': arm_str, 'model': model_str, 'dataset': ds,
            'analysis_type': 'itt', 'cm': act_cm,
        })

        all_metrics.append(metrics)

        print(f"  {arm_str}/{model_str}/{ds}: sens={metrics['sensitivity']:.3f}, "
              f"spec={metrics['specificity']:.3f}, PSR={metrics.get('phantom_symptom_rate', 'N/A')}, "
              f"parse_fail={metrics['n_parse_fail']}")

        # Per-protocol analysis for RLM arms (exclude parse failures).
        if _is_rlm_arm(arm_str):
            n_fail = metrics['n_parse_fail']
            parse_failure_summary.append({
                'arm': arm_str, 'model': model_str, 'dataset': ds,
                'n_cases': len(df), 'n_parse_success': metrics['n_parse_success'],
                'n_parse_fail': n_fail,
                'parse_failure_rate': round(n_fail / len(df), 4) if len(df) > 0 else 0.0,
            })

            df_success = df[df['parse_success'] == True].copy()
            if len(df_success) > 0 and len(df_success) < len(df):
                pp_metrics = compute_metrics(df_success, f'{arm_str}/{model_str}/{ds} (per-protocol)')
                pp_metrics['arm'] = arm_str
                pp_metrics['model'] = model_str
                pp_metrics['dataset'] = ds
                pp_metrics['analysis_type'] = 'per_protocol'

                pp_cm = pp_metrics.pop('action_confusion_matrix')
                all_action_cms.append({
                    'arm': arm_str, 'model': model_str, 'dataset': ds,
                    'analysis_type': 'per_protocol', 'cm': pp_cm,
                })

                all_metrics.append(pp_metrics)
                print(f"    (per-protocol, n={len(df_success)}): sens={pp_metrics['sensitivity']:.3f}, "
                      f"spec={pp_metrics['specificity']:.3f}")

        # Per-category sensitivity (physician set only).
        if ds == 'physician' and phys_cases is not None:
            cat_df = compute_category_sensitivity(df, phys_cases, arm_str, model_str)
            if len(cat_df) > 0:
                all_category_sens.append(cat_df)

    # Save primary metrics.
    metrics_df = pd.DataFrame(all_metrics)
    col_order = ['arm', 'model', 'dataset', 'analysis_type', 'n_cases',
                 'n_parse_success', 'n_parse_fail',
                 'sensitivity', 'sensitivity_ci_lo', 'sensitivity_ci_hi',
                 'specificity', 'specificity_ci_lo', 'specificity_ci_hi',
                 'mcc', 'mcc_ci_lo', 'mcc_ci_hi',
                 'action_accuracy', 'action_accuracy_ci_lo', 'action_accuracy_ci_hi',
                 'critical_under_triage', 'cut_ci_lo', 'cut_ci_hi',
                 'phantom_symptom_rate', 'psr_ci_lo', 'psr_ci_hi',
                 'citation_fidelity', 'cfs_ci_lo', 'cfs_ci_hi',
                 'total_evidence_claims', 'n_cases_with_claims', 'mean_claims_per_case',
                 'parse_success_rate',
                 'latency_median_sec', 'latency_p95_sec']
    metrics_df = metrics_df[[c for c in col_order if c in metrics_df.columns]]
    metrics_df.to_csv(METRICS_DIR / 'all_metrics.csv', index=False)
    print(f"\nMetrics saved to {METRICS_DIR / 'all_metrics.csv'}")

    # Save parse failure summary.
    if parse_failure_summary:
        pf_df = pd.DataFrame(parse_failure_summary)
        pf_df.to_csv(METRICS_DIR / 'parse_failure_summary.csv', index=False)
        print(f"Parse failure summary saved ({len(pf_df)} RLM conditions)")

    # Save per-category sensitivity.
    if all_category_sens:
        cat_all = pd.concat(all_category_sens, ignore_index=True)
        cat_all.to_csv(METRICS_DIR / 'category_sensitivity.csv', index=False)
        print(f"Category sensitivity saved ({len(cat_all)} rows)")

    # Save action confusion matrices.
    act_cm_rows = []
    for item in all_action_cms:
        cm = np.array(item['cm'])
        for i in range(4):
            for j in range(4):
                act_cm_rows.append({
                    'arm': item['arm'], 'model': item['model'], 'dataset': item['dataset'],
                    'true_action': ACTION_LABELS[i], 'pred_action': ACTION_LABELS[j],
                    'count': int(cm[i, j]),
                })
    act_cm_df = pd.DataFrame(act_cm_rows)
    act_cm_df.to_csv(METRICS_DIR / 'action_confusion_matrices.csv', index=False)
    print(f"Action confusion matrices saved")


if __name__ == '__main__':
    main()

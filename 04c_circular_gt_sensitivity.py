"""Sensitivity analysis for circular ground truth in real-world validation set.

The real-world set uses outcome-based ground truth (provider-documented escalation),
which is circular: hazards not escalated by the provider are labeled as benign.
This script estimates the impact by perturbing ground truth labels under alternative
assumptions about the false negative and false positive rates of provider labeling.

Scenarios:
  1. Base: current outcome-based labels as-is
  2. Conservative: assume 20% of escalation labels are false positives
  3. Liberal: assume 10% of "None" labels are false negatives (missed hazards)
  4. Combined: both perturbations simultaneously

For each scenario, performs 1000 Monte Carlo draws, recomputes sensitivity/specificity/MCC,
and reports 95% CIs across draws.

Output: output/metrics/gt_sensitivity_analysis.csv
"""
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(os.environ.get(
    'RLM_DATA_DIR',
    str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
))
PARSED_DIR = Path(__file__).parent / 'output' / 'parsed'
OUTPUT_DIR = Path(__file__).parent / 'output' / 'metrics'

N_DRAWS = 1000
SEED = 42


def compute_metrics(y_true, y_pred):
    """Compute sensitivity, specificity, MCC from binary arrays."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {"sensitivity": sens, "specificity": spec, "mcc": mcc,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def perturb_ground_truth(y_true, rng, fp_rate=0.0, fn_rate=0.0):
    """Perturb ground truth labels.

    fp_rate: probability that a positive label is actually a false positive
    fn_rate: probability that a negative label is actually a false negative
    """
    y_perturbed = y_true.copy()
    for i in range(len(y_perturbed)):
        if y_perturbed[i] == 1 and fp_rate > 0:
            if rng.random() < fp_rate:
                y_perturbed[i] = 0
        elif y_perturbed[i] == 0 and fn_rate > 0:
            if rng.random() < fn_rate:
                y_perturbed[i] = 1
    return y_perturbed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Load all real-world audit CSVs
    audit_files = sorted(PARSED_DIR.glob("*_realworld_audit.csv"))
    if not audit_files:
        print("No real-world audit files found. Run 03_parse_and_audit.py first.")
        return

    scenarios = {
        "base": {"fp_rate": 0.0, "fn_rate": 0.0},
        "conservative_fp20": {"fp_rate": 0.20, "fn_rate": 0.0},
        "liberal_fn10": {"fp_rate": 0.0, "fn_rate": 0.10},
        "combined_fp20_fn10": {"fp_rate": 0.20, "fn_rate": 0.10},
    }

    all_results = []

    for audit_file in audit_files:
        stem = audit_file.stem.replace("_realworld_audit", "")
        parts = stem.split("_", maxsplit=1)
        if len(parts) < 2:
            continue

        df = pd.read_csv(audit_file)
        if "detection_truth" not in df.columns or "detection_pred" not in df.columns:
            print(f"  Skipping {stem}: missing columns")
            continue

        y_true = df["detection_truth"].values.astype(int)
        y_pred = df["detection_pred"].values.astype(int)

        for scenario_name, params in scenarios.items():
            if scenario_name == "base":
                # No perturbation, compute once
                metrics = compute_metrics(y_true, y_pred)
                metrics["scenario"] = scenario_name
                metrics["condition"] = stem
                metrics["n"] = len(y_true)
                metrics["prevalence"] = float(np.mean(y_true))
                all_results.append(metrics)
            else:
                # Monte Carlo draws
                draw_metrics = defaultdict(list)
                for _ in range(N_DRAWS):
                    y_perturbed = perturb_ground_truth(
                        y_true, rng,
                        fp_rate=params["fp_rate"],
                        fn_rate=params["fn_rate"]
                    )
                    m = compute_metrics(y_perturbed, y_pred)
                    for k, v in m.items():
                        if isinstance(v, (int, float)):
                            draw_metrics[k].append(v)

                # Summarize with 95% CIs
                metrics = {}
                for k, vals in draw_metrics.items():
                    vals = np.array(vals)
                    metrics[k] = float(np.mean(vals))
                    metrics[f"{k}_lo"] = float(np.percentile(vals, 2.5))
                    metrics[f"{k}_hi"] = float(np.percentile(vals, 97.5))

                metrics["scenario"] = scenario_name
                metrics["condition"] = stem
                metrics["n"] = len(y_true)
                metrics["prevalence_mean"] = float(np.mean([np.mean(
                    perturb_ground_truth(y_true, rng, params["fp_rate"], params["fn_rate"])
                ) for _ in range(100)]))
                all_results.append(metrics)

        print(f"  {stem}: {len(y_true)} cases, {np.mean(y_true):.1%} prevalence")

    # Save results
    results_df = pd.DataFrame(all_results)
    out_path = OUTPUT_DIR / "gt_sensitivity_analysis.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df)} rows to {out_path}")

    # Print summary
    print("\n=== Ground Truth Sensitivity Analysis Summary ===")
    for scenario in scenarios:
        sub = results_df[results_df["scenario"] == scenario]
        if sub.empty:
            continue
        print(f"\n{scenario}:")
        for _, row in sub.iterrows():
            sens = row.get("sensitivity", 0)
            if f"sensitivity_lo" in row:
                print(f"  {row['condition']}: sens={sens:.3f} [{row['sensitivity_lo']:.3f}-{row['sensitivity_hi']:.3f}]")
            else:
                print(f"  {row['condition']}: sens={sens:.3f}")


if __name__ == "__main__":
    main()

"""
McNemar's test for DeepSeek-R1-70B chain-of-thought vs single-pass at full N=400.
Reads per-case audit CSVs from output/parsed/.
Outputs: output/metrics/deepseek_mcnemar_n400.csv and prints result.

McNemar's test with continuity correction on paired binary detection outcomes.
Paired on case_idx (matched cases present in both conditions).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

PARSED_DIR = os.path.join(os.path.dirname(__file__), "output", "parsed")
OUT_PATH = os.path.join(os.path.dirname(__file__), "output", "metrics", "deepseek_mcnemar_n400.csv")

A_FILE = os.path.join(PARSED_DIR, "A_single_pass_deepseek-r1_70b_physician_audit.csv")
B_FILE = os.path.join(PARSED_DIR, "B_chain_of_thought_deepseek-r1_70b_physician_audit.csv")


def mcnemar_continuity(b, c):
    """McNemar's test with continuity correction. b = A+B-, c = A-B+."""
    statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    p = 1 - chi2.cdf(statistic, df=1)
    return statistic, p


def main():
    df_a = pd.read_csv(A_FILE)
    df_b = pd.read_csv(B_FILE)

    df_a = df_a[df_a["parse_success"] == True][["case_idx", "detection_pred", "detection_truth"]].copy()
    df_b = df_b[df_b["parse_success"] == True][["case_idx", "detection_pred"]].copy()
    df_b = df_b.rename(columns={"detection_pred": "detection_pred_b"})

    paired = df_a.merge(df_b, on="case_idx", how="inner")
    n_paired = len(paired)
    print(f"Paired cases (present in both A and B): {n_paired}")

    # restrict to hazard cases only (McNemar on sensitivity)
    hazard = paired[paired["detection_truth"] == 1].copy()
    n_hazard = len(hazard)
    print(f"Hazard cases in paired set: {n_hazard}")

    # confusion cells
    # A+B- : A detected (pred=1), B missed (pred=0)
    ab_pos_neg = ((hazard["detection_pred"] == 1) & (hazard["detection_pred_b"] == 0)).sum()
    # A-B+ : A missed, B detected
    ab_neg_pos = ((hazard["detection_pred"] == 0) & (hazard["detection_pred_b"] == 1)).sum()
    # concordant correct
    ab_pos_pos = ((hazard["detection_pred"] == 1) & (hazard["detection_pred_b"] == 1)).sum()
    ab_neg_neg = ((hazard["detection_pred"] == 0) & (hazard["detection_pred_b"] == 0)).sum()

    print(f"\nMcNemar table (hazard cases only):")
    print(f"  A+B+ (both detect):   {ab_pos_pos}")
    print(f"  A+B- (A only):        {ab_pos_neg}")
    print(f"  A-B+ (B only):        {ab_neg_pos}")
    print(f"  A-B- (both miss):     {ab_neg_neg}")

    stat, p = mcnemar_continuity(ab_pos_neg, ab_neg_pos)

    sens_a = hazard["detection_pred"].mean()
    sens_b = hazard["detection_pred_b"].mean()
    diff = sens_b - sens_a

    print(f"\nSensitivity:")
    print(f"  Single-pass (A): {sens_a*100:.1f}%")
    print(f"  Chain-of-thought (B): {sens_b*100:.1f}%")
    print(f"  Difference (B-A): {diff*100:+.1f} pp")
    print(f"\nMcNemar chi-squared (continuity corrected): {stat:.3f}")
    print(f"p-value: {p:.4f}")
    print(f"Significant at Bonferroni alpha=0.0021: {'YES' if p < 0.0021 else 'NO'}")
    print(f"Significant at alpha=0.05: {'YES' if p < 0.05 else 'NO'}")

    result = pd.DataFrame([{
        "model": "DeepSeek-R1-70B",
        "comparison": "B_vs_A",
        "dataset": "physician",
        "n_paired": n_paired,
        "n_hazard_paired": n_hazard,
        "sens_a": round(sens_a * 100, 1),
        "sens_b": round(sens_b * 100, 1),
        "diff_pp": round(diff * 100, 1),
        "mcnemar_ab_pos_neg": ab_pos_neg,
        "mcnemar_ab_neg_pos": ab_neg_pos,
        "chi2_stat": round(stat, 3),
        "p_value": round(p, 4),
        "sig_bonferroni": p < 0.0021,
        "sig_nominal": p < 0.05,
    }])
    result.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

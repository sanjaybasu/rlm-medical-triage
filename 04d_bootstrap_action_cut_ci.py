"""
Bootstrap 95% CIs for action accuracy and critical under-triage rate (CUT).
Reads per-case audit CSVs from output/parsed/.
Outputs: output/metrics/action_cut_bootstrap_ci.csv

Action accuracy: proportion of cases where action_pred matches action_truth.
CUT: proportion of urgent/emergent cases (action_truth in CUT_URGENT) assigned
     insufficient escalation (action_pred not in CUT_URGENT).

Action encoding (from 03_parse_and_audit.py):
  0 = Self-Care
  1 = Contact Doctor
  2 = Routine Follow-up
  3 = Call 911/988

Action truth strings: "None", "Routine Follow-up", "Contact Doctor", "Call 911/988"
"""

import os
import glob
import numpy as np
import pandas as pd

PARSED_DIR = os.path.join(os.path.dirname(__file__), "output", "parsed")
OUT_PATH = os.path.join(os.path.dirname(__file__), "output", "metrics", "action_cut_bootstrap_ci.csv")
N_BOOT = 10_000
SEED = 42

# action_truth strings that constitute urgent/emergent (CUT denominator)
CUT_URGENT_TRUTH = {"Call 911/988"}
# action_pred integers considered adequate for urgent cases
CUT_ADEQUATE_PRED = {3}  # only Call 911/988 is adequate for emergent cases

# action_pred integer -> action_truth string mapping
PRED_TO_TRUTH = {0: "None", 1: "Contact Doctor", 2: "Routine Follow-up", 3: "Call 911/988"}


def load_audit(path):
    df = pd.read_csv(path)
    df = df[df["parse_success"] == True].copy()
    return df


def action_accuracy(df):
    pred_str = df["action_pred"].map(PRED_TO_TRUTH)
    return (pred_str == df["action_truth"]).mean()


def cut_rate(df):
    urgent = df[df["action_truth"].isin(CUT_URGENT_TRUTH)]
    if len(urgent) == 0:
        return np.nan
    insufficient = ~urgent["action_pred"].isin(CUT_ADEQUATE_PRED)
    return insufficient.mean()


def bootstrap_ci(df, stat_fn, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(df)
    stats = np.array([
        stat_fn(df.iloc[rng.integers(0, n, size=n)])
        for _ in range(n_boot)
    ])
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return lo, hi


def arm_label(fname):
    base = os.path.basename(fname).replace("_audit.csv", "")
    # e.g. A_single_pass_qwen3_8b_physician
    parts = base.split("_")
    # find dataset suffix
    if base.endswith("_physician"):
        dataset = "physician"
        rest = base[: -len("_physician")]
    elif base.endswith("_realworld"):
        dataset = "realworld"
        rest = base[: -len("_realworld")]
    else:
        dataset = "unknown"
        rest = base
    return rest, dataset


def main():
    rng_seed = SEED
    files = sorted(glob.glob(os.path.join(PARSED_DIR, "*_audit.csv")))
    rows = []
    for fpath in files:
        arm_model, dataset = arm_label(fpath)
        df = load_audit(fpath)
        if len(df) == 0:
            continue
        n = len(df)

        acc = action_accuracy(df)
        acc_lo, acc_hi = bootstrap_ci(df, action_accuracy)

        cut = cut_rate(df)
        cut_lo, cut_hi = bootstrap_ci(df, cut_rate)

        rows.append({
            "arm_model": arm_model,
            "dataset": dataset,
            "n": n,
            "action_accuracy": round(acc * 100, 1),
            "action_accuracy_ci_lo": round(acc_lo * 100, 1) if not np.isnan(acc_lo) else "",
            "action_accuracy_ci_hi": round(acc_hi * 100, 1) if not np.isnan(acc_hi) else "",
            "cut": round(cut * 100, 1) if not np.isnan(cut) else "",
            "cut_ci_lo": round(cut_lo * 100, 1) if not np.isnan(cut_lo) else "",
            "cut_ci_hi": round(cut_hi * 100, 1) if not np.isnan(cut_hi) else "",
        })
        print(f"{arm_model} | {dataset} | n={n} | "
              f"Action Acc={acc*100:.1f}% ({acc_lo*100:.1f}-{acc_hi*100:.1f}%) | "
              f"CUT={cut*100:.1f}% ({cut_lo*100:.1f}-{cut_hi*100:.1f}%)"
              if not np.isnan(cut) else
              f"{arm_model} | {dataset} | n={n} | "
              f"Action Acc={acc*100:.1f}% ({acc_lo*100:.1f}-{acc_hi*100:.1f}%) | CUT=N/A")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()

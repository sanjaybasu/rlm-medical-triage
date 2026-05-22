"""Length-stratified PSR/CFS sensitivity analysis.

Partitions the physician test set (N=450) by patient-message character length into
three tertiles and recomputes the mean per-case PSR and CFS for Arms A and C across
the four open-source models, using per-case audit CSVs from output/parsed/.

Output: revision_v2/frontier_runs/length_stratified_psr.csv and stdout summary.
"""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path


PARSED_DIR = Path(os.environ.get("RLM_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "output"))) / "parsed"
RAW_DIR = Path(os.environ.get("RLM_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "output"))) / "raw"
PHYSICIAN_DATA = Path(os.environ.get("RLM_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))) / "physician_full.json"
OUT_DIR = Path(os.environ.get("RLM_REVISION_DIR", str(Path(__file__).resolve().parents[1]))) / "frontier_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


MODELS = [
    ("Llama-3.1-8B", "llama3.1_8b"),
    ("Qwen3-8B", "qwen3_8b"),
    ("Qwen3-32B", "qwen3_32b"),
    ("DeepSeek-R1-70B", "deepseek-r1_70b"),
]


def load_physician_lengths() -> dict[int, int]:
    with open(PHYSICIAN_DATA) as f:
        cases = json.load(f)
    return {i: len(c.get("prompt", c.get("message", ""))) for i, c in enumerate(cases)}


def load_audit_rows(arm: str, model_slug: str) -> list[dict]:
    csv_path = PARSED_DIR / f"{arm}_{model_slug}_physician_audit.csv"
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                row["case_idx"] = int(row["case_idx"])
                psr = row.get("phantom_symptom_rate", "")
                cfs = row.get("citation_fidelity_score", "")
                row["psr"] = float(psr) if psr not in ("", "None") else None
                row["cfs"] = float(cfs) if cfs not in ("", "None") else None
                row["n_evidence_claims"] = int(row.get("n_evidence_claims", "0") or 0)
                rows.append(row)
            except Exception:
                continue
    return rows


def stratify(lengths: dict[int, int]) -> dict[int, str]:
    """Return mapping case_idx -> 'T1' | 'T2' | 'T3' based on length tertiles."""
    vals = sorted(lengths.values())
    n = len(vals)
    t1_cut = vals[n // 3]
    t2_cut = vals[(2 * n) // 3]
    out = {}
    for idx, L in lengths.items():
        if L <= t1_cut:
            out[idx] = "T1"
        elif L <= t2_cut:
            out[idx] = "T2"
        else:
            out[idx] = "T3"
    return out


def main():
    lengths = load_physician_lengths()
    tertile = stratify(lengths)
    # Tertile bounds for display
    vals = sorted(lengths.values())
    n = len(vals)
    t1_max = vals[n // 3]
    t2_max = vals[(2 * n) // 3]
    t3_max = vals[-1]
    t1_min = vals[0]
    print(f"Length tertiles: T1 ≤ {t1_max} chars; T2 ≤ {t2_max}; T3 ≤ {t3_max} (overall range {t1_min}–{t3_max})\n")

    rows_out = []
    print(f"{'Model':<18} {'Arm':<5} {'Tertile':<8} {'N':>5} {'PSR%':>8} {'CFS%':>8}")
    print("-" * 60)
    for model_name, model_slug in MODELS:
        for arm in ("A_single_pass", "C_repl_only"):
            audit_rows = load_audit_rows(arm, model_slug)
            if not audit_rows:
                continue
            by_tertile = {"T1": [], "T2": [], "T3": []}
            for r in audit_rows:
                t = tertile.get(r["case_idx"])
                if t:
                    by_tertile[t].append(r)
            for t in ("T1", "T2", "T3"):
                trs = by_tertile[t]
                psr_vals = [r["psr"] for r in trs if r.get("psr") is not None]
                cfs_vals = [r["cfs"] for r in trs if r.get("cfs") is not None]
                psr_mean = statistics.mean(psr_vals) if psr_vals else float("nan")
                cfs_mean = statistics.mean(cfs_vals) if cfs_vals else float("nan")
                print(f"{model_name:<18} {arm[0]:<5} {t:<8} {len(trs):>5} {psr_mean*100:>7.1f}% {cfs_mean*100:>7.1f}%")
                rows_out.append({
                    "model": model_name,
                    "arm": arm,
                    "tertile": t,
                    "n_cases": len(trs),
                    "n_with_claims": len(psr_vals),
                    "psr_mean_pct": round(psr_mean * 100, 1) if psr_vals else None,
                    "cfs_mean_pct": round(cfs_mean * 100, 1) if cfs_vals else None,
                })
        print()

    # Save CSV
    out_csv = OUT_DIR / "length_stratified_psr.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nOutput: {out_csv}")
    print(f"Tertile bounds (chars): T1 ≤ {t1_max}, T2 ≤ {t2_max}, T3 ≤ {t3_max}")


if __name__ == "__main__":
    main()

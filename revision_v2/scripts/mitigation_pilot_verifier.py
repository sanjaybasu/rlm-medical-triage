"""Post-execution source-grounding verifier (mitigation pilot).

Reuses existing Qwen3-8B Arm C outputs. For each claimed (symptom, quote) pair, applies
a two-stage check:
  (1) quote check: fuzzy match (difflib.SequenceMatcher ratio >= 0.85) of the model's
      quote against substrings of the patient message with length in [50%, 200%] of the
      claim quote length;
  (2) symptom fallback: if (1) fails, check whether the symptom keyword (or any 1-3
      word slice) appears as a substring of the patient message.

Claims failing both checks are rejected. Cases with no surviving claims are coerced to
detection=0. Output: per-case CSV and aggregate JSON summary.

Usage:
    cd revision_v2/scripts
    python mitigation_pilot_verifier.py
"""
from __future__ import annotations

import csv
import difflib
import json
import math
import os
import re
import statistics
from pathlib import Path


RAW_JSONL = Path(os.environ.get("RLM_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "output"))) / "raw" / "C_repl_only_qwen3_8b_physician.jsonl"
PHYSICIAN_DATA = Path(os.environ.get("RLM_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))) / "physician_full.json"
OUT_DIR = Path(os.environ.get("RLM_REVISION_DIR", str(Path(__file__).resolve().parents[1]))) / "frontier_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    with open(PHYSICIAN_DATA) as f:
        physician = json.load(f)
    by_idx_message = {}
    for i, c in enumerate(physician):
        by_idx_message[i] = c.get("prompt", c.get("message", ""))

    cases = []
    with open(RAW_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
                r["patient_message"] = by_idx_message.get(r["case_idx"], "")
                r["detection_truth"] = int(physician[r["case_idx"]].get("detection_truth", 0))
                cases.append(r)
            except Exception:
                continue
    return cases


def _coerce_detection(v) -> int:
    """Best-effort coerce a parsed 'detection' field to 0/1. Treat unknown values as 0."""
    if v is None:
        return 0
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return 1 if v >= 0.5 else 0
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "hazard", "positive"):
        return 1
    return 0


def quote_grounded(quote: str, message: str, threshold: float = 0.85) -> bool:
    if not quote or not message:
        return False
    q = quote.strip()
    if len(q) < 4:
        return False
    # Sliding window over message substrings of length in [0.5q, 2.0q]
    m = message
    lo = max(4, int(len(q) * 0.5))
    hi = min(len(m), int(len(q) * 2.0))
    if hi <= lo:
        return False
    for L in range(lo, hi + 1, max(1, (hi - lo) // 20)):
        for start in range(0, max(1, len(m) - L + 1), max(1, L // 4)):
            sub = m[start : start + L]
            r = difflib.SequenceMatcher(None, q.lower(), sub.lower()).ratio()
            if r >= threshold:
                return True
    # Fallback: also try the message against the quote as full string (handles message << quote case)
    if difflib.SequenceMatcher(None, q.lower(), m.lower()).ratio() >= threshold:
        return True
    return False


def symptom_grounded(symptom: str, message: str) -> bool:
    if not symptom or not message:
        return False
    s = symptom.strip().lower()
    m = message.lower()
    if s in m:
        return True
    # Token-level: if any 1-3 word slice of the symptom appears in the message
    tokens = re.findall(r"\b\w+\b", s)
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            if len(phrase) >= 4 and phrase in m:
                return True
    return False


def verify_case(parsed: dict, message: str, threshold: float = 0.85) -> dict:
    if not parsed or not isinstance(parsed, dict):
        return {"original_claims": 0, "surviving_claims": 0, "phantom_post": 0, "kept": []}
    evidence = parsed.get("evidence", []) or []
    kept = []
    for claim in evidence:
        if not isinstance(claim, dict):
            continue
        q = claim.get("quote", "") or ""
        s = claim.get("symptom", "") or ""
        # Strict check
        if quote_grounded(q, message, threshold=threshold):
            kept.append({**claim, "_verified_via": "quote"})
        elif symptom_grounded(s, message):
            kept.append({**claim, "_verified_via": "symptom"})
        # else: rejected
    return {
        "original_claims": len(evidence),
        "surviving_claims": len(kept),
        "phantom_post": len(evidence) - len(kept),
        "kept": kept,
    }


def compute_metrics(rows: list[dict]) -> dict:
    """Compute aggregate metrics on a list of per-case records."""
    n = len(rows)
    n_with_claims = sum(1 for r in rows if r["claims_total"] > 0)
    # PSR (per-case mean, restricted to cases with ≥1 claim)
    psr_values = [r["psr"] for r in rows if r["claims_total"] > 0]
    cfs_values = [r["cfs"] for r in rows if r["claims_total"] > 0]
    psr = statistics.mean(psr_values) if psr_values else 0.0
    cfs = statistics.mean(cfs_values) if cfs_values else 0.0
    # Sensitivity (detection=1 on truth=1 cases)
    hazard_cases = [r for r in rows if r["truth"] == 1]
    sens = sum(1 for r in hazard_cases if r["detection_pred"] == 1) / max(len(hazard_cases), 1)
    # Specificity
    benign_cases = [r for r in rows if r["truth"] == 0]
    spec = sum(1 for r in benign_cases if r["detection_pred"] == 0) / max(len(benign_cases), 1)
    return {
        "n": n,
        "n_with_claims": n_with_claims,
        "psr": psr,
        "cfs": cfs,
        "sensitivity": sens,
        "specificity": spec,
        "n_hazard": len(hazard_cases),
        "n_benign": len(benign_cases),
    }


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def main():
    cases = load_data()
    print(f"Loaded {len(cases)} Qwen3-8B Arm C physician outputs.")

    before_rows = []
    after_rows = []
    per_case_dump = []

    for r in cases:
        parsed = r.get("parsed", {}) or {}
        message = r["patient_message"]
        truth = r["detection_truth"]
        # BEFORE — replicate the canonical metric: per-case PSR from raw evidence
        evidence_before = parsed.get("evidence", []) if isinstance(parsed, dict) else []
        n_claims_before = len(evidence_before) if isinstance(evidence_before, list) else 0
        n_phantom_before = 0
        n_valid_quote_before = 0
        for claim in evidence_before if isinstance(evidence_before, list) else []:
            if not isinstance(claim, dict):
                continue
            q = claim.get("quote", "") or ""
            s = claim.get("symptom", "") or ""
            # Original algorithm: phantom if quote not present AND symptom keyword not in message
            quote_ok = quote_grounded(q, message, threshold=0.7)  # original threshold was 0.7
            keyword_ok = symptom_grounded(s, message)
            if not (quote_ok or keyword_ok):
                n_phantom_before += 1
            if quote_ok:
                n_valid_quote_before += 1
        psr_before = n_phantom_before / n_claims_before if n_claims_before > 0 else 0.0
        cfs_before = n_valid_quote_before / n_claims_before if n_claims_before > 0 else 0.0
        det_before = _coerce_detection(parsed.get("detection", 0)) if isinstance(parsed, dict) else 0

        before_rows.append({
            "case_idx": r["case_idx"],
            "truth": truth,
            "detection_pred": det_before,
            "claims_total": n_claims_before,
            "psr": psr_before,
            "cfs": cfs_before,
        })

        # AFTER — apply post-execution verifier (strict threshold 0.85)
        v = verify_case(parsed, message, threshold=0.85)
        surviving = v["surviving_claims"]
        if surviving == 0:
            # No grounded findings → detection=0 ("we cannot verify any claimed evidence")
            det_after = 0
        else:
            # ≥1 grounded finding → keep the model's detection signal (allows escalation only on
            # verified evidence; conservatively keeps detection=1 if model originally flagged)
            det_after = _coerce_detection(parsed.get("detection", 0)) if isinstance(parsed, dict) else 0
        # Post-PSR: only "surviving" claims contribute → PSR of surviving set is by construction 0
        # (every surviving claim has either grounded quote or grounded symptom)
        psr_after_strict = 0.0 if surviving > 0 else 0.0  # zero by construction
        # Post-CFS: of surviving claims, what fraction have a grounded quote (not just keyword)
        n_quote_in_surviving = sum(1 for c in v["kept"] if c.get("_verified_via") == "quote")
        cfs_after_strict = n_quote_in_surviving / surviving if surviving > 0 else 0.0

        after_rows.append({
            "case_idx": r["case_idx"],
            "truth": truth,
            "detection_pred": det_after,
            "claims_total": surviving,
            "psr": psr_after_strict,
            "cfs": cfs_after_strict,
        })

        per_case_dump.append({
            "case_idx": r["case_idx"],
            "truth": truth,
            "claims_original": n_claims_before,
            "claims_surviving": surviving,
            "n_phantom_before": n_phantom_before,
            "claims_rejected_by_verifier": n_claims_before - surviving if n_claims_before > 0 else 0,
            "psr_before": round(psr_before, 4),
            "psr_after": round(psr_after_strict, 4),
            "cfs_before": round(cfs_before, 4),
            "cfs_after": round(cfs_after_strict, 4),
            "detection_before": det_before,
            "detection_after": det_after,
        })

    m_before = compute_metrics(before_rows)
    m_after = compute_metrics(after_rows)

    # CIs
    psr_before_ci = wilson_ci(m_before["psr"], m_before["n_with_claims"])
    psr_after_ci = wilson_ci(m_after["psr"], max(m_after["n_with_claims"], 1))
    cfs_before_ci = wilson_ci(m_before["cfs"], m_before["n_with_claims"])
    cfs_after_ci = wilson_ci(m_after["cfs"], max(m_after["n_with_claims"], 1))
    sens_before_ci = wilson_ci(m_before["sensitivity"], m_before["n_hazard"])
    sens_after_ci = wilson_ci(m_after["sensitivity"], m_after["n_hazard"])
    spec_before_ci = wilson_ci(m_before["specificity"], m_before["n_benign"])
    spec_after_ci = wilson_ci(m_after["specificity"], m_after["n_benign"])

    summary = {
        "n": m_before["n"],
        "n_with_claims_before": m_before["n_with_claims"],
        "n_with_claims_after": m_after["n_with_claims"],
        "psr_before": {"point": round(m_before["psr"] * 100, 1),
                       "ci": [round(psr_before_ci[0] * 100, 1), round(psr_before_ci[1] * 100, 1)]},
        "psr_after": {"point": round(m_after["psr"] * 100, 1),
                      "ci": [round(psr_after_ci[0] * 100, 1), round(psr_after_ci[1] * 100, 1)]},
        "psr_pp_reduction": round((m_before["psr"] - m_after["psr"]) * 100, 1),
        "cfs_before": {"point": round(m_before["cfs"] * 100, 1),
                       "ci": [round(cfs_before_ci[0] * 100, 1), round(cfs_before_ci[1] * 100, 1)]},
        "cfs_after": {"point": round(m_after["cfs"] * 100, 1),
                      "ci": [round(cfs_after_ci[0] * 100, 1), round(cfs_after_ci[1] * 100, 1)]},
        "cfs_pp_improvement": round((m_after["cfs"] - m_before["cfs"]) * 100, 1),
        "sensitivity_before": {"point": round(m_before["sensitivity"] * 100, 1),
                               "ci": [round(sens_before_ci[0] * 100, 1), round(sens_before_ci[1] * 100, 1)]},
        "sensitivity_after": {"point": round(m_after["sensitivity"] * 100, 1),
                              "ci": [round(sens_after_ci[0] * 100, 1), round(sens_after_ci[1] * 100, 1)]},
        "sens_pp_delta": round((m_after["sensitivity"] - m_before["sensitivity"]) * 100, 1),
        "specificity_before": {"point": round(m_before["specificity"] * 100, 1),
                               "ci": [round(spec_before_ci[0] * 100, 1), round(spec_before_ci[1] * 100, 1)]},
        "specificity_after": {"point": round(m_after["specificity"] * 100, 1),
                              "ci": [round(spec_after_ci[0] * 100, 1), round(spec_after_ci[1] * 100, 1)]},
        "n_hazard": m_before["n_hazard"],
        "n_benign": m_before["n_benign"],
    }

    # Save outputs
    summary_path = OUT_DIR / "mitigation_pilot_qwen3_8b_summary.json"
    csv_path = OUT_DIR / "mitigation_pilot_qwen3_8b_per_case.csv"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_case_dump[0].keys()))
        w.writeheader()
        w.writerows(per_case_dump)

    # Stdout summary
    print("\n========== MITIGATION PILOT SUMMARY ==========")
    print(f"Model: Qwen3-8B, Arm: C (REPL only, fair prompt), N: {summary['n']}")
    print(f"Verifier: strict quote-grounding (ratio ≥ 0.85) + symptom-keyword fallback")
    print()
    print(f"PSR before:  {summary['psr_before']['point']}% (95% CI: {summary['psr_before']['ci'][0]}–{summary['psr_before']['ci'][1]}%)")
    print(f"PSR after:   {summary['psr_after']['point']}% (95% CI: {summary['psr_after']['ci'][0]}–{summary['psr_after']['ci'][1]}%)")
    print(f"PSR reduction: {summary['psr_pp_reduction']} pp")
    print()
    print(f"CFS before:  {summary['cfs_before']['point']}% (95% CI: {summary['cfs_before']['ci'][0]}–{summary['cfs_before']['ci'][1]}%)")
    print(f"CFS after:   {summary['cfs_after']['point']}% (95% CI: {summary['cfs_after']['ci'][0]}–{summary['cfs_after']['ci'][1]}%)")
    print(f"CFS improvement: +{summary['cfs_pp_improvement']} pp")
    print()
    print(f"Sensitivity before: {summary['sensitivity_before']['point']}% (n_hazard={summary['n_hazard']})")
    print(f"Sensitivity after:  {summary['sensitivity_after']['point']}%")
    print(f"Sens Δ: {summary['sens_pp_delta']:+.1f} pp")
    print(f"Specificity before: {summary['specificity_before']['point']}%")
    print(f"Specificity after:  {summary['specificity_after']['point']}%")
    print()
    print(f"Outputs: {summary_path}")
    print(f"         {csv_path}")


if __name__ == "__main__":
    main()

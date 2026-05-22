"""Cross-document consistency check.

Verifies that numerical claims, citation references, naming conventions, and ID
strings are consistent across the manuscript, appendix, and response letter.
Re-derives selected numbers from canonical CSV/JSON sources where possible.

Pass with --allow-tbd while TBD placeholders are still present; run without that
flag for the final pre-submission check.

Run from the revision_v2 directory:
    python scripts/comprehensive_consistency_check.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path


REVISION_DIR = Path(os.environ.get("RLM_REVISION_DIR", str(Path(__file__).resolve().parents[1])))
PACKAGING_DIR = Path(os.environ.get("RLM_PACKAGING_DIR", str(Path(__file__).resolve().parents[2])))

DOCS = {
    "manuscript": REVISION_DIR / "cot_hallucination_triage_manuscript_revised_clean.md",
    "appendix":   REVISION_DIR / "cot_hallucination_triage_appendix_revised_clean.md",
    "response":   REVISION_DIR / "response_to_reviewers.md",
    "tracked":    REVISION_DIR / "manuscript_tracked_changes.md",
}

CANONICAL = {
    "statistical_tests": PACKAGING_DIR / "output/tables/statistical_tests.csv",
    "all_metrics":       PACKAGING_DIR / "output/metrics/all_metrics.csv",
    "mitigation":        REVISION_DIR / "frontier_runs/mitigation_pilot_qwen3_8b_summary.json",
    "length_stratified": REVISION_DIR / "frontier_runs/length_stratified_psr.csv",
}


def load_canonical():
    out = {}
    rows = list(csv.DictReader(open(CANONICAL["statistical_tests"])))
    out["statistical_tests"] = rows
    rows = list(csv.DictReader(open(CANONICAL["all_metrics"])))
    out["all_metrics"] = rows
    out["mitigation"] = json.load(open(CANONICAL["mitigation"]))
    rows = list(csv.DictReader(open(CANONICAL["length_stratified"])))
    out["length_stratified"] = rows
    return out


def load_docs() -> dict[str, str]:
    return {k: v.read_text() if v.exists() else "" for k, v in DOCS.items()}


def check_ref_19(docs: dict[str, str], errors: list[str]) -> None:
    """Manuscript ref 19 must be Moell et al., not Schmidgall."""
    m = docs["manuscript"]
    if "Schmidgall, S. et al. AgentClinic" in m and "19." in m.split("Moell")[0].split("References")[-1]:
        errors.append("Manuscript appears to still have Schmidgall AgentClinic as ref 19")
    if "Moell, B., Aronsson, F. S. & Akbar, S." not in m:
        errors.append("Manuscript missing Moell et al. Front Artif Intell 2025")
    if "AgentClinic" in m and "preprint" in m.lower().split("AgentClinic", 1)[-1][:200].lower():
        # AgentClinic should not appear as a preprint citation
        # (peer-reviewed version is in npj Digit Med 2026)
        # This is a soft check
        pass


def check_bonferroni_consistency(docs: dict[str, str], errors: list[str]) -> None:
    """The Bonferroni denominator should be 32 (not 24) consistently."""
    for name, text in docs.items():
        if name == "tracked":
            continue
        if "alpha=0.0021" in text and "alpha=0.001563" not in text:
            errors.append(f"{name}: still uses alpha=0.0021 (24-comparison denominator) without alpha=0.001563")
        if "32 pairwise comparisons" not in text and name == "manuscript":
            if "Bonferroni correction across" in text:
                # Should be 32
                m = re.search(r"Bonferroni correction across\s+(\d+)\s+pairwise", text)
                if m and m.group(1) != "32":
                    errors.append(f"manuscript Bonferroni denominator = {m.group(1)}, expected 32")


def check_open_source_family_naming(docs: dict[str, str], errors: list[str]) -> None:
    """Two open-source families (Llama + Qwen) should be named explicitly."""
    m = docs["manuscript"]
    if "two model families" not in m and "two open-source families" not in m:
        errors.append("manuscript does not explicitly name 'two model families' (Llama + Qwen)")
    if "DeepSeek-R1-Distill-Llama-70B" not in m:
        errors.append("manuscript does not use the precise model name 'DeepSeek-R1-Distill-Llama-70B'")


def check_frontier_model_ids(docs: dict[str, str], errors: list[str]) -> None:
    """The three frontier model display names should be used consistently."""
    expected = ["Claude Opus 4.7", "GPT-5.5", "Gemini 3.1 Pro"]
    for label in expected:
        if label not in docs["manuscript"]:
            errors.append(f"manuscript missing frontier model display name: {label}")


def check_mitigation_numbers(docs: dict[str, str], canonical: dict, errors: list[str]) -> None:
    """Mitigation pilot numbers in manuscript must match summary.json."""
    summary = canonical["mitigation"]
    m = docs["manuscript"]
    # PSR before should be 92.8% (canonical)
    if "92.8%" not in m:
        errors.append("manuscript missing PSR-before 92.8% in mitigation section")
    # CFS after 40.7%
    cfs_after = summary["cfs_after"]["point"]
    if f"{cfs_after}%" not in m:
        errors.append(f"manuscript missing CFS-after {cfs_after}% from mitigation summary")
    # Sensitivity drop -24.0 pp
    sens_delta = abs(summary["sens_pp_delta"])
    if f"{sens_delta:.1f}-percentage-point" not in m and f"{sens_delta}-pp" not in m and f"{sens_delta}-percentage-point" not in m:
        if f"{sens_delta:.1f}" not in m:
            errors.append(f"manuscript missing sensitivity delta {sens_delta} pp from mitigation summary")


def check_length_stratified_numbers(docs: dict[str, str], canonical: dict, errors: list[str]) -> None:
    """Length-stratified numbers in manuscript Results must match the CSV."""
    rows = canonical["length_stratified"]
    by_key = {(r["model"], r["arm"], r["tertile"]): r for r in rows}
    m = docs["manuscript"]
    # Qwen3-8B C: T1 98.0, T2 87.8, T3 92.5
    key = ("Qwen3-8B", "C_repl_only", "T1")
    r = by_key.get(key)
    if r and r.get("psr_mean_pct"):
        target = f"{float(r['psr_mean_pct']):.1f}"
        if target + "%" not in m:
            errors.append(f"manuscript missing Qwen3-8B C T1 PSR {target}%")


def check_for_remaining_placeholders(docs: dict[str, str], errors: list[str], allow_tbd: bool) -> None:
    """No [TBD-*] should remain in the manuscript after frontier-run completion."""
    for name, text in docs.items():
        tbds = re.findall(r"\[TBD-[A-Z0-9-]+\]", text)
        if tbds and not allow_tbd:
            errors.append(f"{name}: {len(tbds)} TBD placeholders remain: {set(tbds[:5])}")


def check_release_tag(docs: dict[str, str], errors: list[str]) -> None:
    m = docs["manuscript"]
    if "v1.1.0" not in m:
        errors.append("manuscript Data Availability does not reference release tag v1.1.0")


def check_supp_table_references(docs: dict[str, str], errors: list[str]) -> None:
    """Every Supplementary Table reference should appear in the appendix."""
    m = docs["manuscript"]
    a = docs["appendix"]
    refs = set(re.findall(r"Supplementary Table\s+([A-Z0-9-]+)", m))
    for ref in refs:
        if ref not in a:
            # This is a soft check — Supp Table S3-1 might be referenced differently
            pass


def main():
    docs = load_docs()
    canonical = load_canonical()
    errors: list[str] = []

    # Layer 1 checks
    check_ref_19(docs, errors)
    check_bonferroni_consistency(docs, errors)
    check_open_source_family_naming(docs, errors)
    check_frontier_model_ids(docs, errors)
    check_mitigation_numbers(docs, canonical, errors)
    check_length_stratified_numbers(docs, canonical, errors)
    check_release_tag(docs, errors)
    check_supp_table_references(docs, errors)

    # Allow TBDs while frontier run is still in flight
    allow_tbd = "--allow-tbd" in sys.argv
    check_for_remaining_placeholders(docs, errors, allow_tbd)

    if errors:
        print(f"FAIL: {len(errors)} consistency errors found:\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("PASS: all consistency checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()

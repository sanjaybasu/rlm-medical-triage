"""M1: Phantom symptom string-distribution analysis.

For each (model, Arm C) output JSONL, tabulate phantom symptom strings (claimed
symptoms whose quote does not match the patient message) and compute concentration
metrics: top-3 and top-10 share, Gini coefficient, and overlap with a 30-term
canonical clinical vocabulary.

Inputs: open-source Arm C JSONLs in output/raw/ and frontier Arm C JSONLs in
revision_v2/frontier_runs/. Output: per-model summary CSV and top-20 phantoms CSV.
"""
from __future__ import annotations

import csv
import difflib
import json
import os
from collections import Counter
from pathlib import Path


PHYSICIAN_DATA = Path(os.environ.get("RLM_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))) / "physician_full.json"
OPEN_SOURCE_DIR = Path(os.environ.get("RLM_OUTPUT_DIR", str(Path(__file__).resolve().parents[2] / "output"))) / "raw"
FRONTIER_DIR = Path(os.environ.get("RLM_REVISION_DIR", str(Path(__file__).resolve().parents[1]))) / "frontier_runs"
OUT_DIR = Path(os.environ.get("RLM_REVISION_DIR", str(Path(__file__).resolve().parents[1]))) / "frontier_runs"

# Canonical training-corpus clinical terms — these appear with high frequency
# in medical textbooks and clinical case reports, and would be the most
# parametric-memory-accessible terms when a model fabricates clinical findings
CANONICAL_TERMS = {
    "chest pain", "shortness of breath", "difficulty breathing", "sob",
    "dyspnea", "nausea", "vomiting", "dizziness", "lightheadedness",
    "fatigue", "weakness", "headache", "abdominal pain", "back pain",
    "fever", "chills", "cough", "rash", "swelling", "edema", "bleeding",
    "palpitations", "syncope", "confusion", "anxiety", "depression",
    "tachycardia", "bradycardia", "hypotension", "hypertension",
}


def is_phantom(quote: str, msg: str, threshold: float = 0.7) -> bool:
    if not quote or len(quote) < 4:
        return True
    return (
        difflib.SequenceMatcher(None, quote.lower(), msg.lower()).ratio() < threshold
        and quote.lower() not in msg.lower()
    )


def is_canonical(symptom: str) -> bool:
    s = symptom.lower().strip()
    for canon in CANONICAL_TERMS:
        if canon in s or s in canon:
            return True
    return False


def load_physician():
    with open(PHYSICIAN_DATA) as f:
        cases = json.load(f)
    return {i: c.get("prompt", c.get("message", "")) for i, c in enumerate(cases)}


def collect_phantoms(jsonl_path: Path, message_by_idx: dict[int, str]) -> list[str]:
    """Return the list of phantom symptom strings across all cases in this file."""
    phantoms: list[str] = []
    if not jsonl_path.exists():
        return phantoms
    for line in open(jsonl_path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("error"):
            continue
        parsed = r.get("parsed") or {}
        if not isinstance(parsed, dict):
            continue
        idx = r["case_idx"]
        msg = message_by_idx.get(idx, "")
        for ev in parsed.get("evidence", []) or []:
            if not isinstance(ev, dict):
                continue
            quote = ev.get("quote", "") or ""
            symptom = ev.get("symptom", "") or ""
            if is_phantom(quote, msg):
                phantoms.append(symptom.strip())
    return phantoms


def gini(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    cum = 0.0
    for i, v in enumerate(sorted_v, 1):
        cum += v * i
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    return (2 * cum) / (n * total) - (n + 1) / n


def main():
    msgs = load_physician()

    # Collect Arm C phantoms across models
    targets = [
        ("Llama-3.1-8B", OPEN_SOURCE_DIR / "C_repl_only_llama3.1_8b_physician.jsonl"),
        ("Qwen3-8B", OPEN_SOURCE_DIR / "C_repl_only_qwen3_8b_physician.jsonl"),
        ("Qwen3-32B", OPEN_SOURCE_DIR / "C_repl_only_qwen3_32b_physician.jsonl"),
        ("DeepSeek-R1-70B", OPEN_SOURCE_DIR / "C_repl_only_deepseek-r1_70b_physician.jsonl"),
        ("Claude Opus 4.7", FRONTIER_DIR / "C_frontier_claude-opus-4-7_physician.jsonl"),
        ("GPT-5.5", FRONTIER_DIR / "C_frontier_gpt-5.5_physician.jsonl"),
        ("Gemini 3.1 Pro", FRONTIER_DIR / "C_frontier_gemini-3.1-pro-preview_physician.jsonl"),
    ]

    summary_rows = []
    top20_rows = []
    print("\n========== M1: PHANTOM SYMPTOM STEREOTYPING (Arm C) ==========\n")
    print(f"{'Model':<22} {'N_phantoms':>10} {'Top10_share':>12} {'Top3_share':>11} {'Canonical_rate':>15} {'Gini':>7}")
    print("-" * 90)

    for model, path in targets:
        phantoms = collect_phantoms(path, msgs)
        n = len(phantoms)
        if n == 0:
            print(f"{model:<22} {n:>10} {'n/a':>12} {'n/a':>11} {'n/a':>15} {'n/a':>7}")
            continue
        counter = Counter(phantoms)
        top20 = counter.most_common(20)
        total = sum(counter.values())
        top10 = sum(c for _, c in counter.most_common(10))
        top3 = sum(c for _, c in counter.most_common(3))
        top10_share = top10 / total
        top3_share = top3 / total
        canon = sum(c for s, c in counter.items() if is_canonical(s))
        canon_rate = canon / total
        g = gini(list(counter.values()))
        print(f"{model:<22} {n:>10} {top10_share*100:>11.1f}% {top3_share*100:>10.1f}% {canon_rate*100:>14.1f}% {g:>7.3f}")
        summary_rows.append({
            "model": model,
            "n_phantoms": n,
            "n_unique_symptoms": len(counter),
            "top10_share_pct": round(top10_share * 100, 1),
            "top3_share_pct": round(top3_share * 100, 1),
            "canonical_term_rate_pct": round(canon_rate * 100, 1),
            "gini_concentration": round(g, 3),
            "top1_symptom": top20[0][0] if top20 else "",
            "top1_count": top20[0][1] if top20 else 0,
        })
        for sym, cnt in top20:
            top20_rows.append({
                "model": model,
                "symptom": sym,
                "count": cnt,
                "share_pct": round(100 * cnt / total, 1),
                "is_canonical": is_canonical(sym),
            })

    # Save
    sum_path = OUT_DIR / "m1_phantom_stereotyping_summary.csv"
    top_path = OUT_DIR / "m1_phantom_stereotyping_top20.csv"
    if summary_rows:
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    if top20_rows:
        with open(top_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(top20_rows[0].keys()))
            w.writeheader()
            w.writerows(top20_rows)

    print("\n--- TOP-5 PHANTOM SYMPTOMS PER MODEL ---")
    for model, _ in targets:
        rows = [r for r in top20_rows if r["model"] == model][:5]
        if not rows:
            continue
        print(f"\n  {model}:")
        for r in rows:
            mark = " (canonical)" if r["is_canonical"] else ""
            print(f"    {r['count']:>4}× ({r['share_pct']:>5.1f}%) {r['symptom'][:60]}{mark}")

    print("\nMechanism prediction:")
    print("  - If open-source REPL hallucinations are parametric-memory shortcuts,")
    print("    canonical-rate should be HIGH (>50%) and top-3 share should be HIGH (>40%)")
    print("    for the high-PSR open-source models.")
    print("  - Frontier models, where Arm C PSR is low, should have LOW canonical-rate AND")
    print("    LOW top-3 share — the few phantoms they produce should be context-specific.")
    print()
    print(f"Outputs: {sum_path}")
    print(f"         {top_path}")


if __name__ == "__main__":
    main()

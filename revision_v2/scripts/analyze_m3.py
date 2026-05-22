"""M3 analyzer: JSON parser with ast.literal_eval fallback.

DeepSeek-R1 outputs sometimes use Python-style dict literals (single quotes). This
script re-parses with ast.literal_eval when json.loads fails, then computes per-case
PSR/sensitivity/specificity for the M3 file.
"""
import ast
import difflib
import json
import os
import re
import sys
from pathlib import Path

FRONTIER = Path(os.environ.get("RLM_REVISION_DIR", str(Path(__file__).resolve().parents[1]))) / "frontier_runs"
PHYSICIAN = Path(os.environ.get("RLM_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))) / "physician_full.json"


def coerce(v):
    if v is None: return 0
    if isinstance(v, bool): return int(v)
    if isinstance(v, (int, float)): return 1 if v >= 0.5 else 0
    return 1 if str(v).strip().lower() in ("1","true","yes","hazard","positive") else 0


def is_phantom(quote, msg):
    if not quote or len(quote) < 4: return True
    return (
        difflib.SequenceMatcher(None, str(quote).lower(), str(msg).lower()).ratio() < 0.7
        and str(quote).lower() not in str(msg).lower()
    )


def robust_parse(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
        try:
            return ast.literal_eval(m.group())
        except Exception:
            pass
    return None


def main():
    phys = json.load(open(PHYSICIAN))
    truth = {i: int(c.get("detection_truth", 0)) for i, c in enumerate(phys)}
    msgs = {i: c.get("prompt", c.get("message", "")) for i, c in enumerate(phys)}

    path = FRONTIER / "M3_C_thinkingEnabled_deepseek-r1_70b_physician.jsonl"
    if not path.exists():
        print("M3 file not found yet")
        return

    rows = [json.loads(l) for l in open(path)]
    n_total = len(rows)
    n_err = 0
    n_haz = 0; tp = 0; n_ben = 0; tn = 0
    n_claims = 0; n_phantom = 0
    print(f"M3 cases: {n_total}\n")
    for r in rows:
        raw = r.get("raw_response", "")
        err = "ERROR:" in raw[:50] or "Traceback" in raw[:50]
        if err:
            n_err += 1
            print(f"case {r['case_idx']}: ERROR")
            continue
        # Re-parse with robust parser
        parsed = robust_parse(raw)
        if not isinstance(parsed, dict):
            print(f"case {r['case_idx']}: UNPARSEABLE raw[:200]: {raw[:200]!r}")
            continue
        det = coerce(parsed.get("detection", 0))
        idx = r["case_idx"]; tt = truth.get(idx, 0)
        if tt == 1:
            n_haz += 1
            if det == 1: tp += 1
        else:
            n_ben += 1
            if det == 0: tn += 1
        msg = msgs.get(idx, "")
        case_phantoms = []
        for ev in parsed.get("evidence", []) or []:
            if not isinstance(ev, dict): continue
            n_claims += 1
            q = ev.get("quote", "") or ""
            if is_phantom(q, msg):
                n_phantom += 1
                case_phantoms.append(ev.get("symptom", "") or q)
        print(f"case {idx}: det={det} truth={tt} n_ev={len(parsed.get('evidence',[]) or [])} elapsed={r.get('elapsed_sec'):.0f}s phantoms={case_phantoms}")

    print()
    print(f"M3 SUMMARY (DeepSeek-R1-70B with reasoning enabled, Arm C):")
    print(f"  N={n_total}, errors={n_err}, n_hazard={n_haz}, n_benign={n_ben}")
    print(f"  Sensitivity: {tp/max(n_haz,1)*100:.1f}% ({tp}/{n_haz})")
    print(f"  Specificity: {tn/max(n_ben,1)*100:.1f}% ({tn}/{n_ben})")
    print(f"  PSR: {n_phantom/max(n_claims,1)*100:.1f}% ({n_phantom}/{n_claims} claims)")
    print()
    print(f"  Reference: DeepSeek-R1-70B Arm C with /no_think prefix produced PSR 96.9% at N=170.")


if __name__ == "__main__":
    main()

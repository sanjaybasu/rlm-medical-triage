"""M3: DeepSeek-R1-Distill-Llama-70B Arm C with reasoning enabled.

Runs locally against an Ollama installation. Requires a 32K-context variant of the
70B model (see README for the Modelfile snippet).

Default per-case wall-clock cap is 1200 s; cases exceeding the cap or the underlying
httpx read timeout are written with an ERROR record and the loop continues.

Usage:
    python m3_local_deepseek_thinking.py --pilot 5
    python m3_local_deepseek_thinking.py --n-max 100
    python m3_local_deepseek_thinking.py

Output: JSONL with one row per case. Resume-safe (case indices already in the file
are skipped on relaunch).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


PHYSICIAN = Path(os.environ.get("RLM_DATA_DIR", str(Path(__file__).resolve().parents[2] / "data"))) / "physician_full.json"
OUTPUT = Path(os.environ.get("RLM_REVISION_DIR", str(Path(__file__).resolve().parents[1]))) / "frontier_runs" / "M3_C_thinkingEnabled_deepseek-r1_70b_physician.jsonl"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# Inline RLM_SYSTEM (same as M4 — Arm C with sub-calls disabled)
# IMPORTANT: curly braces in JSON example are doubled because RLM lib calls .format() internally
RLM_SYSTEM_ARM_C = """You are a medical triage safety system. A patient message is stored in the REPL variable `context`. Your job is to determine if it contains a safety hazard.

RULES:
- Only identify clinical findings EXPLICITLY stated in the patient message.
- Do NOT hallucinate symptoms not in `context`.
- For each clinical finding, you MUST provide an exact quote from the patient message.

TASK: Write Python code in ```repl blocks to programmatically analyze the patient message. You decide what extraction and analysis approach to use. You may use any Python capabilities available, including string operations, regular expressions, or other techniques.

OUTPUT: When done, signal your final answer using FINAL(...) on its own line (NOT inside a code block), containing a JSON object with this structure:
{{"detection": 0 or 1, "action": "None" or "Routine Follow-up" or "Contact Doctor" or "Call 911/988", "evidence": [{{"symptom": "clinical finding", "quote": "exact text from message", "assessment": "benign/routine/urgent/emergent"}}]}}

IMPORTANT: You MUST end with a FINAL(...) statement containing the JSON. Without FINAL(), your answer will not be recorded.
{custom_tools_section}"""


def _parse_json_response(text: str):
    """Parse JSON or Python-style dict response. DeepSeek-R1 sometimes outputs
    Python-repr-style dicts (single quotes); we use ast.literal_eval as a fallback."""
    if not text:
        return None
    import re as _re
    import ast
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _re.search(r"\{.*\}", text, _re.DOTALL)
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


def load_done() -> set[int]:
    done = set()
    if OUTPUT.exists():
        for line in open(OUTPUT):
            try:
                r = json.loads(line)
                done.add(int(r["case_idx"]))
            except Exception:
                continue
    return done


def append_row(row: dict) -> None:
    with open(OUTPUT, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--n-max", type=int, default=0)
    ap.add_argument("--per-case-timeout", type=int, default=900,
                    help="Per-case wall-clock cap (seconds). Default 900s (15 min) — reasoning may be slow")
    args = ap.parse_args()

    # Late import so script can be inspected without rlm installed
    try:
        from rlm import RLM
    except ImportError:
        print("ERROR: rlms library not installed. Install with: pip install rlms==0.1.1", file=sys.stderr)
        sys.exit(1)

    # Verify Ollama is running with deepseek-r1:70b available
    import urllib.request
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read())
        models = [m["name"] for m in tags.get("models", [])]
        if not any("deepseek-r1:70b-32k" in m for m in models):
            print(f"ERROR: deepseek-r1:70b-32k not in local Ollama. Run: ollama create deepseek-r1:70b-32k -f /tmp/Modelfile_deepseek_r1_70b_32k (with FROM deepseek-r1:70b\\nPARAMETER num_ctx 32768)", file=sys.stderr)
            print(f"Available: {models}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Ollama not running at http://localhost:11434: {e}", file=sys.stderr)
        print("Start with: ollama serve  (or open the Ollama app)", file=sys.stderr)
        sys.exit(1)

    # Build RLM with THINKING ENABLED (no /no_think prefix)
    # M3 omits the /no_think prefix used in the original DeepSeek-R1 Arm C runs.
    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": "deepseek-r1:70b-32k",  # 32K context variant created locally to fit DeepSeek-R1's extended thinking chains; vanilla deepseek-r1:70b defaults to 4K context which is too small
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
        },
        environment="local",
        max_depth=1,            # Arm C: shallow (matches paper's Arm C config)
        max_iterations=8,       # matches paper
        custom_system_prompt=RLM_SYSTEM_ARM_C,  # NO /no_think prefix — this is M3's intervention
        verbose=False,
    )

    with open(PHYSICIAN) as f:
        cases = json.load(f)

    if args.pilot:
        # Stratified pilot: take first n_pilot cases (or could do stratified by hazard category)
        cases = cases[: args.pilot]
        print(f"M3 PILOT: {len(cases)} cases")
    elif args.n_max:
        cases = cases[: args.n_max]
        print(f"M3 N_MAX: {len(cases)} cases")
    else:
        print(f"M3 FULL: {len(cases)} cases")

    done = load_done()
    pending = [i for i in range(len(cases)) if i not in done]
    print(f"  {len(done)}/{len(cases)} already done; {len(pending)} pending")

    import concurrent.futures
    for idx, case_idx in enumerate(pending):
        case = cases[case_idx]
        message = case.get("prompt", case.get("message", ""))
        name = case.get("name", case.get("case_label", f"case_{case_idx}"))
        t0 = time.time()
        raw_text = ""
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(rlm.completion, message)
            try:
                result = fut.result(timeout=args.per_case_timeout)
                raw_text = result if isinstance(result, str) else getattr(result, "response", str(result))
            except concurrent.futures.TimeoutError:
                raw_text = f"ERROR: case timed out after {args.per_case_timeout}s"
            except Exception:
                raw_text = f"ERROR: {traceback.format_exc()}"
        elapsed = time.time() - t0

        row = {
            "case_idx": case_idx,
            "case_name": name,
            "model": "deepseek-r1:70b-32k",
            "arm": "M3_C_thinkingEnabled",
            "raw_response": raw_text,
            "parsed": _parse_json_response(raw_text),
            "elapsed_sec": round(elapsed, 2),
        }
        append_row(row)
        print(f"  [{idx + 1}/{len(pending)}] case {case_idx}: {elapsed:.1f}s "
              f"({'ERR' if raw_text.startswith('ERROR') else 'OK'})", flush=True)


if __name__ == "__main__":
    main()

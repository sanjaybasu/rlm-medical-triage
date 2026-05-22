"""M4: Qwen3-8B Arm C with a chain-of-thought thinking preamble.

Modal app launching the Arm C system prompt with a four-step CoT preamble prepended.
All prompts inlined (no dependency on the top-level prompts.py module). Use ::pilot
for a 5-case smoke test and ::full for the full physician set.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

import modal

app = modal.App("rlm-m4-cotpre-v2")

volume = modal.Volume.from_name("rlm-frontier-results", create_if_missing=True)

# Cache-buster: unique env var ensures fresh image build
_IMAGE_REV = "m4-v2-2026-05-21-fresh"

ollama_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "procps", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "openai>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "rlms==0.1.1",
        "json5",
    )
    .env({"_M4_REV": _IMAGE_REV})
)

# Inline the RLM_SYSTEM prompt (matches packaging/prompts.py:RLM_SYSTEM exactly,
# minus llm_query refs since Arm C disables sub-calls)
# IMPORTANT: curly braces in the JSON example are DOUBLED ({{ }}) because the RLM
# library calls .format(custom_tools_section=...) on the system prompt internally.
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

# M4 modification: prepend a chain-of-thought thinking preamble
COT_PREAMBLE = """FIRST, before writing any code, think step by step about your extraction strategy:
  1. Plan: identify which Python operations will read from the `context` variable (which contains the patient message). DO NOT generate clinical findings from your training-derived knowledge — every claim must come from running code that reads from `context`.
  2. Sketch: what regular expressions or string operations will you apply to `context`?
  3. Execute: write the Python code that USES `context` to extract findings.
  4. Verify: every claimed finding's quote must be a literal substring of `context`. If a finding is not in `context`, do not claim it.

Only after you have written down these four steps in your response should you begin writing executable code.

"""

# Simple JSON-extract helper inlined
def _parse_json_response(text: str):
    if not text:
        return None
    import re as _re
    try:
        return json.loads(text)
    except Exception:
        m = _re.search(r"\{.*\}", text, _re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _start_ollama_and_pull(model: str):
    import subprocess
    import time as _t
    import urllib.request
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(60):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags")
            break
        except Exception:
            _t.sleep(1)
    else:
        raise RuntimeError("ollama failed to start")
    subprocess.run(["ollama", "pull", model], check=True, timeout=1800)


def _vol_path(arm: str) -> str:
    return f"/results/raw/{arm}_qwen3_8b_physician.jsonl"


def _load_done(vol_path: str) -> set[int]:
    done = set()
    if os.path.exists(vol_path):
        with open(vol_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(int(r["case_idx"]))
                except Exception:
                    continue
    return done


def _append(vol_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(vol_path), exist_ok=True)
    with open(vol_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


@app.function(image=ollama_image, gpu="A10G", timeout=43200, volumes={"/results": volume})
def run_m4_cot(n_max: int | None = None):
    """Run Qwen3-8B Arm C with CoT preamble. n_max for pilot, None for full 450."""
    _start_ollama_and_pull("qwen3:8b")
    from rlm import RLM

    sys_prompt = COT_PREAMBLE + RLM_SYSTEM_ARM_C
    # No .format() call here — RLM library handles it internally

    with open("/results/data/physician_full.json") as f:
        cases = json.load(f)
    if n_max:
        cases = cases[:n_max]

    vol_path = _vol_path("M4_C_cotPreambleV2")
    done = _load_done(vol_path)
    print(f"M4 v2 Qwen3-8B Arm C CoT preamble: {len(done)}/{len(cases)} done")

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": "qwen3:8b",
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
        },
        environment="local",
        max_depth=1,
        max_iterations=8,
        custom_system_prompt=sys_prompt,
        verbose=False,
    )

    for i, case in enumerate(cases):
        if i in done:
            continue
        message = case.get("prompt", case.get("message", ""))
        t0 = time.time()
        try:
            result = rlm.completion(message)
            raw_text = result if isinstance(result, str) else getattr(result, "response", str(result))
        except Exception:
            raw_text = f"ERROR: {traceback.format_exc()}"
        elapsed = time.time() - t0
        row = {
            "case_idx": i,
            "case_name": case.get("name", case.get("case_label", f"case_{i}")),
            "model": "qwen3:8b",
            "arm": "M4_C_cotPreambleV2",
            "raw_response": raw_text,
            "parsed": _parse_json_response(raw_text),
            "elapsed_sec": round(elapsed, 2),
        }
        _append(vol_path, row)
        volume.commit()
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(cases)} ({elapsed:.1f}s)")


@app.local_entrypoint()
def pilot():
    """5-case pilot to verify the fresh app works."""
    try:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(
                os.environ.get("RLM_PHYSICIAN_DATA", str(Path(__file__).resolve().parents[2] / "data" / "physician_full.json")),
                "data/physician_full.json",
            )
    except Exception:
        pass
    run_m4_cot.remote(n_max=5)


@app.local_entrypoint()
def full():
    """Full N=450 after pilot validates."""
    try:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(
                os.environ.get("RLM_PHYSICIAN_DATA", str(Path(__file__).resolve().parents[2] / "data" / "physician_full.json")),
                "data/physician_full.json",
            )
    except Exception:
        pass
    run_m4_cot.remote()


@app.local_entrypoint()
def download():
    local_dir = Path(os.environ.get("RLM_REVISION_DIR", str(Path(__file__).resolve().parents[1]))) / "frontier_runs"
    local_dir.mkdir(parents=True, exist_ok=True)
    fname = "M4_C_cotPreambleV2_qwen3_8b_physician.jsonl"
    target = local_dir / fname
    data = b""
    for chunk in volume.read_file(f"raw/{fname}"):
        data += chunk
    target.write_bytes(data)
    n = data.decode().count("\n")
    print(f"  {fname}: {n} cases")

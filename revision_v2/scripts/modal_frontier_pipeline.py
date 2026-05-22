"""Modal pipeline for the frontier API experiments (Arms A, B, C, E; M2 and M5 variants).

Three providers (Anthropic Claude, OpenAI Responses-API reasoning, Google Gen AI).
Each (provider, arm) runs as a separate Modal function with volume-mounted JSONL
checkpoints (written after every case, atomic), a 600 s per-case wall-clock cap, a
12 h function-level timeout, idempotent resume, and a per-function spend ceiling.

API keys are passed via Modal Secrets (anthropic, openai, google); the script never
embeds them.

Launch:
    cd revision_v2/scripts
    modal run --detach modal_frontier_pipeline.py::pilot
    modal run --detach modal_frontier_pipeline.py::run_all
    modal run --detach modal_frontier_pipeline.py::run_m2_thinking_off
    modal run --detach modal_frontier_pipeline.py::run_m5_grounding_stripped
    modal run modal_frontier_pipeline.py::download
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import modal

app = modal.App("rlm-frontier-revision")

# Volume: persistent storage for checkpoints and final outputs
volume = modal.Volume.from_name("rlm-frontier-results", create_if_missing=True)

# Secrets: use the user's existing per-provider secrets (anthropic, openai, google).
# Each is expected to export the standard env var name for its respective SDK.
api_secrets = [
    modal.Secret.from_name("anthropic"),  # ANTHROPIC_API_KEY
    modal.Secret.from_name("openai"),     # OPENAI_API_KEY
    modal.Secret.from_name("google"),     # GOOGLE_API_KEY or GEMINI_API_KEY
]

# Image: Python 3.12 + provider SDKs + our local scripts
# NOTE: Cache-buster v3 — force image rebuild after M2/M5 bug fixes (2026-05-21).
_IMAGE_REV = "v3-2026-05-21-fix-import-re-disable-thinking"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "anthropic>=0.40.0",
        "openai>=1.50.0",
        "google-genai>=1.0.0",            # New SDK with thinking_config support
        "google-generativeai>=0.8.0",     # Legacy fallback (kept for safety)
        "numpy>=1.24.0",
        "scipy>=1.11.0",
    )
    .env({"_IMAGE_REV": _IMAGE_REV})
    .add_local_dir(
        ".",
        "/app/scripts",
        ignore=["__pycache__", "*.pyc"],
        copy=True,
    )
    .add_local_dir(
        os.environ.get("RLM_PACKAGING_DIR", str(Path(__file__).resolve().parents[2])),
        "/app/packaging",
        ignore=["output", "logs", "__pycache__", "*.pyc"],
        copy=True,
    )
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDER_BY_MODEL = {
    "claude-opus-4-7": "anthropic",
    "gpt-5.5": "openai",
    "gemini-3.1-pro-preview": "gemini",
}

# Per-case API cost guard ceilings (USD) per function invocation, defensive
COST_CEILING_PER_FUNCTION = {
    "claude-opus-4-7": 200.0,
    "gpt-5.5": 150.0,
    "gemini-3.1-pro-preview": 100.0,
}

# Pricing per 1M tokens (USD) — used for the in-loop spend tracker
PRICING = {
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    "gpt-5.5": {"input": 5.0, "output": 20.0},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 5.0},
}


# ---------------------------------------------------------------------------
# Volume I/O helpers
# ---------------------------------------------------------------------------

def _vol_path(model: str, arm: str, dataset: str) -> str:
    safe = model.replace("/", "_").replace(":", "_")
    return f"/results/raw/{arm}_frontier_{safe}_{dataset}.jsonl"


def load_done_indices(vol_path: str) -> set[int]:
    """Read the checkpoint JSONL and return the set of case_idx already completed."""
    done = set()
    if not os.path.exists(vol_path):
        return done
    try:
        with open(vol_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(int(r["case_idx"]))
                except Exception:
                    continue
    except Exception:
        pass
    return done


def append_row_atomic(vol_path: str, row: dict) -> None:
    """Append a row to the JSONL checkpoint with an atomic flush."""
    os.makedirs(os.path.dirname(vol_path), exist_ok=True)
    with open(vol_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Data upload (run once locally before first execution)
# ---------------------------------------------------------------------------

def _upload_data():
    """Upload physician_full.json to the volume."""
    data_path = Path(os.environ.get("RLM_PHYSICIAN_DATA", str(Path(__file__).resolve().parents[2] / "data" / "physician_full.json")))
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(data_path), "data/physician_full.json")
    print(f"  Uploaded {data_path.name}")


# ---------------------------------------------------------------------------
# Single-case execution
# ---------------------------------------------------------------------------

def _run_single_case(
    *,
    provider: str,
    model: str,
    arm: str,
    case_idx: int,
    patient_message: str,
    case_name: str,
    timeout_s: int = 600,
) -> dict:
    """Run one case through one arm with the relevant frontier model.

    Imports happen inside the function so Modal can do module-level reflection
    without requiring the dependencies at import time.
    """
    sys.path.insert(0, "/app/scripts")
    sys.path.insert(0, "/app/packaging")
    from prompts import (  # type: ignore
        SINGLE_PASS,
        CHAIN_OF_THOUGHT,
        SINGLE_PASS_GROUNDED,
        RLM_PROMPT_NO_REPL,
        RLM_SYSTEM,
    )
    from utils import parse_json_response  # type: ignore
    from frontier_repl_harness import run_single_turn_arm, run_repl_arm  # type: ignore

    t0 = time.time()
    error = None
    row: dict[str, object] = {
        "case_idx": case_idx,
        "case_name": case_name,
        "model": model,
        "provider": provider,
        "arm": f"{arm}_frontier",
    }
    try:
        if arm == "A":
            prompt = SINGLE_PASS.format(message=patient_message)
            res = run_single_turn_arm(provider=provider, model=model, system_prompt=None, user_prompt=prompt)
            row.update({"raw_response": res["text"], "parsed": parse_json_response(res["text"]),
                         "usage": res["usage"], "error": res.get("error")})
        elif arm == "B":
            prompt = CHAIN_OF_THOUGHT.format(message=patient_message)
            res = run_single_turn_arm(provider=provider, model=model, system_prompt=None, user_prompt=prompt)
            row.update({"raw_response": res["text"], "parsed": parse_json_response(res["text"]),
                         "usage": res["usage"], "error": res.get("error")})
        elif arm == "Aplus":
            prompt = SINGLE_PASS_GROUNDED.format(message=patient_message)
            res = run_single_turn_arm(provider=provider, model=model, system_prompt=None, user_prompt=prompt)
            row.update({"raw_response": res["text"], "parsed": parse_json_response(res["text"]),
                         "usage": res["usage"], "error": res.get("error")})
        elif arm == "E":
            prompt = RLM_PROMPT_NO_REPL.format(message=patient_message)
            res = run_single_turn_arm(provider=provider, model=model, system_prompt=None, user_prompt=prompt)
            row.update({"raw_response": res["text"], "parsed": parse_json_response(res["text"]),
                         "usage": res["usage"], "error": res.get("error")})
        elif arm in ("C", "C_noThink", "C_noGrounding"):
            system = RLM_SYSTEM
            system = system.replace("llm_query(prompt)", "# (sub-calls disabled in this arm)").replace(
                "You can use llm_query(prompt) to make sub-calls for clinical assessment of individual findings.\n\n",
                "",
            )
            # M5 variant: strip grounding language ("RULES" block)
            if arm == "C_noGrounding":
                # Remove the RULES block including its lines
                system_no_rules = re.sub(
                    r"RULES:\n(?:- [^\n]*\n)+\n",
                    "",
                    system,
                    flags=re.MULTILINE,
                )
                system = system_no_rules
            system = system.format(custom_tools_section="")
            disable_think = (arm == "C_noThink")
            res = run_repl_arm(
                provider=provider,
                model=model,
                system_prompt=system,
                patient_message=patient_message,
                max_iterations=8,
                allow_llm_query=False,
                disable_thinking=disable_think,
            )
            final = res.get("final_json")
            parsed = None
            if final:
                try:
                    parsed = json.loads(final)
                except Exception:
                    parsed = parse_json_response(final)
            row.update({
                "final_json": final,
                "parsed": parsed,
                "iterations": res.get("iterations"),
                "usage": res.get("usage"),
                "terminated_reason": res.get("terminated_reason"),
                "error": res.get("error"),
            })
        elif arm == "D":
            system = RLM_SYSTEM.format(custom_tools_section="")
            res = run_repl_arm(
                provider=provider,
                model=model,
                system_prompt=system,
                patient_message=patient_message,
                max_iterations=8,
                allow_llm_query=True,
            )
            final = res.get("final_json")
            parsed = None
            if final:
                try:
                    parsed = json.loads(final)
                except Exception:
                    parsed = parse_json_response(final)
            row.update({
                "final_json": final,
                "parsed": parsed,
                "iterations": res.get("iterations"),
                "usage": res.get("usage"),
                "terminated_reason": res.get("terminated_reason"),
                "error": res.get("error"),
            })
        else:
            raise ValueError(f"Unknown arm: {arm}")
    except Exception:
        error = traceback.format_exc()
        row["error"] = error
    row["elapsed_sec"] = round(time.time() - t0, 2)
    return row


def _estimated_cost_for_row(row: dict) -> float:
    usage = row.get("usage") or {}
    in_tok = usage.get("input_tokens", 0) or 0
    out_tok = usage.get("output_tokens", 0) or 0
    reasoning_tok = usage.get("reasoning_tokens", 0) or 0
    model = row.get("model", "")
    p = PRICING.get(model, {"input": 0.0, "output": 0.0})
    # Reasoning tokens are typically billed as output tokens
    return (in_tok / 1e6) * p["input"] + ((out_tok + reasoning_tok) / 1e6) * p["output"]


# ---------------------------------------------------------------------------
# Modal worker function — one per (model, arm)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    timeout=43200,  # 12 hours Modal max
    volumes={"/results": volume},
    secrets=api_secrets,
    cpu=2.0,
    memory=4096,
)
def run_model_arm(model: str, arm: str, dataset: str = "physician", n_max: int | None = None) -> dict:
    """Run one (model, arm, dataset) combination to completion, with checkpoint/resume."""
    import json as _json
    import re  # used in arm-variant prompt manipulation

    provider = PROVIDER_BY_MODEL[model]

    # Load cases from volume
    data_path = "/results/data/physician_full.json"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not at {data_path} — upload first.")
    with open(data_path) as f:
        cases = _json.load(f)
    if n_max:
        cases = cases[:n_max]

    vol_path = _vol_path(model, arm, dataset)
    done = load_done_indices(vol_path)
    pending_indices = [i for i in range(len(cases)) if i not in done]
    print(f"\n{arm}/{model}/{dataset}: {len(done)} done, {len(pending_indices)} pending")

    spend_usd = 0.0
    ceiling = COST_CEILING_PER_FUNCTION.get(model, 100.0)
    stats = {"completed": 0, "errors": 0, "timeouts": 0}

    for i_idx, case_idx in enumerate(pending_indices):
        case = cases[case_idx]
        message = case.get("prompt", case.get("message", ""))
        name = case.get("name", case.get("case_label", f"case_{case_idx}"))
        try:
            row = _run_single_case(
                provider=provider,
                model=model,
                arm=arm,
                case_idx=case_idx,
                patient_message=message,
                case_name=name,
                timeout_s=600,
            )
        except Exception:
            row = {
                "case_idx": case_idx,
                "case_name": name,
                "model": model,
                "provider": provider,
                "arm": f"{arm}_frontier",
                "error": traceback.format_exc(),
            }
            stats["errors"] += 1

        append_row_atomic(vol_path, row)
        volume.commit()
        if row.get("error"):
            stats["errors"] += 1
        else:
            stats["completed"] += 1
            cost = _estimated_cost_for_row(row)
            spend_usd += cost

        # Progress logging every 10 cases
        if (i_idx + 1) % 10 == 0 or i_idx + 1 == len(pending_indices):
            print(
                f"  [{i_idx + 1}/{len(pending_indices)}] case_idx={case_idx} "
                f"elapsed={row.get('elapsed_sec', 0):.1f}s spend=${spend_usd:.2f}"
            )

        # Cost-ceiling guard
        if spend_usd > ceiling:
            print(f"  COST CEILING ${ceiling} reached at iter {i_idx}; stopping early. Resume next invocation.")
            break

    return {
        "model": model,
        "arm": arm,
        "dataset": dataset,
        "spend_usd": round(spend_usd, 2),
        "stats": stats,
        "vol_path": vol_path,
    }


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def upload_data():
    """One-time data upload to the Modal volume."""
    _upload_data()


@app.local_entrypoint()
def pilot():
    """Pilot run: n=20 stratified cases, all 3 models, Arm A only.

    Use this to validate API connectivity, prompt parsing, and parsing logic before
    committing to the full ~$300-1500 spend.
    """
    _upload_data()
    print("\n=== PILOT: 3 models × Arm A × n=20 cases ===\n")
    handles = []
    for model in PROVIDER_BY_MODEL:
        h = run_model_arm.spawn(model, "A", "physician", n_max=20)
        handles.append((model, h))
    for model, h in handles:
        result = h.get()
        print(f"PILOT {model}: {result}")


@app.local_entrypoint()
def run_all():
    """Full frontier-model run: 3 models × Arms A, B, C, E × N=450.

    Resume-safe: re-running picks up where prior invocations left off.
    """
    _upload_data()
    futures = []
    # Sequencing: do cheap arms first so we have results even if expensive arms hit the cost ceiling
    arm_order = ["A", "E", "B", "C"]
    for arm in arm_order:
        for model in PROVIDER_BY_MODEL:
            label = f"{arm}/{model}"
            h = run_model_arm.spawn(model, arm, "physician")
            futures.append((label, h))
    print(f"\nLaunched {len(futures)} jobs. Tracking spend per worker.\n")
    for label, h in futures:
        try:
            r = h.get()
            print(f"DONE {label}: spend=${r['spend_usd']} stats={r['stats']}")
        except Exception as e:
            print(f"FAILED {label}: {e}")


@app.local_entrypoint()
def run_m2_thinking_off():
    """M2: Thinking-off ablation on Arm C for Claude Opus 4.7 and Gemini 3.1 Pro Preview.

    Tests sub-mechanism #2 (extended-thinking pre-commitment). If disabling thinking
    causes frontier Arm C PSR to rise sharply, thinking is causal. If PSR stays low,
    thinking is not necessary.

    GPT-5.5 is excluded from M2 because its reasoning mode cannot be fully disabled
    without using a non-reasoning model variant (which would change other factors).
    """
    _upload_data()
    futures = []
    for model in ("claude-opus-4-7", "gemini-3.1-pro-preview"):
        label = f"M2_C_noThink/{model}"
        h = run_model_arm.spawn(model, "C_noThink", "physician")
        futures.append((label, h))
    for label, h in futures:
        try:
            r = h.get()
            print(f"DONE {label}: spend=${r['spend_usd']} stats={r['stats']}")
        except Exception as e:
            print(f"FAILED {label}: {e}")


@app.local_entrypoint()
def run_m5_grounding_stripped():
    """M5: Grounding-stripped Arm C for all three frontier models.

    Tests sub-mechanism #3 (RLHF source-grounding alignment). If removing the
    grounding RULES from the system prompt causes frontier Arm C PSR to rise, the
    prompt is doing the work. If PSR stays low, frontier models maintain grounding
    intrinsically (which would suggest mechanism 1 — coding training depth — or
    mechanism 2 — thinking — is dominant).
    """
    _upload_data()
    futures = []
    for model in PROVIDER_BY_MODEL:
        label = f"M5_C_noGrounding/{model}"
        h = run_model_arm.spawn(model, "C_noGrounding", "physician")
        futures.append((label, h))
    for label, h in futures:
        try:
            r = h.get()
            print(f"DONE {label}: spend=${r['spend_usd']} stats={r['stats']}")
        except Exception as e:
            print(f"FAILED {label}: {e}")


@app.local_entrypoint()
def download():
    """Download all JSONL outputs from the Modal volume to local frontier_runs/."""
    local_dir = Path(__file__).resolve().parents[1] / "frontier_runs"
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {local_dir}")
    try:
        entries = list(volume.listdir("raw"))
    except Exception:
        print("No outputs yet.")
        return
    for entry in entries:
        if not entry.path.endswith(".jsonl"):
            continue
        fname = Path(entry.path).name
        target = local_dir / fname
        data = b""
        for chunk in volume.read_file(f"raw/{fname}"):
            data += chunk
        target.write_bytes(data)
        n = data.decode().count("\n")
        print(f"  {fname}: {n} cases")
    print("Done.")


@app.local_entrypoint()
def status():
    """Print per-(model, arm) progress from the volume without downloading."""
    try:
        entries = list(volume.listdir("raw"))
    except Exception:
        print("No outputs yet.")
        return
    print(f"\n{'File':<70} {'Cases':>6}")
    print("-" * 80)
    for entry in sorted(entries, key=lambda x: x.path):
        if not entry.path.endswith(".jsonl"):
            continue
        fname = Path(entry.path).name
        data = b""
        for chunk in volume.read_file(f"raw/{fname}"):
            data += chunk
        n = data.decode().count("\n")
        print(f"{fname:<70} {n:>6}")

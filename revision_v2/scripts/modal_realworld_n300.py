"""Frontier-model Arms A and C on a stratified N=300 subset of the real-world set.

Runs Claude Opus 4.7, GPT-5.5, and Gemini 3.1 Pro Preview on the same Arms A and C
prompts used for the physician set, against a stratified random N=300 subsample of
realworld_test.json (seed=42, hazard-stratified at 50 hazard / 250 benign for
adequate sensitivity power).

Output: revision_v2/frontier_runs/{A,C}_frontier_<model>_realworld_n300.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import modal


app = modal.App("rlm-frontier-realworld-n300")
volume = modal.Volume.from_name("rlm-frontier-results", create_if_missing=True)

_IMAGE_REV = "realworld-n300-2026-05-22-v3-fixreplreturn"

def _build_image():
    here = Path(__file__).resolve().parent
    packaging = Path(os.environ.get("RLM_PACKAGING_DIR", str(Path(__file__).resolve().parents[2])))
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "anthropic>=0.40.0",
            "openai>=1.50.0",
            "google-genai>=0.3.0",
            "google-generativeai>=0.8.0",
            "numpy>=1.24.0",
            "scipy>=1.11.0",
        )
        .env({"_IMAGE_REV": _IMAGE_REV})
        .add_local_dir(str(here), "/app/scripts", ignore=["__pycache__", "*.pyc"], copy=True)
        .add_local_dir(
            str(packaging),
            "/app/packaging",
            ignore=["output", "logs", "__pycache__", "*.pyc"],
            copy=True,
        )
    )


try:
    image = _build_image()
except Exception:
    image = modal.Image.debian_slim(python_version="3.12")


api_secrets = [
    modal.Secret.from_name("anthropic"),
    modal.Secret.from_name("openai"),
    modal.Secret.from_name("google"),
]


PROVIDER_BY_MODEL = {
    "claude-opus-4-7": "anthropic",
    "gpt-5.5": "openai",
    "gemini-3.1-pro-preview": "gemini",
}

COST_CEILING_PER_FUNCTION = {
    "claude-opus-4-7": 200.0,
    "gpt-5.5": 100.0,
    "gemini-3.1-pro-preview": 75.0,
}

PRICING = {
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    "gpt-5.5": {"input": 5.0, "output": 20.0},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 5.0},
}


def _vol_path(model: str, arm: str) -> str:
    safe = model.replace("/", "_").replace(":", "_")
    return f"/results/raw/{arm}_frontier_{safe}_realworld_n300.jsonl"


def load_done_indices(vol_path: str) -> set[int]:
    done: set[int] = set()
    if not os.path.exists(vol_path):
        return done
    with open(vol_path) as f:
        for line in f:
            try:
                done.add(int(json.loads(line)["case_idx"]))
            except Exception:
                continue
    return done


def append_row_atomic(vol_path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(vol_path), exist_ok=True)
    with open(vol_path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _stratified_sample(cases: list[dict], n_hazard: int = 50, n_benign: int = 250, seed: int = 42) -> list[tuple[int, dict]]:
    """Return n_hazard + n_benign (original_idx, case) pairs, stratified on detection_truth."""
    hazards = [(i, c) for i, c in enumerate(cases) if int(c.get("detection_truth", 0)) == 1]
    benigns = [(i, c) for i, c in enumerate(cases) if int(c.get("detection_truth", 0)) == 0]
    rng = random.Random(seed)
    rng.shuffle(hazards)
    rng.shuffle(benigns)
    return hazards[:n_hazard] + benigns[:n_benign]


def _upload_data():
    src = Path(os.environ.get(
        "RLM_REALWORLD_DATA",
        os.environ.get("RLM_REALWORLD_DATA", str(Path(__file__).resolve().parents[2] / "data" / "realworld_test.json")),
    ))
    with open(src) as f:
        cases = json.load(f)
    subset = _stratified_sample(cases)
    out = []
    for original_idx, case in subset:
        case = dict(case)
        case["_original_idx"] = original_idx
        out.append(case)
    print(f"Stratified subset: N={len(out)} ({sum(1 for c in out if int(c.get('detection_truth', 0)) == 1)} hazard / {sum(1 for c in out if int(c.get('detection_truth', 0)) == 0)} benign)")
    tmp = Path("/tmp/realworld_n300.json")
    tmp.write_text(json.dumps(out, default=str))
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(tmp), "data/realworld_n300.json")
    print(f"Uploaded to volume: /results/data/realworld_n300.json")


def _run_single_case(*, provider: str, model: str, arm: str, case_idx: int,
                     patient_message: str, case_name: str, timeout_s: int = 600) -> dict:
    sys.path.insert(0, "/app/scripts")
    sys.path.insert(0, "/app/packaging")
    from prompts import SINGLE_PASS, RLM_SYSTEM  # type: ignore
    from utils import parse_json_response  # type: ignore
    from frontier_repl_harness import run_single_turn_arm, run_repl_arm  # type: ignore

    t0 = time.time()
    row: dict[str, object] = {
        "case_idx": case_idx, "case_name": case_name,
        "model": model, "provider": provider, "arm": f"{arm}_frontier",
    }
    try:
        if arm == "A":
            prompt = SINGLE_PASS.format(message=patient_message)
            res = run_single_turn_arm(provider=provider, model=model,
                                      system_prompt=None, user_prompt=prompt)
            row.update({"raw_response": res["text"], "parsed": parse_json_response(res["text"]),
                        "usage": res["usage"], "error": res.get("error")})
        elif arm == "C":
            res = run_repl_arm(provider=provider, model=model,
                               system_prompt=RLM_SYSTEM.format(custom_tools_section=""),
                               patient_message=patient_message, max_iterations=8)
            final_json = res.get("final_json")
            raw_text = json.dumps(final_json, default=str) if isinstance(final_json, dict) else (final_json or "")
            row.update({"raw_response": raw_text,
                        "parsed": final_json if isinstance(final_json, dict) else parse_json_response(raw_text),
                        "usage": res.get("usage", {}),
                        "error": res.get("error"),
                        "n_iterations": res.get("iterations"),
                        "terminated_reason": res.get("terminated_reason")})
        else:
            row["error"] = f"unknown arm {arm}"
    except Exception:
        row["error"] = traceback.format_exc()
    row["elapsed_sec"] = round(time.time() - t0, 2)
    return row


def _estimated_cost_for_row(row: dict) -> float:
    if row.get("error"):
        return 0.0
    usage = row.get("usage") or {}
    model = row.get("model", "")
    pricing = PRICING.get(model, {"input": 1.0, "output": 1.0})
    in_tok = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
    out_tok = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0
    return (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000


@app.function(
    image=image,
    timeout=43200,
    volumes={"/results": volume},
    secrets=api_secrets,
    cpu=2.0,
    memory=4096,
)
def run_model_arm(model: str, arm: str) -> dict:
    data_path = "/results/data/realworld_n300.json"
    if not os.path.exists(data_path):
        raise FileNotFoundError("realworld_n300.json not on volume — call upload first")
    with open(data_path) as f:
        cases = json.load(f)
    vol_path = _vol_path(model, arm)
    done = load_done_indices(vol_path)
    pending = [i for i in range(len(cases)) if i not in done]
    print(f"{arm}/{model}: {len(done)} done, {len(pending)} pending")

    spend = 0.0
    ceiling = COST_CEILING_PER_FUNCTION.get(model, 100.0)
    stats = {"completed": 0, "errors": 0}

    for i_idx, case_idx in enumerate(pending):
        case = cases[case_idx]
        message = case.get("prompt", case.get("message", ""))
        name = case.get("case_label", case.get("message_id", f"case_{case_idx}"))
        try:
            row = _run_single_case(provider=PROVIDER_BY_MODEL[model], model=model, arm=arm,
                                   case_idx=case_idx, patient_message=message, case_name=name)
        except Exception:
            row = {"case_idx": case_idx, "case_name": name, "model": model,
                   "arm": f"{arm}_frontier", "error": traceback.format_exc()}
        append_row_atomic(vol_path, row)
        volume.commit()
        if row.get("error"):
            stats["errors"] += 1
        else:
            stats["completed"] += 1
            spend += _estimated_cost_for_row(row)
        if (i_idx + 1) % 10 == 0 or i_idx + 1 == len(pending):
            print(f"  [{i_idx+1}/{len(pending)}] case={case_idx} spend=${spend:.2f}")
        if spend > ceiling:
            print(f"  cost ceiling ${ceiling} reached at iter {i_idx}; stopping")
            break

    return {"model": model, "arm": arm, "spend_usd": round(spend, 2), "stats": stats, "vol_path": vol_path}


@app.local_entrypoint()
def upload():
    _upload_data()


@app.local_entrypoint()
def run_all():
    _upload_data()
    handles = []
    for arm in ("A", "C"):
        for model in PROVIDER_BY_MODEL:
            h = run_model_arm.spawn(model, arm)
            handles.append((f"{arm}/{model}", h))
    for label, h in handles:
        try:
            r = h.get()
            print(f"DONE {label}: spend=${r['spend_usd']} stats={r['stats']}")
        except Exception as e:
            print(f"FAILED {label}: {e}")


@app.local_entrypoint()
def download():
    local_dir = Path(__file__).resolve().parents[1] / "frontier_runs"
    local_dir.mkdir(parents=True, exist_ok=True)
    for arm in ("A", "C"):
        for model in PROVIDER_BY_MODEL:
            safe = model.replace("/", "_").replace(":", "_")
            fname = f"{arm}_frontier_{safe}_realworld_n300.jsonl"
            target = local_dir / fname
            try:
                with volume.open(f"raw/{fname}", "rb") as f:
                    target.write_bytes(f.read())
                print(f"  downloaded {fname} ({target.stat().st_size:,} bytes)")
            except Exception as e:
                print(f"  skip {fname}: {e}")

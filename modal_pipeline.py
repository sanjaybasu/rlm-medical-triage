"""Modal cloud execution for RLM medical triage experiments.

Runs ollama with multiple model sizes on GPU, executes
baseline and RLM arms in parallel across multiple containers.

Models:
  - llama3.1:8b, qwen3:8b         -> A10G (24GB)
  - qwen3:32b                      -> A100 (40GB)
  - deepseek-r1:70b                -> A100-80GB

Usage:
    cd packaging/rlm-medical-triage

    # Launch ALL experiments (4 models, 4 arms, 2 datasets, 3 GPU tiers):
    modal run --detach modal_pipeline.py::run_all

    # Launch expanded experiments (qwen3:32b + deepseek-r1:70b only):
    modal run --detach modal_pipeline.py::run_expanded

    # Complete remaining 8B experiments:
    modal run --detach modal_pipeline.py::run_remaining_8b

    # Download results to local output/raw/:
    modal run modal_pipeline.py::download

GPU tiers:
  - A10G:      8B models (llama3.1:8b, qwen3:8b)
  - A100 40GB: 32B models (qwen3:32b)
  - A100 80GB: 70B models (deepseek-r1:70b) — baselines + RLM arms
"""

import modal
import json
import os
from pathlib import Path

app = modal.App("rlm-medical-triage")

volume = modal.Volume.from_name("rlm-triage-results", create_if_missing=True)

# Image with ollama + Python deps + rlm library + experiment code
ollama_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "procps", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install(
        "openai>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "rlms>=0.1.1",
        "json5",
        "statsmodels",
    )
    .add_local_dir(
        ".",
        "/app",
        ignore=["output", "logs", "__pycache__", "*.pyc", ".git", "run_remaining*"],
        copy=True,
    )
    # Cap RLM LLM calls to 2048 tokens to prevent DeepSeek-R1-70B from
    # generating unbounded thinking chains (>10K tokens/call) in REPL iterations.
    .run_commands("python3 /app/patch_rlm.py")
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _start_ollama_and_pull(model: str):
    """Start ollama server and pull the specified model."""
    import subprocess
    import time
    import urllib.request

    subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for _ in range(60):
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags")
            break
        except Exception:
            time.sleep(1)
    else:
        raise RuntimeError("ollama failed to start")

    print(f"Pulling {model}...")
    subprocess.run(["ollama", "pull", model], check=True, timeout=1800)
    print(f"  {model} ready")


def _load_cases(dataset: str, arm_name: str = ""):
    """Load test cases from volume."""
    is_rlm = arm_name.startswith(("C_", "D_", "Cp_", "Dp_"))
    if dataset == "realworld" and is_rlm:
        path = "/results/data/realworld_rlm_subsample.json"
    elif dataset == "realworld":
        path = "/results/data/realworld_full.json"
    else:
        path = "/results/data/physician_full.json"
    with open(path) as f:
        return json.load(f)


def _load_existing(filename: str) -> dict:
    """Load existing results from volume for checkpoint/resume."""
    import os
    existing = {}
    vol_path = f"/results/raw/{filename}"
    if os.path.exists(vol_path):
        with open(vol_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    existing[r["case_idx"]] = r
                except Exception:
                    pass
    return existing


def _save_jsonl(filename: str, results: list):
    """Save results as JSONL to volume."""
    import os
    out_dir = "/results/raw"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/{filename}"
    with open(path, "w") as f:
        for r in sorted(results, key=lambda x: x["case_idx"]):
            f.write(json.dumps(r) + "\n")


def _run_cot_impl(model: str, dataset: str):
    """Run only Chain-of-Thought (Arm B) — used to fill B realworld gaps quickly."""
    import sys
    import time
    sys.path.insert(0, "/app")

    _start_ollama_and_pull(model)

    from openai import OpenAI
    from prompts import CHAIN_OF_THOUGHT
    from utils import parse_json_response

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    cases = _load_cases(dataset)

    arm_name = "B_chain_of_thought"
    filename = f"{arm_name}_{model.replace(':', '_')}_{dataset}.jsonl"
    existing = _load_existing(filename)
    results = list(existing.values())
    print(f"\n{arm_name}/{model}/{dataset}: {len(existing)}/{len(cases)} done, resuming...")

    for i, case in enumerate(cases):
        if i in existing:
            continue
        message = case.get("prompt", case.get("message", ""))
        prompt = CHAIN_OF_THOUGHT.format(message=message)

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
            )
            raw_text = resp.choices[0].message.content or ""
        except Exception as e:
            raw_text = f"ERROR: {e}"
        elapsed = time.time() - t0

        results.append({
            "case_idx": i,
            "case_name": case.get("name", case.get("case_label", f"case_{i}")),
            "model": model,
            "arm": arm_name,
            "raw_response": raw_text,
            "parsed": parse_json_response(raw_text),
            "elapsed_sec": round(elapsed, 2),
        })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(cases)} ({elapsed:.1f}s)")
            _save_jsonl(filename, results)
            volume.commit()

    _save_jsonl(filename, results)
    volume.commit()
    print(f"  {arm_name}/{model}/{dataset}: DONE ({len(results)})")


def _run_baselines_impl(model: str, dataset: str):
    """Shared implementation for baseline arms A and B."""
    import sys
    import time
    sys.path.insert(0, "/app")

    _start_ollama_and_pull(model)

    from openai import OpenAI
    from prompts import SINGLE_PASS, CHAIN_OF_THOUGHT, SINGLE_PASS_GROUNDED, RLM_PROMPT_NO_REPL
    from utils import parse_json_response

    client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    cases = _load_cases(dataset)

    for arm_name, prompt_tpl in [
        ("A_single_pass", SINGLE_PASS),
        ("Aplus_grounded", SINGLE_PASS_GROUNDED),
        ("B_chain_of_thought", CHAIN_OF_THOUGHT),
        ("E_rlm_prompt_no_repl", RLM_PROMPT_NO_REPL),
    ]:
        filename = f"{arm_name}_{model.replace(':', '_')}_{dataset}.jsonl"
        existing = _load_existing(filename)
        results = list(existing.values())
        print(f"\n{arm_name}/{model}/{dataset}: {len(existing)}/{len(cases)} done, resuming...")

        for i, case in enumerate(cases):
            if i in existing:
                continue
            message = case.get("prompt", case.get("message", ""))
            prompt = prompt_tpl.format(message=message)

            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096,
                )
                raw_text = resp.choices[0].message.content or ""
            except Exception as e:
                raw_text = f"ERROR: {e}"
            elapsed = time.time() - t0

            results.append({
                "case_idx": i,
                "case_name": case.get("name", case.get("case_label", f"case_{i}")),
                "model": model,
                "arm": arm_name,
                "raw_response": raw_text,
                "parsed": parse_json_response(raw_text),
                "elapsed_sec": round(elapsed, 2),
            })

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(cases)} ({elapsed:.1f}s)")
                _save_jsonl(filename, results)
                volume.commit()

        _save_jsonl(filename, results)
        volume.commit()
        print(f"  {arm_name}/{model}/{dataset}: DONE ({len(results)})")


def _run_rlm_impl(model: str, arm_name: str, dataset: str):
    """Shared implementation for RLM arms C and D (fair and prescriptive)."""
    import sys
    import time
    import traceback
    import concurrent.futures
    sys.path.insert(0, "/app")

    _start_ollama_and_pull(model)

    from rlm import RLM
    from prompts import RLM_SYSTEM, RLM_SYSTEM_PRESCRIPTIVE
    from utils import parse_json_response

    ARMS = {
        "C_repl_only": {"max_depth": 1, "max_iterations": 8},
        "D_rlm_full": {"max_depth": 3, "max_iterations": 8},
        "Cp_repl_only_prescriptive": {"max_depth": 1, "max_iterations": 8},
        "Dp_rlm_full_prescriptive": {"max_depth": 3, "max_iterations": 8},
    }
    arm_cfg = ARMS[arm_name]
    cases = _load_cases(dataset, arm_name)

    # Select prompt: prescriptive (old hardcoded regex) or fair (autonomous)
    if arm_name.startswith(("Cp_", "Dp_")):
        sys_prompt = RLM_SYSTEM_PRESCRIPTIVE
    else:
        sys_prompt = RLM_SYSTEM

    # Suppress DeepSeek-R1's extended chain-of-thought reasoning in REPL iterations.
    # Without /no_think, DeepSeek generates thousands of tokens of <think> per call,
    # exhausting context and causing httpx connection failures.
    if "deepseek" in model.lower():
        sys_prompt = "/no_think\n\n" + sys_prompt

    # For shallow arms (C variants), remove sub-call references
    if arm_name in ("C_repl_only", "Cp_repl_only_prescriptive"):
        sys_prompt = sys_prompt.replace(
            "llm_query(prompt)", "# (sub-calls disabled)"
        ).replace(
            "You can use llm_query(prompt) to make sub-calls for clinical assessment of individual findings.\n\n",
            ""
        )

    rlm_instance = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": model,
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
        },
        environment="local",
        max_depth=arm_cfg["max_depth"],
        max_iterations=arm_cfg["max_iterations"],
        custom_system_prompt=sys_prompt,
        verbose=False,
    )

    filename = f"{arm_name}_{model.replace(':', '_')}_{dataset}.jsonl"
    existing = _load_existing(filename)
    results = list(existing.values())
    print(f"\n{arm_name}/{model}/{dataset}: {len(existing)}/{len(cases)} done, resuming...")

    for i, case in enumerate(cases):
        if i in existing:
            continue
        message = case.get("prompt", case.get("message", ""))

        t0 = time.time()
        raw_text = ""
        # Per-case timeout: max 8 min (DeepSeek REPL can hang on
        # complex cases even with max_tokens cap on individual calls).
        _CASE_TIMEOUT = 480
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                _fut = _ex.submit(rlm_instance.completion, message)
                try:
                    result = _fut.result(timeout=_CASE_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    result = None
                    raw_text = f"ERROR: case timed out after {_CASE_TIMEOUT}s"
            if raw_text == "":
                if isinstance(result, str):
                    raw_text = result
                elif hasattr(result, "response"):
                    raw_text = result.response or ""
                else:
                    raw_text = str(result)
        except Exception:
            raw_text = f"ERROR: {traceback.format_exc()}"
        elapsed = time.time() - t0

        results.append({
            "case_idx": i,
            "case_name": case.get("name", case.get("case_label", f"case_{i}")),
            "model": model,
            "arm": arm_name,
            "raw_response": raw_text,
            "parsed": parse_json_response(raw_text),
            "elapsed_sec": round(elapsed, 2),
        })

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(cases)} ({elapsed:.1f}s)")
            _save_jsonl(filename, results)
            volume.commit()

    _save_jsonl(filename, results)
    volume.commit()
    print(f"  {arm_name}/{model}/{dataset}: DONE ({len(results)})")


# ---------------------------------------------------------------------------
# GPU-tiered Modal functions
# ---------------------------------------------------------------------------

# --- A10G tier (8B models) ---
@app.function(image=ollama_image, gpu="A10G", timeout=14400, volumes={"/results": volume})
def baselines_a10g(model: str, dataset: str):
    _run_baselines_impl(model, dataset)

@app.function(image=ollama_image, gpu="A10G", timeout=43200, volumes={"/results": volume})
def rlm_a10g(model: str, arm_name: str, dataset: str):
    _run_rlm_impl(model, arm_name, dataset)

# --- A100 40GB tier (32B models) ---
@app.function(image=ollama_image, gpu="A100", timeout=21600, volumes={"/results": volume})
def baselines_a100(model: str, dataset: str):
    _run_baselines_impl(model, dataset)

@app.function(image=ollama_image, gpu="A100", timeout=43200, volumes={"/results": volume})
def rlm_a100(model: str, arm_name: str, dataset: str):
    _run_rlm_impl(model, arm_name, dataset)

# --- A100 80GB tier (70B models) ---
@app.function(image=ollama_image, gpu="A100-80GB", timeout=43200, volumes={"/results": volume})
def baselines_a100_80(model: str, dataset: str):
    _run_baselines_impl(model, dataset)

@app.function(image=ollama_image, gpu="A100-80GB", timeout=86400, volumes={"/results": volume})
def rlm_a100_80(model: str, arm_name: str, dataset: str):
    _run_rlm_impl(model, arm_name, dataset)

# CoT-only fast-fill entrypoints (skip A/A+/E, run only Arm B)
@app.function(image=ollama_image, gpu="A100", timeout=43200, volumes={"/results": volume})
def cot_only_a100(model: str, dataset: str):
    _run_cot_impl(model, dataset)

@app.function(image=ollama_image, gpu="A100-80GB", timeout=43200, volumes={"/results": volume})
def cot_only_a100_80(model: str, dataset: str):
    _run_cot_impl(model, dataset)


# ---------------------------------------------------------------------------
# Wave 2: Trajectory generation
# ---------------------------------------------------------------------------

def _load_train_cases():
    """Load training cases from volume."""
    import os
    path = "/results/data/combined_train.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training data not found at {path}. Run _upload_data() first.")
    with open(path) as f:
        return json.load(f)


def _run_trajectories_impl(model: str):
    """Generate RLM trajectories from training data for LoRA fine-tuning.

    Runs the fair-prompt D arm (full RLM, max_depth=3) on all training cases,
    records full conversation trajectories, labels correct/incorrect, and
    creates a filtered JSONL of correct trajectories for fine-tuning.
    """
    import sys
    import time
    import traceback
    sys.path.insert(0, "/app")
    import os

    _start_ollama_and_pull(model)

    from rlm import RLM
    from prompts import RLM_SYSTEM
    from utils import parse_json_response, extract_detection_action

    model_slug = model.replace(":", "_")
    traj_dir = "/results/trajectories"
    os.makedirs(traj_dir, exist_ok=True)

    out_path = f"{traj_dir}/{model_slug}_trajectories.jsonl"
    filtered_path = f"{traj_dir}/{model_slug}_train_filtered.jsonl"

    # Checkpoint/resume
    done_indices = set()
    results = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_indices.add(r["case_idx"])
                    results.append(r)
                except Exception:
                    pass

    cases = _load_train_cases()
    print(f"\nTrajectory gen/{model}: {len(done_indices)}/{len(cases)} done, resuming...")

    rlm_instance = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": model,
            "api_key": "ollama",
            "base_url": "http://localhost:11434/v1",
        },
        environment="local",
        max_depth=3,
        max_iterations=8,
        custom_system_prompt=RLM_SYSTEM,
        verbose=False,
    )

    with open(out_path, "a") as fout:
        for i, case in enumerate(cases):
            if i in done_indices:
                continue
            message = case.get("prompt", case.get("message", ""))
            t0 = time.time()
            raw_text = ""
            try:
                result = rlm_instance.completion(message)
                if isinstance(result, str):
                    raw_text = result
                elif hasattr(result, "response"):
                    raw_text = result.response or ""
                else:
                    raw_text = str(result)
                elapsed = time.time() - t0

                parsed = parse_json_response(raw_text)
                det_pred, act_pred = extract_detection_action(parsed) if parsed else (0, 0)
                det_truth = int(case.get("detection_truth", 0))
                act_truth_str = case.get("action_truth", "None")
                from utils import ACTION_MAP
                act_truth = ACTION_MAP.get(act_truth_str, 0)
                correct = (det_pred == det_truth) and (act_pred == act_truth)

                trajectory = [
                    {"role": "system", "content": RLM_SYSTEM},
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": raw_text},
                ]

                row = {
                    "case_idx": i,
                    "case_name": case.get("name", case.get("case_label", f"case_{i}")),
                    "model": model,
                    "correct": correct,
                    "detection_truth": det_truth,
                    "action_truth": act_truth_str,
                    "detection_pred": det_pred,
                    "action_pred": act_pred,
                    "trajectory": trajectory,
                    "elapsed_sec": round(elapsed, 2),
                }
            except Exception:
                elapsed = time.time() - t0
                row = {
                    "case_idx": i,
                    "case_name": case.get("name", case.get("case_label", f"case_{i}")),
                    "model": model,
                    "correct": False,
                    "error": traceback.format_exc()[:500],
                    "elapsed_sec": round(elapsed, 2),
                }

            results.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

            if (len(results) - len(done_indices)) % 10 == 0:
                n_correct = sum(1 for r in results if r.get("correct"))
                n_total = len(results)
                print(f"    {n_total}/{len(cases)}: {n_correct} correct ({n_correct/n_total*100:.1f}%)")
                volume.commit()

    # Create filtered training file (correct trajectories only)
    n_filtered = 0
    with open(filtered_path, "w") as fout:
        for row in results:
            if row.get("correct") and "trajectory" in row:
                fout.write(json.dumps({"messages": row["trajectory"]}) + "\n")
                n_filtered += 1

    volume.commit()
    n_correct = sum(1 for r in results if r.get("correct"))
    print(f"  Trajectories/{model}: {len(results)} total, {n_correct} correct ({n_correct/max(len(results),1)*100:.1f}%)")
    print(f"  Filtered training set: {n_filtered} examples -> {filtered_path}")


# GPU-tiered trajectory generation functions
@app.function(image=ollama_image, gpu="A10G", timeout=86400, volumes={"/results": volume})
def trajectory_a10g(model: str):
    _run_trajectories_impl(model)

@app.function(image=ollama_image, gpu="A100", timeout=86400, volumes={"/results": volume})
def trajectory_a100(model: str):
    _run_trajectories_impl(model)


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------

def _upload_data():
    """Upload data files and existing checkpoints to volume."""
    data_dir = Path(os.environ.get(
        'RLM_DATA_DIR',
        str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
    ))
    local_raw = Path(__file__).resolve().parent / 'output' / 'raw'

    # Clean and re-upload data files
    for subdir in ["data"]:
        try:
            for entry in volume.listdir(subdir):
                volume.remove_file(entry.path)
        except Exception:
            pass

    # Get list of existing volume files to avoid conflicts
    existing_vol = set()
    try:
        for entry in volume.listdir("raw"):
            existing_vol.add(Path(entry.path).name)
    except Exception:
        pass

    with volume.batch_upload() as batch:
        for name in ["physician_full.json", "realworld_full.json", "realworld_rlm_subsample.json",
                     "combined_train.json"]:
            local_path = data_dir / name
            if local_path.exists():
                batch.put_file(str(local_path), f"data/{name}")
                print(f"  Uploaded {name}")

        # Upload local checkpoints that don't already exist on volume
        if local_raw.exists():
            for f in local_raw.glob("*.jsonl"):
                if f.name not in existing_vol:
                    batch.put_file(str(f), f"raw/{f.name}")
                    print(f"  Uploaded checkpoint: {f.name} ({sum(1 for _ in open(f))} lines)")
                else:
                    print(f"  Skipped (exists on volume): {f.name}")


@app.local_entrypoint()
def run_expanded():
    """Launch expanded experiments: new models (qwen3:32b, deepseek-r1:70b).

    Arms A+B on both new models, Arms C+D on qwen3:32b only.
    Run with: modal run --detach modal_pipeline.py::run_expanded
    """
    print("Uploading data...")
    _upload_data()

    futures = []

    # --- Qwen3-32B: Arms A+B (baselines) ---
    for ds in ["physician", "realworld"]:
        label = f"baselines/qwen3:32b/{ds}"
        futures.append((label, baselines_a100.spawn("qwen3:32b", ds)))

    # --- Qwen3-32B: Arms C+D (RLM) ---
    for arm in ["C_repl_only", "D_rlm_full"]:
        for ds in ["physician", "realworld"]:
            label = f"{arm}/qwen3:32b/{ds}"
            futures.append((label, rlm_a100.spawn("qwen3:32b", arm, ds)))

    # --- DeepSeek-R1 70B: Arms A+B only ---
    for ds in ["physician", "realworld"]:
        label = f"baselines/deepseek-r1:70b/{ds}"
        futures.append((label, baselines_a100_80.spawn("deepseek-r1:70b", ds)))

    print(f"\nLaunched {len(futures)} jobs:")
    for label, _ in futures:
        print(f"  - {label}")
    print("\nWaiting for completion (safe to disconnect with --detach)...")

    for label, fut in futures:
        try:
            fut.get()
            print(f"  DONE: {label}")
        except Exception as e:
            print(f"  FAILED: {label}: {e}")


@app.local_entrypoint()
def run_remaining_8b():
    """Complete any remaining 8B experiments.

    Run with: modal run --detach modal_pipeline.py::run_remaining_8b
    """
    print("Uploading data...")
    _upload_data()

    futures = []

    # Complete remaining 8B baselines
    for model in ["llama3.1:8b", "qwen3:8b"]:
        for ds in ["physician", "realworld"]:
            label = f"baselines/{model}/{ds}"
            futures.append((label, baselines_a10g.spawn(model, ds)))

    # Complete remaining 8B RLM
    for model in ["llama3.1:8b", "qwen3:8b"]:
        for arm in ["C_repl_only", "D_rlm_full"]:
            for ds in ["physician", "realworld"]:
                label = f"{arm}/{model}/{ds}"
                futures.append((label, rlm_a10g.spawn(model, arm, ds)))

    print(f"\nLaunched {len(futures)} jobs. Waiting...")
    for label, fut in futures:
        try:
            fut.get()
            print(f"  DONE: {label}")
        except Exception as e:
            print(f"  FAILED: {label}: {e}")


@app.local_entrypoint()
def run_all():
    """Launch all experiments across all model tiers.

    Resumes from volume checkpoints — safe to relaunch.
    Run with: modal run --detach modal_pipeline.py::run_all
    """
    print("Uploading data...")
    _upload_data()

    futures = []

    # --- 8B models on A10G ---
    for model in ["llama3.1:8b", "qwen3:8b"]:
        for ds in ["physician", "realworld"]:
            label = f"baselines/{model}/{ds}"
            futures.append((label, baselines_a10g.spawn(model, ds)))
        for arm in ["C_repl_only", "D_rlm_full"]:
            for ds in ["physician", "realworld"]:
                label = f"{arm}/{model}/{ds}"
                futures.append((label, rlm_a10g.spawn(model, arm, ds)))

    # --- Qwen3-32B on A100 ---
    for ds in ["physician", "realworld"]:
        label = f"baselines/qwen3:32b/{ds}"
        futures.append((label, baselines_a100.spawn("qwen3:32b", ds)))
    for arm in ["C_repl_only", "D_rlm_full"]:
        for ds in ["physician", "realworld"]:
            label = f"{arm}/qwen3:32b/{ds}"
            futures.append((label, rlm_a100.spawn("qwen3:32b", arm, ds)))

    # --- DeepSeek-R1 70B on A100-80GB ---
    for ds in ["physician", "realworld"]:
        label = f"baselines/deepseek-r1:70b/{ds}"
        futures.append((label, baselines_a100_80.spawn("deepseek-r1:70b", ds)))
    for arm in ["C_repl_only", "D_rlm_full"]:
        for ds in ["physician", "realworld"]:
            label = f"{arm}/deepseek-r1:70b/{ds}"
            futures.append((label, rlm_a100_80.spawn("deepseek-r1:70b", arm, ds)))

    print(f"\nLaunched {len(futures)} jobs:")
    for label, _ in futures:
        print(f"  - {label}")
    print("\nAll jobs resume from checkpoints. Waiting for completion...")

    for label, fut in futures:
        try:
            fut.get()
            print(f"  DONE: {label}")
        except Exception as e:
            print(f"  FAILED: {label}: {e}")


@app.local_entrypoint()
def run_wave1():
    """Wave 1: Fair-prompt C/D for ALL models + complete ALL missing conditions.

    This is the primary entrypoint for the methodology redesign.
    - Reruns Arms C/D with the fair (autonomous) RLM prompt for all 4 models
    - Completes DeepSeek-R1-70B baselines (B physician, A/B realworld)
    - Completes Qwen3-32B gaps (A realworld)
    - Runs sensitivity analyses (A+, E) for all models

    Run with: modal run --detach modal_pipeline.py::run_wave1
    """
    print("Uploading data...")
    _upload_data()

    # Archive old prescriptive C/D results on volume before overwriting
    print("Archiving old prescriptive C/D results...")
    try:
        for entry in volume.listdir("raw"):
            fname = Path(entry.path).name
            if fname.startswith(("C_repl_only_", "D_rlm_full_")):
                # Copy to prescriptive archive
                data = b""
                for chunk in volume.read_file(f"raw/{fname}"):
                    data += chunk
                new_name = fname.replace("C_repl_only_", "Cp_repl_only_prescriptive_").replace("D_rlm_full_", "Dp_rlm_full_prescriptive_")
                with volume.batch_upload() as batch:
                    import io
                    batch.put_file(io.BytesIO(data), f"raw_prescriptive_archive/{new_name}")
                # Remove old file so checkpoint/resume starts fresh
                volume.remove_file(f"raw/{fname}")
                print(f"  Archived {fname} -> raw_prescriptive_archive/{new_name}")
    except Exception as e:
        print(f"  Archive step: {e} (may be first run)")

    futures = []

    # --- Fair-prompt C/D for ALL 4 models (new results, replace old prescriptive) ---
    # 8B models on A10G
    for model in ["llama3.1:8b", "qwen3:8b"]:
        for arm in ["C_repl_only", "D_rlm_full"]:
            for ds in ["physician", "realworld"]:
                label = f"FAIR {arm}/{model}/{ds}"
                futures.append((label, rlm_a10g.spawn(model, arm, ds)))

    # Qwen3-32B on A100
    for arm in ["C_repl_only", "D_rlm_full"]:
        for ds in ["physician", "realworld"]:
            label = f"FAIR {arm}/qwen3:32b/{ds}"
            futures.append((label, rlm_a100.spawn("qwen3:32b", arm, ds)))

    # DeepSeek-R1-70B on A100-80GB
    for arm in ["C_repl_only", "D_rlm_full"]:
        for ds in ["physician", "realworld"]:
            label = f"FAIR {arm}/deepseek-r1:70b/{ds}"
            futures.append((label, rlm_a100_80.spawn("deepseek-r1:70b", arm, ds)))

    # --- Complete ALL remaining baselines ---
    # DeepSeek-R1-70B: B physician (only 100/450 done), A+B realworld
    for ds in ["physician", "realworld"]:
        label = f"baselines/deepseek-r1:70b/{ds}"
        futures.append((label, baselines_a100_80.spawn("deepseek-r1:70b", ds)))

    # Qwen3-32B: baselines (complete gaps)
    for ds in ["physician", "realworld"]:
        label = f"baselines/qwen3:32b/{ds}"
        futures.append((label, baselines_a100.spawn("qwen3:32b", ds)))

    # 8B models: baselines (complete gaps including A+ and E)
    for model in ["llama3.1:8b", "qwen3:8b"]:
        for ds in ["physician", "realworld"]:
            label = f"baselines/{model}/{ds}"
            futures.append((label, baselines_a10g.spawn(model, ds)))

    print(f"\nLaunched {len(futures)} jobs:")
    for label, _ in futures:
        print(f"  - {label}")
    print("\nAll jobs resume from checkpoints. Waiting for completion...")

    for label, fut in futures:
        try:
            fut.get()
            print(f"  DONE: {label}")
        except Exception as e:
            print(f"  FAILED: {label}: {e}")


@app.local_entrypoint()
def run_wave2():
    """Wave 2: Generate RLM trajectories from training data for LoRA fine-tuning.

    Runs the fair-prompt D arm (full RLM) on 1330 training cases for each model,
    captures correct trajectories, and creates filtered JSONL for LoRA training.
    Skips DeepSeek-R1-70B (computationally prohibitive; ~50 min/case).

    Run with: modal run --detach modal_pipeline.py::run_wave2
    """
    print("Uploading training data...")
    _upload_data()

    futures = []

    # 8B models on A10G (~1330 cases × ~2 min/case = ~44 hrs each)
    for model in ["llama3.1:8b", "qwen3:8b"]:
        label = f"trajectories/{model}"
        futures.append((label, trajectory_a10g.spawn(model)))

    # Qwen3-32B on A100 (~1330 cases × ~8 min/case = ~180 hrs — may need multiple runs)
    futures.append(("trajectories/qwen3:32b", trajectory_a100.spawn("qwen3:32b")))

    print(f"\nLaunched {len(futures)} trajectory generation jobs:")
    for label, _ in futures:
        print(f"  - {label}")
    print("\nJobs resume from checkpoints. Download trajectories with:")
    print("  modal run modal_pipeline.py::download_trajectories")

    for label, fut in futures:
        try:
            fut.get()
            print(f"  DONE: {label}")
        except Exception as e:
            print(f"  FAILED: {label}: {e}")


@app.local_entrypoint()
def download_trajectories():
    """Download trajectory JSONL files from Modal volume to local output/trajectories/."""
    local_traj = Path("output/trajectories")
    local_traj.mkdir(parents=True, exist_ok=True)

    print("Downloading trajectories from Modal volume...")
    try:
        entries = list(volume.listdir("trajectories"))
    except Exception:
        print("No trajectories found on volume yet.")
        return

    for entry in entries:
        if not entry.path.endswith(".jsonl"):
            continue
        fname = Path(entry.path).name
        local_path = local_traj / fname

        data = b""
        for chunk in volume.read_file(f"trajectories/{fname}"):
            data += chunk
        modal_lines = len(data.decode().strip().split("\n"))

        local_lines = 0
        if local_path.exists():
            local_lines = sum(1 for _ in open(local_path))

        if modal_lines >= local_lines:
            local_path.write_bytes(data)
            print(f"  {fname}: {modal_lines} trajectories (was {local_lines})")
        else:
            print(f"  {fname}: KEPT local ({local_lines} > modal {modal_lines})")

    print(f"\nTrajectories saved to {local_traj.resolve()}")


@app.local_entrypoint()
def download():
    """Download all JSONL results from Modal volume to local output/raw/.

    Only overwrites local files if Modal version has more results.
    """
    local_raw = Path("output/raw")
    local_raw.mkdir(parents=True, exist_ok=True)

    print("Downloading results from Modal volume...")
    try:
        entries = list(volume.listdir("raw"))
    except Exception:
        print("No results found on volume yet.")
        return

    for entry in entries:
        if not entry.path.endswith(".jsonl"):
            continue
        fname = Path(entry.path).name
        local_path = local_raw / fname

        data = b""
        for chunk in volume.read_file(f"raw/{fname}"):
            data += chunk
        modal_lines = len(data.decode().strip().split("\n"))

        # Only overwrite if Modal has more results
        local_lines = 0
        if local_path.exists():
            local_lines = sum(1 for _ in open(local_path))

        if modal_lines >= local_lines:
            local_path.write_bytes(data)
            print(f"  {fname}: {modal_lines} results (was {local_lines})")
        else:
            print(f"  {fname}: KEPT local ({local_lines} > modal {modal_lines})")

    print(f"\nResults saved to {local_raw.resolve()}")

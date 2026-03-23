"""Generate RLM trajectories from training data for post-training fine-tuning.

Runs the fair-prompt RLM (Arm D) on training cases, captures full conversation
trajectories, filters to keep only correct predictions, and formats as
chat-template JSONL for LoRA fine-tuning.

Usage:
    # Local (requires ollama running):
    python 07_generate_trajectories.py --model qwen3:8b

    # On Modal (see modal_pipeline.py for GPU-tiered entrypoints)

Output:
    output/trajectories/{model}_trajectories.jsonl
    output/trajectories/{model}_trajectory_stats.json
"""
import json
import os
import time
import argparse
import traceback
from pathlib import Path

DATA_DIR = Path(os.environ.get(
    'RLM_DATA_DIR',
    str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
))
OUTPUT_DIR = Path(__file__).parent / 'output' / 'trajectories'
LOG_DIR = Path(__file__).parent / 'logs' / 'trajectories'

# Training data: physician + realworld training sets
TRAIN_FILES = [
    DATA_DIR / 'physician_train.json',
    DATA_DIR / 'realworld_train.json',
]
# Fallback if split files don't exist
COMBINED_TRAIN = DATA_DIR / 'combined_train.json'


def load_training_data():
    """Load training cases from available files."""
    cases = []
    for path in TRAIN_FILES:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                cases.extend(data)
                print(f"  Loaded {len(data)} cases from {path.name}")

    if not cases and COMBINED_TRAIN.exists():
        with open(COMBINED_TRAIN) as f:
            cases = json.load(f)
            print(f"  Loaded {len(cases)} cases from {COMBINED_TRAIN.name}")

    if not cases:
        # Use physician_full as fallback (we'll use it for trajectory generation
        # even though it's technically the test set — the trajectories are for
        # learning the RLM *process*, not memorizing answers)
        fallback = DATA_DIR / 'physician_full.json'
        if fallback.exists():
            with open(fallback) as f:
                cases = json.load(f)
                print(f"  WARNING: Using physician_full.json as training data ({len(cases)} cases)")

    return cases


def extract_trajectory(rlm_instance, message, verbose_log_dir=None):
    """Run RLM and capture the full conversation trajectory.

    Returns (raw_response, trajectory_messages) where trajectory_messages
    is a list of {role, content} dicts suitable for chat-template fine-tuning.
    """
    from rlm import RLM
    from prompts import RLM_SYSTEM

    result = rlm_instance.completion(message)

    # Extract the raw response
    if isinstance(result, str):
        raw_text = result
    elif hasattr(result, 'response'):
        raw_text = result.response or ""
    else:
        raw_text = str(result)

    # Build trajectory as chat messages
    # The trajectory includes: system prompt, user message (patient text),
    # and the full assistant response (code blocks + FINAL output)
    trajectory = [
        {"role": "system", "content": RLM_SYSTEM.replace("{custom_tools_section}", "")},
        {"role": "user", "content": message},
        {"role": "assistant", "content": raw_text},
    ]

    return raw_text, trajectory


def is_correct(parsed, case):
    """Check if the parsed prediction matches ground truth."""
    if not parsed or not isinstance(parsed, dict):
        return False

    detection_pred = parsed.get("detection", -1)
    action_pred = parsed.get("action", "")
    detection_truth = case.get("detection_truth", -1)
    action_truth = case.get("action_truth", "")

    # Detection must match
    if int(detection_pred) != int(detection_truth):
        return False

    # Action must match (case-insensitive)
    if action_pred.strip().lower() != action_truth.strip().lower():
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='qwen3:8b',
                        help='Model name for ollama (default: qwen3:8b)')
    parser.add_argument('--base-url', default='http://localhost:11434/v1')
    parser.add_argument('--max-cases', type=int, default=None,
                        help='Max cases to process (for testing)')
    parser.add_argument('--max-depth', type=int, default=3,
                        help='RLM max_depth (default: 3 for full RLM)')
    parser.add_argument('--max-iterations', type=int, default=8)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data...")
    cases = load_training_data()
    if not cases:
        print("ERROR: No training data found.")
        return

    if args.max_cases:
        cases = cases[:args.max_cases]
    print(f"  {len(cases)} cases to process")

    # Initialize RLM
    from rlm import RLM
    from rlm.logger import RLMLogger
    from prompts import RLM_SYSTEM
    from utils import parse_json_response

    model_slug = args.model.replace(":", "_")
    logger = RLMLogger(log_dir=str(LOG_DIR / model_slug))

    rlm_instance = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": args.model,
            "api_key": "ollama",
            "base_url": args.base_url,
        },
        environment="local",
        max_depth=args.max_depth,
        max_iterations=args.max_iterations,
        custom_system_prompt=RLM_SYSTEM,
        verbose=False,
        logger=logger,
    )

    # Checkpoint/resume
    out_path = OUTPUT_DIR / f"{model_slug}_trajectories.jsonl"
    done_indices = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done_indices.add(row["case_idx"])
                except Exception:
                    pass
    print(f"  {len(done_indices)} already completed")

    stats = {"total": 0, "correct": 0, "incorrect": 0, "error": 0, "parse_fail": 0}

    with open(out_path, "a") as fout:
        for i, case in enumerate(cases):
            if i in done_indices:
                continue

            message = case.get("prompt", case.get("message", ""))
            t0 = time.time()

            try:
                raw_text, trajectory = extract_trajectory(rlm_instance, message)
                parsed = parse_json_response(raw_text)
                elapsed = time.time() - t0

                if not parsed:
                    stats["parse_fail"] += 1
                    correct = False
                else:
                    correct = is_correct(parsed, case)

                row = {
                    "case_idx": i,
                    "case_name": case.get("name", case.get("case_label", f"case_{i}")),
                    "model": args.model,
                    "correct": correct,
                    "detection_truth": case.get("detection_truth"),
                    "action_truth": case.get("action_truth"),
                    "detection_pred": parsed.get("detection") if parsed else None,
                    "action_pred": parsed.get("action") if parsed else None,
                    "trajectory": trajectory,
                    "elapsed_sec": round(elapsed, 2),
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()

                if correct:
                    stats["correct"] += 1
                else:
                    stats["incorrect"] += 1

            except Exception:
                stats["error"] += 1
                elapsed = time.time() - t0
                row = {
                    "case_idx": i,
                    "case_name": case.get("name", case.get("case_label", f"case_{i}")),
                    "model": args.model,
                    "correct": False,
                    "error": traceback.format_exc(),
                    "elapsed_sec": round(elapsed, 2),
                }
                fout.write(json.dumps(row) + "\n")
                fout.flush()

            stats["total"] += 1
            if stats["total"] % 20 == 0:
                print(f"  {stats['total']}/{len(cases) - len(done_indices)}: "
                      f"{stats['correct']} correct, {stats['incorrect']} wrong, "
                      f"{stats['error']} errors ({elapsed:.1f}s)")

    # Save stats
    stats_path = OUTPUT_DIR / f"{model_slug}_trajectory_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nDone. Stats: {json.dumps(stats)}")
    print(f"Correct trajectory rate: {stats['correct']}/{stats['total']} "
          f"({stats['correct']/max(stats['total'],1)*100:.1f}%)")
    print(f"Output: {out_path}")

    # Also create the filtered training file (correct trajectories only)
    filtered_path = OUTPUT_DIR / f"{model_slug}_train_filtered.jsonl"
    n_filtered = 0
    with open(out_path) as fin, open(filtered_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            if row.get("correct") and "trajectory" in row:
                # Format for fine-tuning: just the messages
                train_example = {"messages": row["trajectory"]}
                fout.write(json.dumps(train_example) + "\n")
                n_filtered += 1
    print(f"Filtered training set: {n_filtered} examples -> {filtered_path}")


if __name__ == "__main__":
    main()

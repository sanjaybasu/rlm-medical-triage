"""
Arms C (REPL-only, max_depth=0) and D (full RLM, max_depth=2) via rlm library.
Uses ollama as OpenAI-compatible backend.
The patient message is passed directly as the prompt; the RLM framework
stores it as the `context` variable in the REPL automatically.
Saves results as JSONL with checkpoint/resume.
"""
import json
import os
import time
import argparse
import traceback
from pathlib import Path

from rlm import RLM
from rlm.logger import RLMLogger

from prompts import RLM_SYSTEM, RLM_SYSTEM_PRESCRIPTIVE
from utils import parse_json_response

DATA_DIR = Path(os.environ.get(
    'RLM_DATA_DIR',
    str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
))
OUTPUT_DIR = Path(__file__).parent / 'output' / 'raw'
LOG_DIR = Path(__file__).parent / 'logs'

MODELS = ['qwen3:8b', 'llama3.1:8b']
TEST_SETS = {
    'physician': DATA_DIR / 'physician_full.json',
    'realworld': DATA_DIR / 'realworld_rlm_subsample.json',
}

# Arm C: REPL environment but shallow sub-calls (max_depth=1).
# Arm D: Full RLM with recursive decomposition (max_depth=3).
# Both use the fair prompt (RLM_SYSTEM) by default.
# Prescriptive variants use the old hardcoded-regex prompt for sensitivity analysis.
ARMS = {
    'C_repl_only': {'max_depth': 1, 'max_iterations': 8, 'prompt': 'fair'},
    'D_rlm_full': {'max_depth': 3, 'max_iterations': 8, 'prompt': 'fair'},
    'Cp_repl_only_prescriptive': {'max_depth': 1, 'max_iterations': 8, 'prompt': 'prescriptive'},
    'Dp_rlm_full_prescriptive': {'max_depth': 3, 'max_iterations': 8, 'prompt': 'prescriptive'},
}


def load_completed(output_path):
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                    done.add(row['case_idx'])
                except Exception:
                    pass
    return done


def _extract_response(result):
    """Handle both string and RLMChatCompletion return types."""
    if isinstance(result, str):
        return result
    if hasattr(result, 'response'):
        return result.response or ""
    return str(result)


def run_rlm_arm(model, arm_name, arm_cfg, cases, output_path, base_url):
    done = load_completed(output_path)
    print(f"  {arm_name}/{model}: {len(done)}/{len(cases)} already done")

    logger = RLMLogger(log_dir=str(LOG_DIR / f'{arm_name}_{model.replace(":", "_")}'))

    # Select prompt: fair (autonomous) or prescriptive (hardcoded regex).
    prompt_type = arm_cfg.get('prompt', 'fair')
    if prompt_type == 'prescriptive':
        sys_prompt = RLM_SYSTEM_PRESCRIPTIVE
    else:
        sys_prompt = RLM_SYSTEM

    # For shallow arms (C variants), remove llm_query references so the model
    # doesn't attempt sub-calls that will fail at max_depth=1.
    if 'C_repl_only' in arm_name or 'Cp_repl_only' in arm_name:
        sys_prompt = sys_prompt.replace(
            'llm_query(prompt)', '# (sub-calls disabled in this arm)'
        ).replace(
            'You can use llm_query(prompt) to make sub-calls for clinical assessment of individual findings.\n\n',
            ''
        )

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": model,
            "api_key": "ollama",
            "base_url": base_url,
        },
        environment="local",
        max_depth=arm_cfg['max_depth'],
        max_iterations=arm_cfg['max_iterations'],
        custom_system_prompt=sys_prompt,
        verbose=False,
        logger=logger,
    )

    with open(output_path, 'a') as fout:
        for i, case in enumerate(cases):
            if i in done:
                continue

            # Pass patient message directly as the prompt.
            # RLM framework stores it as `context` in the REPL.
            message = case.get('prompt', case.get('message', ''))

            t0 = time.time()
            raw_text = ""
            metadata = {}
            try:
                result = rlm.completion(message)
                raw_text = _extract_response(result)
                if hasattr(result, 'execution_time'):
                    metadata['execution_time'] = result.execution_time
                if hasattr(result, 'usage_summary'):
                    us = result.usage_summary
                    metadata['input_tokens'] = getattr(us, 'input_tokens', None)
                    metadata['output_tokens'] = getattr(us, 'output_tokens', None)
                elapsed = time.time() - t0
            except Exception:
                raw_text = f"ERROR: {traceback.format_exc()}"
                elapsed = time.time() - t0

            parsed = parse_json_response(raw_text)
            row = {
                'case_idx': i,
                'case_name': case.get('name', case.get('case_label', f'case_{i}')),
                'model': model,
                'arm': arm_name,
                'raw_response': raw_text,
                'parsed': parsed,
                'elapsed_sec': round(elapsed, 2),
                'rlm_metadata': metadata,
            }
            fout.write(json.dumps(row) + '\n')
            fout.flush()

            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(cases)} complete ({elapsed:.1f}s last)")

    print(f"  {arm_name}/{model}: done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--arms', nargs='+', default=list(ARMS.keys()))
    parser.add_argument('--datasets', nargs='+', default=list(TEST_SETS.keys()))
    parser.add_argument('--base-url', default='http://localhost:11434/v1')
    args = parser.parse_args()

    for ds_name in args.datasets:
        ds_path = TEST_SETS[ds_name]
        with open(ds_path) as f:
            cases = json.load(f)
        print(f"\nDataset: {ds_name} ({len(cases)} cases)")

        for model in args.models:
            for arm_name in args.arms:
                arm_cfg = ARMS[arm_name]
                out_path = OUTPUT_DIR / f'{arm_name}_{model.replace(":", "_")}_{ds_name}.jsonl'
                run_rlm_arm(model, arm_name, arm_cfg, cases, out_path, args.base_url)


if __name__ == '__main__':
    main()

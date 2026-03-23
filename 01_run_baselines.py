"""
Arms A (single-pass) and B (chain-of-thought) via ollama OpenAI-compatible API.
Iterates over physician + real-world test sets x 2 models.
Saves results as JSONL with checkpoint/resume.
"""
import json
import os
import time
import argparse
from pathlib import Path
from openai import OpenAI

from prompts import SINGLE_PASS, CHAIN_OF_THOUGHT, SINGLE_PASS_GROUNDED, RLM_PROMPT_NO_REPL
from utils import parse_json_response

DATA_DIR = Path(os.environ.get(
    'RLM_DATA_DIR',
    str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
))
OUTPUT_DIR = Path(__file__).parent / 'output' / 'raw'

MODELS = ['qwen3:8b', 'llama3.1:8b']
ARMS = {
    'A_single_pass': SINGLE_PASS,
    'Aplus_grounded': SINGLE_PASS_GROUNDED,
    'B_chain_of_thought': CHAIN_OF_THOUGHT,
    'E_rlm_prompt_no_repl': RLM_PROMPT_NO_REPL,
}
TEST_SETS = {
    'physician': DATA_DIR / 'physician_full.json',
    'realworld': DATA_DIR / 'realworld_full.json',
}


def load_completed(output_path):
    """Return set of completed case indices from existing JSONL."""
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


def run_arm(client, model, arm_name, prompt_template, cases, output_path):
    """Run a single arm on all cases with checkpoint/resume."""
    done = load_completed(output_path)
    print(f"  {arm_name}/{model}: {len(done)}/{len(cases)} already done")

    with open(output_path, 'a') as fout:
        for i, case in enumerate(cases):
            if i in done:
                continue

            message = case.get('prompt', case.get('message', ''))
            prompt = prompt_template.format(message=message)

            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2048,
                )
                raw_text = resp.choices[0].message.content
                elapsed = time.time() - t0
            except Exception as e:
                raw_text = f"ERROR: {e}"
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
            }
            fout.write(json.dumps(row) + '\n')
            fout.flush()

            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(cases)} complete ({elapsed:.1f}s last)")

    print(f"  {arm_name}/{model}: done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--arms', nargs='+', default=list(ARMS.keys()))
    parser.add_argument('--datasets', nargs='+', default=list(TEST_SETS.keys()))
    parser.add_argument('--base-url', default='http://localhost:11434/v1')
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key='ollama')

    for ds_name in args.datasets:
        ds_path = TEST_SETS[ds_name]
        with open(ds_path) as f:
            cases = json.load(f)
        print(f"\nDataset: {ds_name} ({len(cases)} cases)")

        for model in args.models:
            for arm_name in args.arms:
                prompt_template = ARMS[arm_name]
                out_path = OUTPUT_DIR / f'{arm_name}_{model.replace(":", "_")}_{ds_name}.jsonl'
                run_arm(client, model, arm_name, prompt_template, cases, out_path)


if __name__ == '__main__':
    main()

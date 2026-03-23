"""
Smoke test: run 5 physician cases through all 4 arms on one model.
Verifies ollama connectivity, RLM library, and output parsing.
"""
import json
import os
import time
from pathlib import Path
from openai import OpenAI

from prompts import SINGLE_PASS, CHAIN_OF_THOUGHT, RLM_SYSTEM
from utils import parse_json_response, extract_detection_action, compute_phantom_symptom_rate

DATA_DIR = Path(os.environ.get(
    'RLM_DATA_DIR',
    str(Path(__file__).resolve().parents[1] / 'notebooks' / 'rl_vs_llm_safety_v2' / 'data_final')
))
MODEL = 'llama3.1:8b'
BASE_URL = 'http://localhost:11434/v1'
N_CASES = 5


def test_baseline(client, cases, prompt_template, arm_name):
    print(f"\n--- {arm_name} ---")
    for i, case in enumerate(cases[:N_CASES]):
        msg = case.get('prompt', '')
        prompt = prompt_template.format(message=msg)
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content
        elapsed = time.time() - t0
        parsed = parse_json_response(raw)
        det, act = extract_detection_action(parsed)
        truth_det = case.get('detection_truth', 0)
        evidence = parsed.get('evidence', []) if parsed else []
        psr, cfs, nc = compute_phantom_symptom_rate(evidence, msg)
        print(f"  Case {i}: det={det}(truth={truth_det}), act={act}, "
              f"PSR={psr:.2f}, CFS={cfs:.2f}, claims={nc}, {elapsed:.1f}s")
        if parsed is None:
            print(f"    PARSE FAILURE. Raw (first 200): {raw[:200]}")


def test_rlm(cases, max_depth, arm_name):
    print(f"\n--- {arm_name} (max_depth={max_depth}) ---")
    try:
        from rlm import RLM
    except ImportError:
        print("  SKIP: rlm library not installed. pip install rlms")
        return

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": MODEL,
            "api_key": "ollama",
            "base_url": BASE_URL,
        },
        environment="local",
        max_depth=max_depth,
        max_iterations=5,
        custom_system_prompt=RLM_SYSTEM,
        verbose=True,
    )

    for i, case in enumerate(cases[:N_CASES]):
        msg = case.get('prompt', '')
        t0 = time.time()
        try:
            result = rlm.completion(msg)
            if isinstance(result, str):
                raw = result
            elif hasattr(result, 'response'):
                raw = result.response or ""
            else:
                raw = str(result)
        except Exception as e:
            raw = f"ERROR: {e}"
        elapsed = time.time() - t0
        parsed = parse_json_response(raw)
        det, act = extract_detection_action(parsed)
        truth_det = case.get('detection_truth', 0)
        evidence = parsed.get('evidence', []) if parsed else []
        psr, cfs, nc = compute_phantom_symptom_rate(evidence, msg)
        print(f"  Case {i}: det={det}(truth={truth_det}), act={act}, "
              f"PSR={psr:.2f}, CFS={cfs:.2f}, claims={nc}, {elapsed:.1f}s")
        if 'ERROR' in raw:
            print(f"    {raw[:300]}")


def main():
    with open(DATA_DIR / 'physician_test.json') as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} physician cases, testing first {N_CASES}")
    print(f"Model: {MODEL}, Endpoint: {BASE_URL}")

    # Verify connectivity.
    client = OpenAI(base_url=BASE_URL, api_key='ollama')
    print("\nConnectivity check...")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}],
        max_tokens=10,
    )
    print(f"  Response: {resp.choices[0].message.content}")

    # Arm A.
    test_baseline(client, cases, SINGLE_PASS, 'A_single_pass')

    # Arm B.
    test_baseline(client, cases, CHAIN_OF_THOUGHT, 'B_chain_of_thought')

    # Arm C: REPL-only (max_depth=1 gives REPL but sub-calls fall back to plain LM).
    test_rlm(cases, max_depth=1, arm_name='C_repl_only')

    # Arm D: Full RLM with recursive sub-calls.
    test_rlm(cases, max_depth=3, arm_name='D_rlm_full')

    print("\nSmoke test complete.")


if __name__ == '__main__':
    main()

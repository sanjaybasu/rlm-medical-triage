# revision_v2/

Scripts added during the major-revision cycle. Only code is included here; manuscript artifacts, results JSONLs, figures, and internal status documents are not in this repository.

## Layout

```
revision_v2/
  scripts/
    frontier_repl_harness.py        # RLM REPL re-implementation for Anthropic / OpenAI / Google
    modal_frontier_pipeline.py      # Modal orchestrator: Arms A/B/C/E + M2 (thinking off) + M5 (grounding stripped)
    modal_m4_fresh.py               # M4: Qwen3-8B Arm C with a four-step chain-of-thought preamble (Modal)
    m3_local_deepseek_thinking.py   # M3: DeepSeek-R1-70B Arm C with reasoning enabled (local Ollama, 32K-context variant)
    analyze_m3.py                   # M3 output analyzer (JSON + ast.literal_eval parser, per-case PSR/sens/spec)
    m1_phantom_stereotyping.py      # M1: phantom-symptom string-distribution concentration analysis
    length_stratified_analysis.py   # PSR/CFS by patient-message length tertile
    mitigation_pilot_verifier.py    # Post-execution source-grounding verifier (two-stage quote + symptom check)
    comprehensive_consistency_check.py  # Cross-document numerical / citation / naming consistency linter
  audit/
    verify_numbers.py               # Re-derive registered claims from canonical sources, ±0.001 tolerance
```

### Provenance registry schema

`verify_numbers.py` reads `audit/data_provenance.csv` (not included in this repository — it indexes manuscript-specific reported values). The expected schema is:

```
claim_id, description, reported_value, source_file, derivation,
location_in_manuscript, location_in_appendix
```

One row per reportable number in the manuscript; the `derivation` field describes how to compute `reported_value` from `source_file`.


## Paths and environment variables

The scripts resolve paths relative to the repository root by default. Override with:

| Variable             | Default                                            |
|----------------------|----------------------------------------------------|
| `RLM_DATA_DIR`       | `<repo>/data`                                      |
| `RLM_OUTPUT_DIR`     | `<repo>/output`                                    |
| `RLM_REVISION_DIR`   | `<repo>/revision_v2`                               |
| `RLM_PACKAGING_DIR`  | `<repo>`                                           |
| `RLM_PHYSICIAN_DATA` | `<repo>/data/physician_full.json`                  |

## API credentials

Modal scripts read provider API keys from Modal Secrets named `anthropic`, `openai`, and `google`. Keys are not embedded in source.

## Frontier models evaluated

| Provider  | Model                       | Mode                                                              |
|-----------|-----------------------------|-------------------------------------------------------------------|
| Anthropic | `claude-opus-4-7`           | Messages API, `thinking={"type":"adaptive"}`, `output_config.effort="low"` |
| OpenAI    | `gpt-5.5`                   | Responses API, reasoning enabled, `max_output_tokens` floor 2000  |
| Google    | `gemini-3.1-pro-preview`    | Gen AI SDK, `ThinkingConfig(thinking_budget>=1)`                  |

## Local M3 setup

DeepSeek-R1-70B Arm C with reasoning enabled requires a 32K-context Ollama variant:

```
# /tmp/Modelfile_deepseek_r1_70b_32k
FROM deepseek-r1:70b
PARAMETER num_ctx 32768
```

```
ollama create deepseek-r1:70b-32k -f /tmp/Modelfile_deepseek_r1_70b_32k
python revision_v2/scripts/m3_local_deepseek_thinking.py --pilot 5
```

## Resume semantics

All long-running scripts (`modal_frontier_pipeline.py`, `modal_m4_fresh.py`, `m3_local_deepseek_thinking.py`) write per-case JSONL after each completion and skip case indices already present on relaunch.

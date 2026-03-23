# rlm-medical-triage


Pre-registration: https://osf.io/c69j4/
IRB: WCG protocol #20253751

---

## Overview

Four-arm ablation study comparing inference strategies for medical triage safety:

| Arm | Method | Description |
|-----|--------|-------------|
| A | Single-pass | Standard completion with structured JSON output |
| B | Chain-of-thought | Step-by-step reasoning with evidence quoting |
| C | REPL-only | RLM REPL environment, shallow sub-calls (max_depth=1) |
| D | Full RLM | Full recursive decomposition (max_depth=3) |
| A+ | Grounding sensitivity | Arm A + anti-hallucination instruction only |
| E | Extraction prompt | Prescriptive regex extraction, no REPL |

Models: Llama-3.1-8B, Qwen3-8B, Qwen3-32B, DeepSeek-R1-70B
Test sets: physician-created (N=450), real-world Medicaid (N=2,000)

---

## Setup

```bash
pip install rlms json5 openai statsmodels scikit-learn matplotlib scipy
ollama pull qwen3:8b qwen3:32b llama3.1:8b deepseek-r1:70b
```

Requires Python ≥3.10. Statistical analysis requires `statsmodels`; run with anaconda python if system python lacks it.

---

## Pipeline

```bash
# Smoke test on 5 cases (~5 min)
python 00_smoke_test.py

# Arm A (single-pass) and B (chain-of-thought)
python 01_run_baselines.py

# Arms C (REPL) and D (full RLM) using fair (autonomous) prompt
python 02_run_rlm.py

# Parse outputs and compute PSR/CFS hallucination metrics
python 03_parse_and_audit.py

# Compute all metrics with 95% CIs (requires statsmodels)
python 04_evaluate.py

# CFS threshold sensitivity analysis
python 04b_cfs_threshold_sensitivity.py

# Ground truth sensitivity analysis (Monte Carlo)
python 04c_circular_gt_sensitivity.py

# Statistical tests and figures 1-3
python 05_analysis.py

# Nature-quality heatmap figures 4-5
python 05b_figure_heatmap.py

# Table 1 (test set characteristics)
python 06_generate_table1.py
```

For large-scale cloud inference (Modal GPU), use `modal_pipeline.py`.

---

## Data availability

The physician-created test set (N=450), all system prompts, and analysis scripts are included in this repository.

The real-world patient messages cannot be shared due to HIPAA; real-world raw JSONL outputs are excluded from this repository. Aggregate metrics and de-identified audit data are in `output/metrics/` and `output/adjudication/`.

---

## Outputs

```
output/
  metrics/
    all_metrics.csv          # all sensitivity/specificity/PSR/CFS/MCC by arm/model/dataset
    category_sensitivity.csv # per-category sensitivity
    parse_failure_summary.csv
    cfs_threshold_sensitivity.csv
    gt_sensitivity_analysis.csv
  figures/
    figure1_sensitivity_psr.{png,pdf}
    figure2_scaling_analysis.{png,pdf}
    figure3_cot_scaling_effect.{png,pdf}
    figure4_heatmap_physician.{png,pdf}
    figure5_heatmap_realworld.{png,pdf}
    sfigure1-3.{png,pdf}
  adjudication/
    psr_adjudication_claims.csv   # 86-claim physician adjudication dataset
    adjudication_instructions.md
```

---

## Key metrics

**Phantom Symptom Rate (PSR):** proportion of model-claimed clinical findings absent from the patient message (lower = fewer hallucinations).

**Citation Fidelity Score (CFS):** proportion of model-claimed findings with verifiable quotes from the source message (higher = better grounding).

Both computed via fuzzy string matching (Python `difflib.SequenceMatcher`, threshold 0.7). Validated against three-physician blinded adjudication (Fleiss κ=0.61, 95.2% positive predictive value for PHANTOM verdicts).

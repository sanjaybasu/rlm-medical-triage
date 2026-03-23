# rlm-medical-triage

Code and data for: **"AI code execution amplifies clinical hallucinations in medical triage"**

Basu S, Patel S, Batniji R. (2026).

Pre-registration: https://osf.io/c69j4/
IRB: WCG protocol #20253751

---

## Overview

Four-arm ablation study comparing inference strategies for medical triage safety across four open-source models (8B to 70B parameters).

| Arm | Method | Description |
|-----|--------|-------------|
| A | Single-pass | Standard completion with structured JSON output |
| B | Chain-of-thought | Step-by-step reasoning with evidence quoting |
| C | REPL-only | RLM REPL environment, shallow sub-calls (max_depth=1) |
| D | Full RLM | Full recursive decomposition (max_depth=3) |
| A+ | Grounding instruction | Arm A + anti-hallucination instruction only |
| E | Extraction prompt | Prescriptive regex extraction, no REPL |

Models: Llama-3.1-8B, Qwen3-8B, Qwen3-32B, DeepSeek-R1-70B
Test sets: physician-created (N=450), real-world Medicaid (N=2,000)

---

## Setup

```bash
pip install rlms json5 openai statsmodels scikit-learn matplotlib scipy
ollama pull qwen3:8b qwen3:32b llama3.1:8b deepseek-r1:70b
```

Requires Python 3.10 or later.

---

## Pipeline

```bash
# Smoke test on 5 cases
python 00_smoke_test.py

# Arm A (single-pass) and B (chain-of-thought)
python 01_run_baselines.py

# Arms C (REPL) and D (full RLM)
python 02_run_rlm.py

# Parse outputs and compute PSR/CFS hallucination metrics
python 03_parse_and_audit.py

# Compute all metrics with 95% CIs
python 04_evaluate.py

# CFS threshold sensitivity analysis
python 04b_cfs_threshold_sensitivity.py

# Ground truth sensitivity analysis (Monte Carlo)
python 04c_circular_gt_sensitivity.py

# Bootstrap CIs for action accuracy and critical under-triage
python 04d_bootstrap_action_cut_ci.py

# Statistical tests and figures 1-3, supplementary figures 1-3
python 05_analysis.py

# Heatmap figures 4-5
python 05b_figure_heatmap.py

# Table 1 (test set characteristics)
python 06_generate_table1.py
```

For cloud GPU inference, use `modal_pipeline.py` with [Modal](https://modal.com).

---

## Data availability

The physician-created test set (N=450), all system prompts, and analysis scripts are included in this repository.

Real-world patient messages cannot be shared due to HIPAA. Aggregate metrics and de-identified audit data are in `output/metrics/` and `output/adjudication/`.

---

## Outputs

```
output/
  metrics/
    all_metrics.csv               # sensitivity/specificity/PSR/CFS/MCC by arm/model/dataset
    category_sensitivity.csv      # per-hazard-category sensitivity (physician set)
    parse_failure_summary.csv     # parse failure rates by condition
    cfs_threshold_sensitivity.csv # fuzzy match threshold sensitivity analysis
    gt_sensitivity_analysis.csv   # Monte Carlo ground truth perturbation
    action_cut_bootstrap_ci.csv   # bootstrap CIs for action accuracy and CUT
    statistical_tests.csv         # McNemar and bootstrap pairwise tests
  figures/
    figure1_sensitivity_psr.{png,pdf}
    figure2_scaling_analysis.{png,pdf}
    figure3_cot_scaling_effect.{png,pdf}
    figure4_heatmap_physician.{png,pdf}
    figure5_heatmap_realworld.{png,pdf}
    sfigure1_realworld_sensitivity.{png,pdf}
    sfigure2_action_accuracy.{png,pdf}
    sfigure3_sensitivity_vs_psr.{png,pdf}
  adjudication/
    psr_adjudication_claims.csv   # 86-claim physician adjudication dataset
    adjudication_instructions.md  # reviewer protocol
```

---

## Key metrics

**Phantom Symptom Rate (PSR):** proportion of model-claimed clinical findings absent from the patient message. Lower values indicate fewer hallucinated findings.

**Citation Fidelity Score (CFS):** proportion of model-claimed findings with verifiable quotes from the source message. Higher values indicate better grounding.

Both metrics are computed via fuzzy string matching (`difflib.SequenceMatcher`, threshold 0.7) and validated against three-physician blinded adjudication (Fleiss kappa = 0.61, 95.2% positive predictive value for fabrication verdicts).

---

## License

Code: MIT. Data: see Data Availability above.

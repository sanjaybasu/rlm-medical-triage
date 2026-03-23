#!/bin/bash
# Run the full experiment pipeline sequentially.
# Baselines take ~2hr, RLM arms take ~18hr.
# Run with: nohup bash run_all.sh > run_all.log 2>&1 &

set -e
cd "$(dirname "$0")"

echo "=== Starting full experiment pipeline ==="
echo "Start time: $(date)"

# Step 1: Baselines (Arms A+B) - one model at a time to avoid ollama contention.
echo ""
echo "=== Step 1a: Baselines with llama3.1:8b ==="
python 01_run_baselines.py --models llama3.1:8b --datasets physician realworld

echo ""
echo "=== Step 1b: Baselines with qwen3:8b ==="
python 01_run_baselines.py --models qwen3:8b --datasets physician realworld

# Step 2: RLM arms (C+D) - one model at a time.
echo ""
echo "=== Step 2a: RLM arms with llama3.1:8b ==="
python 02_run_rlm.py --models llama3.1:8b --datasets physician realworld

echo ""
echo "=== Step 2b: RLM arms with qwen3:8b ==="
python 02_run_rlm.py --models qwen3:8b --datasets physician realworld

# Step 3: Parse and audit.
echo ""
echo "=== Step 3: Parse and audit ==="
python 03_parse_and_audit.py

# Step 4: Evaluate metrics.
echo ""
echo "=== Step 4: Evaluate ==="
python 04_evaluate.py

# Step 5: Statistical tests and figures.
echo ""
echo "=== Step 5: Analysis ==="
python 05_analysis.py

echo ""
echo "=== Pipeline complete ==="
echo "End time: $(date)"

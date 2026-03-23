#!/bin/bash
# Finalize RLM triage paper: download results, parse, evaluate, generate figures.
# Run after all Modal jobs complete: bash finalize_paper.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=/opt/anaconda3/bin/python3
NOTEBOOK_FIGS=../../notebooks/rlm-medical-triage/figures

echo "=== RLM Triage Paper Finalization — $(date) ==="

# Step 1: Download all results from Modal volume
echo ""
echo "[1/5] Downloading results from Modal volume..."
modal run modal_pipeline.py::download 2>&1 | tail -10

# Step 2: Parse and audit
echo ""
echo "[2/5] Parsing and auditing all results..."
$PYTHON 03_parse_and_audit.py 2>&1 | tail -5

# Step 3: Evaluate (compute metrics with bootstrap CIs)
echo ""
echo "[3/5] Computing metrics (this takes ~5 min for bootstrap CIs)..."
$PYTHON 04_evaluate.py 2>&1 | tail -10

# Step 4: Generate all figures
echo ""
echo "[4/5] Generating figures..."
$PYTHON 05_analysis.py 2>&1 | tail -15
$PYTHON 05b_figure_heatmap.py 2>&1 | tail -5

# Step 5: Copy figures to manuscript directory
echo ""
echo "[5/5] Copying figures to manuscript directory..."
mkdir -p "$NOTEBOOK_FIGS"
cp output/figures/*.png output/figures/*.pdf "$NOTEBOOK_FIGS/"
echo "  Copied $(ls output/figures/*.png | wc -l) PNG + PDF figures to $NOTEBOOK_FIGS/"

# Summary
echo ""
echo "=== DONE ==="
echo "Figures in: output/figures/"
echo "Metrics in: output/metrics/all_metrics.csv"
echo ""
echo "Remaining manual steps:"
echo "  1. Review figures for missing markers (open output/figures/*.png)"
echo "  2. Update manuscript numbers from output/metrics/all_metrics.csv"
echo "  3. Update heatmap REALWORLD_ROWS in 05b_figure_heatmap.py if new rows available"
echo "  4. Add DeepSeek C/D rows to heatmap if /no_think worked"
echo ""
echo "Key files to check:"
ls -la output/raw/ | grep -E "deepseek.*C_|deepseek.*D_|B_chain.*realworld" | awk '{print "  " $NF ": " $5 " bytes"}'

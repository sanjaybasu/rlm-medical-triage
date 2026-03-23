#!/bin/bash
# Monitor and report status of all Modal RLM triage jobs.
# Run: bash monitor_jobs.sh
# Or set up as cron: */30 * * * * cd /Users/sanjaybasu/waymark-local/packaging/rlm-medical-triage && bash monitor_jobs.sh >> logs/monitor.log 2>&1

set -euo pipefail
cd "$(dirname "$0")"

echo "=== RLM Triage Job Monitor — $(date) ==="
echo ""

# List all active rlm-medical apps
echo "--- Active Modal Apps ---"
modal app list 2>&1 | grep -E "rlm-medica.*ephemeral" || echo "  (none running)"
echo ""

# Check volume for completed files
echo "--- Volume Contents ---"
EXPECTED_FILES=(
    "B_chain_of_thought_qwen3_32b_realworld"
    "B_chain_of_thought_deepseek-r1_70b_realworld"
    "C_repl_only_deepseek-r1_70b_physician"
    "C_repl_only_deepseek-r1_70b_realworld"
    "D_rlm_full_deepseek-r1_70b_physician"
    "D_rlm_full_deepseek-r1_70b_realworld"
)

VOLUME_FILES=$(modal volume ls rlm-triage-results raw 2>&1)

ALL_DONE=true
for f in "${EXPECTED_FILES[@]}"; do
    if echo "$VOLUME_FILES" | grep -q "$f"; then
        echo "  [x] $f.jsonl"
    else
        echo "  [ ] $f.jsonl  (MISSING)"
        ALL_DONE=false
    fi
done
echo ""

if $ALL_DONE; then
    echo "ALL FILES PRESENT. Run: bash finalize_paper.sh"
else
    echo "Some files still missing. Jobs may still be running."
fi
echo ""
echo "=========================================="

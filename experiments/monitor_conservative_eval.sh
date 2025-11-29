#!/bin/bash

echo "====================================================================="
echo "Monitoring Format A (conservative) - will auto-start Format B when complete"
echo "====================================================================="

LOG_FILE="/home/omeerdogan23/drugRAG/experiments/format_a_qwen_retry_conservative.log"

while true; do
    if grep -q "Results saved to" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "✅ Format A COMPLETED!"
        echo ""
        echo "📊 Format A Results:"
        echo "=============================================="
        grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_FILE" | head -15
        echo "=============================================="
        echo ""
        echo "🚀 Starting Format B in 10 seconds..."
        sleep 10

        cd /home/omeerdogan23/drugRAG/experiments
        /home/omeerdogan23/drugRAG/.venv/bin/python3 evaluate_vllm.py --architecture format_b_qwen --test_size 19520 2>&1 | tee format_b_qwen_conservative.log

        echo ""
        echo "✅ Format B COMPLETED!"
        echo ""
        echo "📊 Format B Results:"
        echo "=============================================="
        grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" format_b_qwen_conservative.log | head -15
        echo "=============================================="

        echo ""
        echo "📝 Creating comparison report..."
        cat > final_comparison_conservative.txt <<REPORT
========================================================================
FINAL QWEN EVALUATION - FULL DATASET (19,520 samples)
CONSERVATIVE SETTINGS to prevent OOM
========================================================================
Date: $(date)
Model: Qwen 2.5-7B-Instruct via vLLM
Settings:
  - GPU mem: 0.50 (down from 0.75)
  - max_seqs: 8 (down from 64)
  - max_batched_tokens: 4096 (down from 16384)
  - max_workers: 1 (down from 3)
  - max_model_len: 32768 (kept)

FORMAT A RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_FILE" | head -15)

FORMAT B RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" format_b_qwen_conservative.log | head -15)

========================================================================
REPORT

        echo "✅ All evaluations complete! Report: final_comparison_conservative.txt"
        cat final_comparison_conservative.txt
        break
    fi

    echo "$(date '+%H:%M:%S') - Format A running..."
    sleep 60
done

#!/bin/bash

echo "=========================================="
echo "Monitoring Format A + B with Ray Executor"
echo "=========================================="

LOG_A="/home/omeerdogan23/drugRAG/experiments/format_a_qwen_ray_final.log"
LOG_B="/home/omeerdogan23/drugRAG/experiments/format_b_qwen_ray_final.log"

# Wait for Format A log to exist
while [ ! -f "$LOG_A" ]; do
    echo "$(date '+%H:%M:%S') - Waiting for Format A to start..."
    sleep 10
done

echo "✅ Format A started!"

# Monitor Format A
while true; do
    if grep -q "Results saved to" "$LOG_A" 2>/dev/null; then
        echo ""
        echo "✅ Format A COMPLETED!"
        echo ""
        echo "📊 Format A Results:"
        echo "=============================================="
        grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_A" | head -15
        echo "=============================================="
        echo ""
        echo "🚀 Starting Format B in 10 seconds..."
        sleep 10

        cd /home/omeerdogan23/drugRAG/experiments
        /home/omeerdogan23/drugRAG/.venv/bin/python3 evaluate_vllm.py \
            --architecture format_b_qwen \
            --test_size 19520 \
            2>&1 | tee "$LOG_B"

        echo ""
        echo "✅ Format B COMPLETED!"
        echo ""
        echo "📊 Format B Results:"
        echo "=============================================="
        grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_B" | head -15
        echo "=============================================="

        echo ""
        echo "📝 Creating final comparison report..."
        cat > final_comparison_ray_executor.txt <<REPORT
========================================================================
FINAL QWEN EVALUATION - FULL DATASET (19,520 samples)
WITH RAY EXECUTOR (Fixed VllmWorker crash issue)
========================================================================
Date: $(date)
Model: Qwen 2.5-7B-Instruct via vLLM + Ray
Settings:
  - distributed_executor_backend: ray (was: mp)
  - GPU mem: 0.50
  - max_seqs: 8
  - max_batched_tokens: 4096
  - max_workers: 1
  - max_model_len: 32768

FORMAT A RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_A" | head -15)

FORMAT B RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_B" | head -15)

========================================================================
REPORT

        echo "✅ All evaluations complete! Report: final_comparison_ray_executor.txt"
        cat final_comparison_ray_executor.txt
        break
    fi

    echo "$(date '+%H:%M:%S') - Format A running..."
    sleep 60
done

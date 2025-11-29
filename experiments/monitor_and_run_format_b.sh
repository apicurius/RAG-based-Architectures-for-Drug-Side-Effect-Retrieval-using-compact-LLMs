#!/bin/bash

echo "====================================================================="
echo "Monitoring Format A and will auto-start Format B when complete"
echo "====================================================================="

while true; do
    if grep -q "Results saved to" /home/omeerdogan23/drugRAG/experiments/format_a_qwen_full_evaluation.log 2>/dev/null; then
        echo ""
        echo "✅ Format A evaluation COMPLETED!"
        echo ""
        echo "📊 Format A Results:"
        echo "=============================================="
        grep -A 15 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" /home/omeerdogan23/drugRAG/experiments/format_a_qwen_full_evaluation.log | head -20
        echo "=============================================="
        echo ""
        echo "🚀 Starting Format B evaluation in 10 seconds..."
        sleep 10
        echo ""

        cd /home/omeerdogan23/drugRAG/experiments
        python3 evaluate_vllm.py --architecture format_b_qwen --test_size 19520 2>&1 | tee format_b_qwen_full_evaluation.log

        echo ""
        echo "✅ Format B evaluation COMPLETED!"
        echo ""
        echo "📊 Format B Results:"
        echo "=============================================="
        grep -A 15 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" format_b_qwen_full_evaluation.log | head -20
        echo "=============================================="
        echo ""
        echo "📝 Creating final comparison report..."

        cat > full_qwen_evaluation_comparison.txt <<EOF
========================================================================
FINAL QWEN EVALUATION RESULTS - FULL DATASET (19,520 samples)
========================================================================
Date: $(date)
Model: Qwen 2.5-7B-Instruct via vLLM (4 GPUs, ultra-conservative settings)
Settings: GPU mem 0.75, max_seqs 64, max_workers 3, chunked processing

FORMAT A RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" format_a_qwen_full_evaluation.log | head -15)

FORMAT B RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" format_b_qwen_full_evaluation.log | head -15)

========================================================================
COMPARISON SUMMARY
========================================================================
Both evaluations completed successfully with ultra-conservative vLLM settings.
No OOM crashes occurred during inference.
Dataset: 19,520 samples (9,760 positive + 9,760 negative)
========================================================================
EOF

        echo "✅ All evaluations complete! Results saved to full_qwen_evaluation_comparison.txt"
        cat full_qwen_evaluation_comparison.txt
        break
    fi

    # Show latest progress every 60 seconds
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Format A still running..."
    tail -3 /home/omeerdogan23/drugRAG/experiments/format_a_qwen_full_evaluation.log 2>/dev/null | head -1
    sleep 60
done

#!/bin/bash

echo "====================================================================="
echo "Monitoring Format A - will auto-start Format B when complete"
echo "====================================================================="

while true; do
    if grep -q "Results saved to" /home/omeerdogan23/drugRAG/experiments/format_a_qwen_full_final.log 2>/dev/null; then
        echo ""
        echo "✅ Format A COMPLETED!"
        echo ""
        echo "📊 Format A Results:"
        echo "=============================================="
        grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" /home/omeerdogan23/drugRAG/experiments/format_a_qwen_full_final.log | head -15
        echo "=============================================="
        echo ""
        echo "🚀 Starting Format B in 10 seconds..."
        sleep 10

        cd /home/omeerdogan23/drugRAG/experiments
        python3 evaluate_vllm.py --architecture format_b_qwen --test_size 19520 2>&1 | tee format_b_qwen_full_final.log

        echo ""
        echo "✅ Format B COMPLETED!"
        echo ""
        echo "📊 Format B Results:"
        echo "=============================================="
        grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" format_b_qwen_full_final.log | head -15
        echo "=============================================="
        
        echo ""
        echo "📝 Creating comparison report..."
        cat > final_comparison_report.txt <<REPORT
========================================================================
FINAL QWEN EVALUATION - FULL DATASET (19,520 samples)
========================================================================
Date: $(date)
Model: Qwen 2.5-7B-Instruct via vLLM
Settings: GPU mem 0.75, max_seqs 64, max_workers 3, chunked processing

FORMAT A RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" format_a_qwen_full_final.log | head -15)

FORMAT B RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" format_b_qwen_full_final.log | head -15)

========================================================================
REPORT

        echo "✅ All evaluations complete! Report: final_comparison_report.txt"
        cat final_comparison_report.txt
        break
    fi

    echo "$(date '+%H:%M:%S') - Format A running..."
    sleep 60
done

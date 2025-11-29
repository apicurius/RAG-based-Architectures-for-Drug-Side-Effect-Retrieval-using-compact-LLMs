#!/bin/bash

# Script to run Format A, then restart vLLM, then run Format B
# This avoids the server longevity/freeze issue

LOG_A="/home/omeerdogan23/drugRAG/experiments/format_a_qwen_mp_final.log"
LOG_B="/home/omeerdogan23/drugRAG/experiments/format_b_qwen_mp_final.log"

echo "=========================================="
echo "Format A + B Sequential with Server Restart"
echo "=========================================="
echo ""

# Wait for vLLM server to be ready
echo "⏳ Waiting for vLLM server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8002/v1/models 2>/dev/null | grep -q "Qwen"; then
        echo "✅ vLLM Server is READY!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ Server did not start in 10 minutes. Exiting."
        exit 1
    fi
    echo "$(date '+%H:%M:%S') - Waiting for server (attempt $i/60)..."
    sleep 10
done

echo ""
echo "🚀 Starting Format A evaluation..."
echo "Started at: $(date)"
echo ""

cd /home/omeerdogan23/drugRAG/experiments
/home/omeerdogan23/drugRAG/.venv/bin/python3 evaluate_vllm.py \
    --architecture format_a_qwen \
    --test_size 19520 \
    2>&1 | tee "$LOG_A"

echo ""
echo "✅ Format A COMPLETED!"
echo "Completed at: $(date)"
echo ""
echo "📊 Format A Results:"
echo "=============================================="
grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_A" | head -15
echo "=============================================="

echo ""
echo "⏸️  Restarting vLLM server for Format B..."
echo ""

# Kill current vLLM server
pkill -9 -f "port 8002"
sleep 5

# Restart vLLM server
cd /home/omeerdogan23/drugRAG
bash qwen.sh > vllm_format_b_restart.log 2>&1 &

# Wait for server to be ready again
echo "⏳ Waiting for vLLM server to restart..."
for i in {1..60}; do
    if curl -s http://localhost:8002/v1/models 2>/dev/null | grep -q "Qwen"; then
        echo "✅ vLLM Server is READY again!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ Server did not restart in 10 minutes. Exiting."
        exit 1
    fi
    echo "$(date '+%H:%M:%S') - Waiting for server restart (attempt $i/60)..."
    sleep 10
done

echo ""
echo "🚀 Starting Format B evaluation..."
echo "Started at: $(date)"
echo ""

cd /home/omeerdogan23/drugRAG/experiments
/home/omeerdogan23/drugRAG/.venv/bin/python3 evaluate_vllm.py \
    --architecture format_b_qwen \
    --test_size 19520 \
    2>&1 | tee "$LOG_B"

echo ""
echo "✅ Format B COMPLETED!"
echo "Completed at: $(date)"
echo ""
echo "📊 Format B Results:"
echo "=============================================="
grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_B" | head -15
echo "=============================================="

echo ""
echo "📝 Creating final comparison report..."
cat > final_comparison_mp_sequential.txt <<REPORT
========================================================================
FINAL QWEN EVALUATION - FULL DATASET (19,520 samples)
MULTIPROCESS EXECUTOR with SERVER RESTART between evaluations
========================================================================
Date: $(date)
Model: Qwen 2.5-7B-Instruct via vLLM
Settings:
  - distributed_executor_backend: mp (multiprocess)
  - GPU mem: 0.50
  - max_seqs: 8
  - max_batched_tokens: 4096
  - max_workers: 1
  - max_model_len: 32768
  - Server restarted between Format A and B to avoid longevity issues

FORMAT A RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_A" | head -15)

FORMAT B RESULTS:
-----------------
$(grep -A 12 "COMPREHENSIVE BINARY CLASSIFICATION RESULTS" "$LOG_B" | head -15)

========================================================================
REPORT

echo ""
echo "✅ ALL EVALUATIONS COMPLETE!"
echo "Final report: final_comparison_mp_sequential.txt"
echo ""
cat final_comparison_mp_sequential.txt

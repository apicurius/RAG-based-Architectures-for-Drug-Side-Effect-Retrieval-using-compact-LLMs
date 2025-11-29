#!/bin/bash

# Llama-3.1-8B-Instruct vLLM Server - 4 GPU Configuration
# Based on 2025 best practices from official documentation

echo "Starting Llama-3.1-8B-Instruct on port 8003 with 4 GPUs..."

# Check if already running
if curl -s http://localhost:8003/v1/models 2>/dev/null | grep -q "Llama"; then
    echo "✅ Llama server already running on port 8003"
    exit 0
fi

# Clean up any stuck processes
pkill -9 -f "port 8003" 2>/dev/null
sleep 2

# Environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start vLLM server with recommended 2025 configuration
/home/omeerdogan23/drugRAG/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8003 \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --enforce-eager \
    --max-num-seqs 256 \
    --distributed-executor-backend mp \
    > /home/omeerdogan23/drugRAG/logs/llama_server.log 2>&1 &

echo "🚀 Llama server starting in background..."
echo "📋 Logs: /home/omeerdogan23/drugRAG/logs/llama_server.log"
echo "⏳ Server will be ready in 2-3 minutes..."

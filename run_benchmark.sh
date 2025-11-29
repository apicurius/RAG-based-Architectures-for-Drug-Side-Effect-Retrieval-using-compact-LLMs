#!/bin/bash

# DrugRAG Benchmark Runner
# Usage: ./run_benchmark.sh [experiment_type] [options]

set -e

function show_help {
    echo "Usage: ./run_benchmark.sh [command]"
    echo ""
    echo "Commands:"
    echo "  reverse       Run Reverse Query Benchmark"
    echo "  binary        Run Binary Query Benchmark (vLLM)"
    echo "  misspelling   Run Misspelling Evaluation"
    echo "  server        Start vLLM Server (helper)"
    echo ""
    echo "Examples:"
    echo "  ./run_benchmark.sh reverse"
    echo "  ./run_benchmark.sh binary --model qwen --architecture graphrag"
}

if [ -z "$1" ]; then
    show_help
    exit 1
fi

COMMAND=$1
shift

case $COMMAND in
    reverse)
        echo "Starting Reverse Query Benchmark..."
        python3 experiments/reverse_query_benchmark.py "$@"
        ;;
    binary)
        echo "Starting Binary Query Benchmark (vLLM)..."
        python3 experiments/evaluate_vllm.py "$@"
        ;;
    misspelling)
        echo "Starting Misspelling Evaluation..."
        python3 experiments/evaluate_misspelling.py "$@"
        ;;
    server)
        echo "Starting vLLM Server..."
        # Check if start_vllm_server.sh exists, if not create a simple one or warn
        if [ -f "./start_vllm_server.sh" ]; then
            ./start_vllm_server.sh "$@"
        else
            echo "Error: start_vllm_server.sh not found."
            exit 1
        fi
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

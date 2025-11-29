#!/bin/bash

# Bundled Evaluation Demo Script
# Demonstrates various use cases of the bundled vLLM + evaluation system

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  $1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════════╝${NC}\n"
}

print_step() {
    echo -e "${CYAN}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Navigate to project root
cd "$(dirname "$0")/.."

print_header "BUNDLED VLLM + EVALUATION SYSTEM - DEMO"

echo "This demo shows different use cases of the bundled evaluation system."
echo "Each example is a complete workflow that:"
echo "  1. Auto-starts vLLM server (if needed)"
echo "  2. Runs evaluation"
echo "  3. Manages cleanup"
echo ""
read -p "Press Enter to continue..."

# Demo 1: Quick Test with Auto-Shutdown
print_header "DEMO 1: Quick Test with Auto-Shutdown"
print_step "Use case: Fast iteration during development"
print_step "Sample: 50 queries"
print_step "Cleanup: Auto-shutdown after completion"
echo ""

cat << 'EOF'
Command:
./run_bundled_eval.sh \
    --model qwen \
    --arch format_a_qwen \
    --size 50 \
    --shutdown

What happens:
✓ Checks if Qwen server is running (port 8002)
✓ Starts server if needed (4-GPU tensor-parallel)
✓ Waits for health check
✓ Runs evaluation on 50 queries
✓ Saves results to JSON
✓ Shuts down server
✓ Cleans up resources

Best for: Quick testing, CI/CD, clean environments
EOF

echo ""
read -p "Run this demo? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./run_bundled_eval.sh --model qwen --arch format_a_qwen --size 50 --shutdown
    print_success "Demo 1 completed"
else
    print_warning "Demo 1 skipped"
fi

# Demo 2: Keep Server for Multiple Runs
print_header "DEMO 2: Keep Server for Multiple Runs"
print_step "Use case: Running multiple evaluations in sequence"
print_step "Sample: 100 queries per architecture"
print_step "Cleanup: Keep server running between runs"
echo ""

cat << 'EOF'
Commands:
./run_bundled_eval.sh --model llama3 --arch format_a_llama3 --size 100 --keep
./run_bundled_eval.sh --model llama3 --arch format_b_llama3 --size 100 --keep
./run_bundled_eval.sh --model llama3 --arch graphrag_llama3 --size 100 --shutdown

What happens:
✓ First run: Starts LLAMA3 server, runs eval, keeps server
✓ Second run: Detects existing server, runs eval, keeps server
✓ Third run: Uses existing server, runs eval, shuts down

Best for: Comparing architectures, benchmarking, batch evaluations
EOF

echo ""
read -p "Run this demo? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Running first evaluation (Format A)..."
    ./run_bundled_eval.sh --model llama3 --arch format_a_llama3 --size 100 --keep

    print_step "Running second evaluation (Format B)..."
    ./run_bundled_eval.sh --model llama3 --arch format_b_llama3 --size 100 --keep

    print_step "Running third evaluation (GraphRAG) with shutdown..."
    ./run_bundled_eval.sh --model llama3 --arch graphrag_llama3 --size 100 --shutdown

    print_success "Demo 2 completed - server automatically managed across 3 runs"
else
    print_warning "Demo 2 skipped"
fi

# Demo 3: Python API Usage
print_header "DEMO 3: Python API Direct Usage"
print_step "Use case: Programmatic access from Python"
print_step "Sample: 50 queries"
echo ""

cat << 'EOF'
Python code:
from experiments.run_reverse_binary_bundled import VLLMServerManager
from experiments.evaluate_reverse_binary import ReverseBinaryEvaluator

# Context manager automatically handles startup/shutdown
with VLLMServerManager('qwen') as server:
    server.start_server()

    evaluator = ReverseBinaryEvaluator()
    result = evaluator.evaluate('format_a_qwen', test_size=50)

    print(f"Accuracy: {result['accuracy']:.2%}")
    print(f"F1 Score: {result['f1_score']:.4f}")

# Server automatically shut down here

Best for: Custom workflows, Jupyter notebooks, automation scripts
EOF

echo ""
read -p "Run this demo? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 << 'PYTHON_EOF'
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.run_reverse_binary_bundled import VLLMServerManager
from experiments.evaluate_reverse_binary import ReverseBinaryEvaluator

print("\n🚀 Starting Python API demo...\n")

with VLLMServerManager('qwen') as server:
    server.start_server()

    evaluator = ReverseBinaryEvaluator()
    result = evaluator.evaluate('format_a_qwen', test_size=50)

    print(f"\n📊 Results:")
    print(f"   Accuracy:  {result['accuracy']:.2%}")
    print(f"   Precision: {result['precision']:.4f}")
    print(f"   Recall:    {result['recall']:.4f}")
    print(f"   F1 Score:  {result['f1_score']:.4f}")
    print(f"   Speed:     {result['queries_per_sec']:.1f} queries/sec")

print("\n✅ Server automatically cleaned up via context manager")
PYTHON_EOF
    print_success "Demo 3 completed"
else
    print_warning "Demo 3 skipped"
fi

# Demo 4: Custom Output and Timeout
print_header "DEMO 4: Custom Configuration"
print_step "Use case: Production deployment with custom settings"
print_step "Features: Custom timeout, output path, extended evaluation"
echo ""

cat << 'EOF'
Command:
./run_bundled_eval.sh \
    --model qwen \
    --arch enhanced_graphrag_qwen \
    --size 200 \
    --timeout 300 \
    --output ./custom_results/production_eval_$(date +%Y%m%d).json \
    --shutdown

What happens:
✓ Extended 5-minute timeout for server startup
✓ Custom output path with date stamp
✓ 200-query evaluation sample
✓ Enhanced GraphRAG architecture
✓ Clean shutdown

Best for: Production deployments, scheduled jobs, archival
EOF

echo ""
read -p "Run this demo? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    mkdir -p custom_results
    ./run_bundled_eval.sh \
        --model qwen \
        --arch enhanced_graphrag_qwen \
        --size 200 \
        --timeout 300 \
        --output "./custom_results/production_eval_$(date +%Y%m%d).json" \
        --shutdown
    print_success "Demo 4 completed - check custom_results/ directory"
else
    print_warning "Demo 4 skipped"
fi

# Summary
print_header "DEMO COMPLETE - SUMMARY"

echo "You've seen the bundled evaluation system in action:"
echo ""
echo "✅ Quick Test          - Fast iteration with auto-cleanup"
echo "✅ Multiple Runs       - Keep server alive for efficiency"
echo "✅ Python API          - Programmatic control"
echo "✅ Custom Config       - Production-ready configuration"
echo ""
echo "Key Benefits:"
echo "  • Automatic server management (start/health check/shutdown)"
echo "  • No manual steps required"
echo "  • Flexible cleanup options (keep vs. shutdown)"
echo "  • Fully compatible with existing evaluation pipeline"
echo "  • Context manager guarantees cleanup"
echo ""
echo "Next Steps:"
echo "  📖 Read the full guide: BUNDLED_EVAL_GUIDE.md"
echo "  🚀 Run your own evaluation: ./run_bundled_eval.sh --help"
echo "  🔬 Explore Python API: experiments/run_reverse_binary_bundled.py"
echo ""
print_success "Happy evaluating! 🎉"

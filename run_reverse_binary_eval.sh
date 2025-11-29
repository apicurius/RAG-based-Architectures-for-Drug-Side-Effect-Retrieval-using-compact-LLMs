#!/bin/bash

# Reverse Binary Query Evaluation Script
# Evaluates reverse_queries_binary.csv with all DrugRAG architectures

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
DEFAULT_LLM="llama3"
DEFAULT_STRATEGY="all"
DEFAULT_TEST_SIZE=""  # Empty means use all 1,200 queries

# Variables
LLM=""
STRATEGY=""
TEST_SIZE=""
AUTO_START_SERVER=true

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
show_usage() {
    echo "╔═══════════════════════════════════════════════════════════════════════════════════╗"
    echo "║            REVERSE BINARY QUERY EVALUATION FOR DRUGRAG                          ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Evaluates the reverse_queries_binary.csv dataset (1,200 queries)"
    echo "Format: side_effect → drug → YES/NO"
    echo "Example: 'Which drugs cause dizziness?' → 'octreotide' → YES"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --llm MODEL           LLM model to use (qwen|llama3|both) [default: $DEFAULT_LLM]"
    echo "  --strategy ARCH       Architecture strategy [default: $DEFAULT_STRATEGY]"
    echo "                        Options: pure|format_a|format_b|graphrag|"
    echo "                                 enhanced_b|enhanced_graphrag|all"
    echo "  --test-size N         Number of queries to test [default: all 1,200]"
    echo "  --no-auto-start       Don't automatically start vLLM servers"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Available Architectures:"
    echo "  pure            : Pure LLM without RAG"
    echo "  format_a        : RAG Format A - Drug → [side effects]"
    echo "  format_b        : RAG Format B - Drug-effect pairs"
    echo "  graphrag        : GraphRAG with Neo4j"
    echo "  enhanced_b      : Enhanced Format B with metadata"
    echo "  enhanced_graphrag : Enhanced GraphRAG with CoT reasoning"
    echo "  all             : Run ALL architectures (12 total = 6 architectures × 2 LLMs)"
    echo ""
    echo "Examples:"
    echo "  # Quick test - 100 queries with LLAMA3 GraphRAG"
    echo "  $0 --llm llama3 --strategy graphrag --test-size 100"
    echo ""
    echo "  # Full evaluation - all 1,200 queries with Qwen Format B"
    echo "  $0 --llm qwen --strategy format_b"
    echo ""
    echo "  # Evaluate all architectures with both LLMs (12 runs)"
    echo "  $0 --llm both --strategy all"
    echo ""
    echo "  # Test specific enhanced architecture"
    echo "  $0 --llm both --strategy enhanced_graphrag --test-size 200"
    echo ""
}

# Function to parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --llm)
                LLM="$2"
                shift 2
                ;;
            --strategy)
                STRATEGY="$2"
                shift 2
                ;;
            --test-size)
                TEST_SIZE="$2"
                shift 2
                ;;
            --no-auto-start)
                AUTO_START_SERVER=false
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_color $RED "❌ Unknown option: $1"
                echo ""
                show_usage
                exit 1
                ;;
        esac
    done
}

# Function to check if vLLM server is running
check_vllm_server() {
    local port=$1
    if curl -s http://localhost:$port/v1/models > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to run evaluation for a specific architecture
run_evaluation() {
    local architecture=$1
    local test_size_arg=""

    if [[ -n "$TEST_SIZE" ]]; then
        test_size_arg="--test-size $TEST_SIZE"
    fi

    print_color $CYAN "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_color $CYAN "   Evaluating: $architecture"
    print_color $CYAN "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

    cd experiments
    python evaluate_reverse_binary.py \
        --architecture "$architecture" \
        $test_size_arg

    if [ $? -eq 0 ]; then
        print_color $GREEN "✅ Completed: $architecture"
    else
        print_color $RED "❌ Failed: $architecture"
    fi
    cd ..
}

# Main execution
main() {
    # Parse arguments
    parse_arguments "$@"

    # Set defaults if not provided
    LLM=${LLM:-$DEFAULT_LLM}
    STRATEGY=${STRATEGY:-$DEFAULT_STRATEGY}

    # Validate LLM
    if [[ ! "$LLM" =~ ^(qwen|llama3|both)$ ]]; then
        print_color $RED "❌ Invalid LLM: $LLM. Must be qwen, llama3, or both"
        exit 1
    fi

    # Validate strategy
    if [[ ! "$STRATEGY" =~ ^(pure|format_a|format_b|graphrag|enhanced_b|enhanced_graphrag|all)$ ]]; then
        print_color $RED "❌ Invalid strategy: $STRATEGY"
        exit 1
    fi

    # Print configuration
    print_color $BLUE "╔═══════════════════════════════════════════════════════════════════════════════════╗"
    print_color $BLUE "║               REVERSE BINARY QUERY EVALUATION - CONFIGURATION                    ║"
    print_color $BLUE "╚═══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    print_color $YELLOW "Dataset:      reverse_queries_binary.csv (1,200 queries)"
    print_color $YELLOW "LLM Model:    $LLM"
    print_color $YELLOW "Strategy:     $STRATEGY"
    print_color $YELLOW "Test Size:    ${TEST_SIZE:-All 1,200 queries}"
    echo ""

    # Check vLLM servers if auto-start is enabled
    if [ "$AUTO_START_SERVER" = true ]; then
        if [[ "$LLM" == "qwen" ]] || [[ "$LLM" == "both" ]]; then
            if ! check_vllm_server 8002; then
                print_color $YELLOW "⚠️  Qwen vLLM server not running on port 8002"
                print_color $YELLOW "   Please start it with: ./qwen.sh"
            else
                print_color $GREEN "✅ Qwen vLLM server is running"
            fi
        fi

        if [[ "$LLM" == "llama3" ]] || [[ "$LLM" == "both" ]]; then
            if ! check_vllm_server 8003; then
                print_color $YELLOW "⚠️  LLAMA3 vLLM server not running on port 8003"
                print_color $YELLOW "   Please start it with: ./llama.sh"
            else
                print_color $GREEN "✅ LLAMA3 vLLM server is running"
            fi
        fi
    fi

    echo ""
    print_color $GREEN "Starting evaluation..."
    echo ""

    # Build architecture list based on strategy and LLM
    architectures=()

    if [[ "$STRATEGY" == "all" ]]; then
        # All architectures
        base_archs=("pure_llm" "format_a" "format_b" "graphrag" "enhanced_format_b" "enhanced_graphrag")
    else
        # Single strategy
        base_archs=("${STRATEGY}")
        if [[ "$STRATEGY" == "enhanced_b" ]]; then
            base_archs=("enhanced_format_b")
        fi
    fi

    # Combine with LLM models
    for arch in "${base_archs[@]}"; do
        if [[ "$LLM" == "both" ]]; then
            architectures+=("${arch}_qwen")
            architectures+=("${arch}_llama3")
        elif [[ "$LLM" == "qwen" ]]; then
            architectures+=("${arch}_qwen")
        else
            architectures+=("${arch}_llama3")
        fi
    done

    # Run evaluations
    print_color $CYAN "📊 Running ${#architectures[@]} evaluation(s)..."
    echo ""

    for arch in "${architectures[@]}"; do
        run_evaluation "$arch"
    done

    # Summary
    print_color $BLUE "\n╔═══════════════════════════════════════════════════════════════════════════════════╗"
    print_color $BLUE "║                         EVALUATION COMPLETE                                      ║"
    print_color $BLUE "╚═══════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    print_color $GREEN "✅ Completed ${#architectures[@]} evaluation(s)"
    print_color $YELLOW "📁 Results saved in: experiments/results_reverse_binary_*.json"
    echo ""
}

# Run main function
main "$@"

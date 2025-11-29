#!/bin/bash
# Convenient script to run chunked strategy evaluation

# Wait for vLLM server to be ready
echo "Waiting for vLLM server on port 8002..."
while ! curl -s http://localhost:8002/health > /dev/null 2>&1; do
    echo "  Server not ready yet, waiting 10s..."
    sleep 10
done

echo "✅ vLLM server is ready!"
echo ""
echo "🔬 Starting chunked strategy evaluation..."
echo "  This will test both monolithic and chunked approaches on 5 queries:"
echo "    1. dry mouth (462 pairs)"
echo "    2. nausea (915 pairs) - LARGE"
echo "    3. candida infection (142 pairs) - SMALL"
echo "    4. thrombocytopenia (517 pairs)"
echo "    5. increased blood pressure (0 pairs) - CONTROL"
echo ""

cd experiments
python evaluate_chunked_strategy.py --model qwen --strategy both

echo ""
echo "✅ Evaluation complete! Check the results file."

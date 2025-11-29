#!/usr/bin/env python3
"""
Evaluate Chunked vs Monolithic Extraction Strategies for Reverse Queries

This script tests the new chunked iterative extraction approach against
the original monolithic approach on the 5 representative queries from
the REVERSE_QUERY_FINAL_SUMMARY.md documentation.

Expected improvement:
- Small queries (<200 pairs): Similar performance (~87% recall)
- Medium queries (200-600 pairs): Moderate improvement (~69% → ~80% recall)
- Large queries (>600 pairs): Significant improvement (~49% → ~85% recall)
"""

import sys
import os
import json
import time
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.architectures.rag_format_b import FormatBRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Ground truth from data/processed/reverse_queries.csv
GROUND_TRUTH = {
    'dry mouth': 543,
    'nausea': 1140,
    'candida infection': 162,
    'thrombocytopenia': 589,
    'increased blood pressure': 0  # Control (no results expected)
}


def evaluate_strategy(strategy: str, model: str = "qwen"):
    """
    Evaluate a single extraction strategy on all test queries

    Args:
        strategy: "monolithic" or "chunked"
        model: "qwen" or "llama3"

    Returns:
        Dict with results for all queries
    """
    logger.info("="*80)
    logger.info(f"EVALUATING STRATEGY: {strategy.upper()}")
    logger.info(f"Model: {model}")
    logger.info("="*80)

    # Initialize Format B RAG
    config_path = "../config.json"
    rag = FormatBRAG(config_path, model=model)

    results = {}
    total_start = time.time()

    for side_effect, expected_count in GROUND_TRUTH.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Query: Which drugs cause {side_effect}?")
        logger.info(f"Expected drugs: {expected_count}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"{'='*80}")

        # Run query
        start_time = time.time()
        result = rag.reverse_query(side_effect, strategy=strategy)
        elapsed = time.time() - start_time

        # Extract results
        extracted_drugs = result.get('drugs', [])
        extracted_count = len(extracted_drugs)
        retrieved_pairs = result.get('retrieved_pairs', 0)

        # Calculate metrics (simplified - full evaluation needs ground truth drug list)
        # For now, we'll use drug count as a proxy
        recall_estimate = min(extracted_count / expected_count, 1.0) if expected_count > 0 else 0.0

        # Log results
        logger.info(f"\n📊 RESULTS:")
        logger.info(f"   Retrieved pairs: {retrieved_pairs}")

        if strategy == "chunked":
            chunks = result.get('chunks_processed', 0)
            chunk_size = result.get('chunk_size', 0)
            logger.info(f"   Chunks processed: {chunks} (size: {chunk_size})")

        logger.info(f"   Extracted drugs: {extracted_count}")
        logger.info(f"   Expected drugs: {expected_count}")
        logger.info(f"   Recall estimate: {recall_estimate:.2%}")
        logger.info(f"   Time: {elapsed:.2f}s")

        # Store results
        results[side_effect] = {
            'side_effect': side_effect,
            'strategy': strategy,
            'model': model,
            'expected_count': expected_count,
            'extracted_count': extracted_count,
            'retrieved_pairs': retrieved_pairs,
            'recall_estimate': recall_estimate,
            'elapsed_time': elapsed,
            'chunks_processed': result.get('chunks_processed'),
            'chunk_size': result.get('chunk_size'),
            'sample_drugs': extracted_drugs[:10] if extracted_drugs else []
        }

    total_elapsed = time.time() - total_start

    # Calculate aggregate metrics
    avg_recall = sum(r['recall_estimate'] for r in results.values()) / len(results)
    total_extracted = sum(r['extracted_count'] for r in results.values())
    total_expected = sum(r['expected_count'] for r in results.values())

    logger.info(f"\n{'='*80}")
    logger.info(f"AGGREGATE RESULTS - {strategy.upper()}")
    logger.info(f"{'='*80}")
    logger.info(f"Average recall estimate: {avg_recall:.2%}")
    logger.info(f"Total extracted: {total_extracted}")
    logger.info(f"Total expected: {total_expected}")
    logger.info(f"Total time: {total_elapsed:.2f}s")
    logger.info(f"{'='*80}\n")

    return {
        'strategy': strategy,
        'model': model,
        'queries': results,
        'aggregate': {
            'avg_recall_estimate': avg_recall,
            'total_extracted': total_extracted,
            'total_expected': total_expected,
            'total_time': total_elapsed
        },
        'timestamp': datetime.now().isoformat()
    }


def compare_strategies(model: str = "qwen"):
    """
    Compare monolithic vs chunked strategies

    Args:
        model: "qwen" or "llama3"
    """
    logger.info("🔬 CHUNKED STRATEGY EVALUATION")
    logger.info(f"Testing 5 representative queries with both strategies")
    logger.info(f"Model: {model}\n")

    # Evaluate both strategies
    monolithic_results = evaluate_strategy("monolithic", model)
    chunked_results = evaluate_strategy("chunked", model)

    # Compare results
    logger.info("\n" + "="*80)
    logger.info("STRATEGY COMPARISON")
    logger.info("="*80)
    logger.info(f"{'Side Effect':<30} {'Pairs':<8} {'Monolithic':<12} {'Chunked':<12} {'Improvement':<12}")
    logger.info("-"*80)

    improvements = []

    for side_effect in GROUND_TRUTH.keys():
        mono = monolithic_results['queries'][side_effect]
        chunk = chunked_results['queries'][side_effect]

        pairs = mono['retrieved_pairs']
        mono_extracted = mono['extracted_count']
        chunk_extracted = chunk['extracted_count']
        improvement = ((chunk_extracted - mono_extracted) / mono_extracted * 100) if mono_extracted > 0 else 0.0

        improvements.append(improvement)

        logger.info(
            f"{side_effect:<30} {pairs:<8} "
            f"{mono_extracted:<12} {chunk_extracted:<12} "
            f"{improvement:+.1f}%"
        )

    logger.info("-"*80)

    # Aggregate comparison
    mono_agg = monolithic_results['aggregate']
    chunk_agg = chunked_results['aggregate']

    recall_improvement = ((chunk_agg['avg_recall_estimate'] - mono_agg['avg_recall_estimate'])
                          / mono_agg['avg_recall_estimate'] * 100) if mono_agg['avg_recall_estimate'] > 0 else 0.0

    time_overhead = ((chunk_agg['total_time'] - mono_agg['total_time'])
                     / mono_agg['total_time'] * 100) if mono_agg['total_time'] > 0 else 0.0

    logger.info(f"\nAGGREGATE METRICS:")
    logger.info(f"  Monolithic avg recall: {mono_agg['avg_recall_estimate']:.2%}")
    logger.info(f"  Chunked avg recall:    {chunk_agg['avg_recall_estimate']:.2%}")
    logger.info(f"  Recall improvement:    {recall_improvement:+.1f}%")
    logger.info(f"\n  Monolithic time:       {mono_agg['total_time']:.2f}s")
    logger.info(f"  Chunked time:          {chunk_agg['total_time']:.2f}s")
    logger.info(f"  Time overhead:         {time_overhead:+.1f}%")
    logger.info("="*80)

    # Save results
    output = {
        'monolithic': monolithic_results,
        'chunked': chunked_results,
        'comparison': {
            'recall_improvement_pct': recall_improvement,
            'time_overhead_pct': time_overhead,
            'per_query_improvements': {
                side_effect: improvement
                for side_effect, improvement in zip(GROUND_TRUTH.keys(), improvements)
            }
        }
    }

    output_file = f"results_chunked_strategy_comparison_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n✅ Results saved to: {output_file}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare monolithic vs chunked extraction strategies"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='qwen',
        choices=['qwen', 'llama3'],
        help='Model to use (default: qwen)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='both',
        choices=['monolithic', 'chunked', 'both'],
        help='Strategy to evaluate (default: both)'
    )

    args = parser.parse_args()

    if args.strategy == 'both':
        compare_strategies(args.model)
    else:
        evaluate_strategy(args.strategy, args.model)

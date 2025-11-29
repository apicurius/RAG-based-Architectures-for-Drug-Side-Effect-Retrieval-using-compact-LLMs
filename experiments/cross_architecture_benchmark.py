#!/usr/bin/env python3
"""
Cross-Architecture Benchmark Evaluation

Compares all RAG architectures on a stratified sample from the comprehensive dataset.

Architectures tested:
1. Enhanced Format B (chunked) - Current champion
2. Standard Format B (chunked) - Baseline
3. Enhanced GraphRAG - Hybrid approach
4. Format A - Legacy comparison

Metrics collected:
- Accuracy: Recall, Precision, F1
- Performance: Latency per query
- Cost: API calls and tokens

Usage:
    python cross_architecture_benchmark.py
"""

import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ground_truth(ground_truth_path: str = '../data/processed/neo4j_ground_truth.json') -> Dict:
    """Load ground truth mappings"""
    logger.info(f"Loading ground truth from {ground_truth_path}")
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    logger.info(f"Loaded ground truth for {len(ground_truth)} side effects")
    return ground_truth


def create_stratified_sample(
    dataset_path: str = '../data/processed/comprehensive_reverse_queries_20251102_225909_case_corrected.json',
    sample_sizes: Dict[str, int] = None
) -> List[Dict]:
    """
    Create stratified sample from comprehensive dataset

    Default sampling:
    - large: All 31 (100% of available)
    - medium: 40 (14% of 288)
    - small: 40 (13% of 300)
    - rare: 10 (20% of 50)
    Total: ~121 queries
    """

    if sample_sizes is None:
        sample_sizes = {
            'large': 31,    # All available
            'medium': 40,
            'small': 40,
            'rare': 10
        }

    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Group by tier
    tier_queries = {}
    for ex in data['yes_examples']:
        tier = ex.get('tier', 'unknown')
        if tier not in tier_queries:
            tier_queries[tier] = []
        tier_queries[tier].append(ex)

    # Sample from each tier
    sampled_queries = []
    for tier, size in sample_sizes.items():
        if tier in tier_queries:
            available = len(tier_queries[tier])
            actual_size = min(size, available)

            # Random sample
            sample = random.sample(tier_queries[tier], actual_size)
            sampled_queries.extend(sample)

            logger.info(f"Sampled {actual_size}/{available} queries from {tier} tier")

    logger.info(f"Total sampled queries: {len(sampled_queries)}")
    return sampled_queries


def calculate_metrics(extracted: List[str], expected: List[str]) -> Dict[str, float]:
    """Calculate recall, precision, F1 with case-insensitive matching"""
    extracted_lower = set([d.lower() for d in extracted])
    expected_lower = set([d.lower() for d in expected])

    true_positives = len(extracted_lower & expected_lower)
    false_positives = len(extracted_lower - expected_lower)
    false_negatives = len(expected_lower - extracted_lower)

    recall = true_positives / len(expected_lower) if expected_lower else 0
    precision = true_positives / len(extracted_lower) if extracted_lower else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def evaluate_architecture(
    arch_name: str,
    arch_instance: Any,
    queries: List[Dict],
    ground_truth: Dict
) -> Dict:
    """Evaluate a single architecture on all queries"""

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating: {arch_name}")
    logger.info(f"{'='*80}")

    results = []
    total_start_time = time.time()

    for idx, query in enumerate(queries, 1):
        side_effect = query['side_effect']
        expected_drugs = ground_truth.get(side_effect, [])

        # Run query with timing
        start_time = time.time()
        try:
            result = arch_instance.reverse_query(side_effect)
            extracted_drugs = result.get('drugs', [])
            error = None
        except Exception as e:
            logger.error(f"Error processing {side_effect}: {str(e)}")
            extracted_drugs = []
            error = str(e)

        latency = time.time() - start_time

        # Calculate metrics
        metrics = calculate_metrics(extracted_drugs, expected_drugs)

        results.append({
            'side_effect': side_effect,
            'tier': query.get('tier', 'unknown'),
            'expected_count': len(expected_drugs),
            'extracted_count': len(extracted_drugs),
            **metrics,
            'latency': latency,
            'error': error
        })

        # Progress logging
        if idx % 10 == 0:
            avg_recall = sum(r['recall'] for r in results) / len(results)
            avg_latency = sum(r['latency'] for r in results) / len(results)
            logger.info(f"Progress: {idx}/{len(queries)} | Avg Recall: {avg_recall:.2%} | Avg Latency: {avg_latency:.2f}s")

    total_time = time.time() - total_start_time

    # Aggregate metrics
    summary = {
        'architecture': arch_name,
        'total_queries': len(queries),
        'successful_queries': sum(1 for r in results if r['error'] is None),
        'failed_queries': sum(1 for r in results if r['error'] is not None),
        'avg_recall': sum(r['recall'] for r in results) / len(results),
        'avg_precision': sum(r['precision'] for r in results) / len(results),
        'avg_f1': sum(r['f1'] for r in results) / len(results),
        'avg_latency': sum(r['latency'] for r in results) / len(results),
        'total_time': total_time,
        'queries_per_second': len(queries) / total_time,
        'detailed_results': results
    }

    # Tier-specific metrics
    tier_metrics = {}
    for tier in ['large', 'medium', 'small', 'rare']:
        tier_results = [r for r in results if r['tier'] == tier]
        if tier_results:
            tier_metrics[tier] = {
                'count': len(tier_results),
                'avg_recall': sum(r['recall'] for r in tier_results) / len(tier_results),
                'avg_latency': sum(r['latency'] for r in tier_results) / len(tier_results)
            }

    summary['tier_metrics'] = tier_metrics

    logger.info(f"\n{arch_name} Summary:")
    logger.info(f"  Avg Recall:    {summary['avg_recall']:.2%}")
    logger.info(f"  Avg Precision: {summary['avg_precision']:.2%}")
    logger.info(f"  Avg F1:        {summary['avg_f1']:.2%}")
    logger.info(f"  Avg Latency:   {summary['avg_latency']:.2f}s")
    logger.info(f"  Total Time:    {total_time/60:.1f} minutes")

    return summary


def main():
    """Run cross-architecture benchmark"""

    print("="*80)
    print("CROSS-ARCHITECTURE BENCHMARK EVALUATION")
    print("="*80)
    print()

    # Load ground truth
    ground_truth = load_ground_truth()

    # Create stratified sample
    print("\n📊 Creating stratified sample...")
    sampled_queries = create_stratified_sample()

    # Save sample for reproducibility
    sample_file = f"../data/processed/benchmark_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(sample_file, 'w') as f:
        json.dump(sampled_queries, f, indent=2)
    logger.info(f"Sample saved to: {sample_file}")

    # Initialize architectures
    print("\n🔧 Initializing architectures...")

    from src.architectures.enhanced_rag_format_b import EnhancedRAGFormatB
    from src.architectures.rag_format_b import FormatBRAG
    from src.architectures.enhanced_graphrag import EnhancedGraphRAG
    from src.architectures.rag_format_a import FormatARAG

    architectures = {
        'Enhanced_Format_B': EnhancedRAGFormatB(),
        'Standard_Format_B': FormatBRAG(),
        'Enhanced_GraphRAG': EnhancedGraphRAG(),
        'Format_A': FormatARAG()
    }

    logger.info(f"Initialized {len(architectures)} architectures")

    # Run benchmarks
    all_results = {}

    for arch_name, arch_instance in architectures.items():
        try:
            results = evaluate_architecture(
                arch_name,
                arch_instance,
                sampled_queries,
                ground_truth
            )
            all_results[arch_name] = results
        except Exception as e:
            logger.error(f"Failed to evaluate {arch_name}: {str(e)}")
            all_results[arch_name] = {'error': str(e)}

    # Save results
    output_file = f"results_cross_architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Benchmark complete! Results saved to: {output_file}")
    print("\n📊 SUMMARY COMPARISON:")
    print("-" * 80)
    print(f"{'Architecture':<25} {'Recall':<10} {'Precision':<10} {'F1':<10} {'Latency':<10}")
    print("-" * 80)

    for arch_name, results in all_results.items():
        if 'error' not in results:
            print(f"{arch_name:<25} "
                  f"{results['avg_recall']:<10.2%} "
                  f"{results['avg_precision']:<10.2%} "
                  f"{results['avg_f1']:<10.2%} "
                  f"{results['avg_latency']:<10.2f}s")

    print("=" * 80)


if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(42)
    main()

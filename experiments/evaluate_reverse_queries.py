#!/usr/bin/env python3
"""
Reverse Query Evaluation for DrugRAG

Evaluates the reverse_queries.csv dataset using all available architectures.
This dataset contains reverse queries: "Which drugs cause [side_effect]?"

Usage:
    python evaluate_reverse_queries.py --architecture format_a_qwen --test_size 50
    python evaluate_reverse_queries.py --architecture all --test_size 100
    python evaluate_reverse_queries.py --architecture graphrag_qwen --test_size 10
"""

import argparse
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import time
import re
import ast
from typing import List, Dict, Any, Set
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vllm_model import VLLMQwenModel, VLLMLLAMA3Model
from src.architectures.rag_format_a import FormatARAG
from src.architectures.rag_format_b import FormatBRAG
from src.architectures.graphrag import GraphRAG

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_drug_name(drug: str) -> str:
    """
    Normalize drug names for comparison
    """
    if not drug:
        return ""

    # Lowercase
    drug = drug.lower().strip()

    # Remove common suffixes
    drug = re.sub(r'\s+(hydrochloride|sulfate|sodium|injection|tablet|capsule)$', '', drug)

    # Remove parentheses content
    drug = re.sub(r'\s*\([^)]*\)', '', drug)

    # Remove extra spaces
    drug = re.sub(r'\s+', ' ', drug).strip()

    return drug


def calculate_reverse_query_metrics(predicted: List[str], expected: List[str]) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for reverse queries

    Args:
        predicted: List of predicted drug names
        expected: List of expected drug names (ground truth)

    Returns:
        Dict with precision, recall, F1, coverage, hallucination_rate, exact_match
    """
    # Normalize all drug names
    predicted_set = set(normalize_drug_name(d) for d in predicted if d)
    expected_set = set(normalize_drug_name(d) for d in expected if d)

    # Remove empty strings
    predicted_set.discard("")
    expected_set.discard("")

    # Calculate intersection
    intersection = predicted_set & expected_set

    # Calculate metrics
    precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
    recall = len(intersection) / len(expected_set) if expected_set else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    coverage = recall * 100  # Same as recall, but in percentage
    hallucination_rate = len(predicted_set - expected_set) / len(predicted_set) * 100 if predicted_set else 0.0
    exact_match = 1.0 if predicted_set == expected_set else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'coverage': coverage,
        'hallucination_rate': hallucination_rate,
        'exact_match': exact_match,
        'true_positives': len(intersection),
        'false_positives': len(predicted_set - expected_set),
        'false_negatives': len(expected_set - predicted_set),
        'predicted_count': len(predicted_set),
        'expected_count': len(expected_set)
    }


class ReverseQueryEvaluator:
    """Evaluator for reverse queries (side_effect -> drugs)"""

    def __init__(self, config_path: str = "../config.json"):
        self.config_path = config_path
        self.dataset_path = "../data/processed/reverse_queries.csv"

    def load_dataset(self, test_size: int = None):
        """Load reverse queries dataset"""
        logger.info(f"📂 Loading dataset: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)

        # Parse expected_drugs from string representation of list
        logger.info("   Parsing expected drug lists...")
        df['expected_drugs_parsed'] = df['expected_drugs'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        # Sample if test_size specified
        if test_size and test_size < len(df):
            logger.info(f"   Sampling {test_size} examples...")
            df = df.sample(n=test_size, random_state=42).reset_index(drop=True)

        logger.info(f"✅ Loaded {len(df)} examples")
        logger.info(f"   Unique side effects: {df['side_effect'].nunique()}")
        logger.info(f"   Avg drugs per side effect: {df['drug_count'].mean():.1f}")
        logger.info(f"   Max drugs per side effect: {df['drug_count'].max()}")
        logger.info(f"   Min drugs per side effect: {df['drug_count'].min()}")

        return df

    def initialize_architecture(self, architecture: str):
        """Initialize the specified architecture"""
        logger.info(f"🔧 Initializing architecture: {architecture}")

        if architecture == 'format_a_qwen':
            return FormatARAG(self.config_path, model="qwen")
        elif architecture == 'format_a_llama3':
            return FormatARAG(self.config_path, model="llama3")
        elif architecture == 'format_b_qwen':
            return FormatBRAG(self.config_path, model="qwen")
        elif architecture == 'format_b_llama3':
            return FormatBRAG(self.config_path, model="llama3")
        elif architecture == 'graphrag_qwen':
            return GraphRAG(self.config_path, model="qwen")
        elif architecture == 'graphrag_llama3':
            return GraphRAG(self.config_path, model="llama3")
        elif architecture == 'pure_llm_qwen':
            return VLLMQwenModel(self.config_path)
        elif architecture == 'pure_llm_llama3':
            return VLLMLLAMA3Model(self.config_path)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def evaluate(self, architecture: str, test_size: int = None):
        """Run evaluation for a single architecture"""
        logger.info("="*80)
        logger.info(f"REVERSE QUERY EVALUATION")
        logger.info(f"Architecture: {architecture}")
        logger.info(f"Dataset: reverse_queries.csv")
        logger.info("="*80)

        # Load dataset
        dataset = self.load_dataset(test_size)

        # Initialize architecture
        arch = self.initialize_architecture(architecture)

        # Check if architecture supports reverse queries
        if not hasattr(arch, 'reverse_query'):
            logger.error(f"❌ Architecture {architecture} does not support reverse_query method")
            return None

        # Start timing
        start_time = time.time()

        # Process queries
        results = []
        logger.info(f"\n🚀 Processing {len(dataset)} reverse queries...")

        for idx, row in tqdm(dataset.iterrows(), total=len(dataset), desc="🔍 Processing", unit="query"):
            side_effect = row['side_effect']
            expected_drugs = row['expected_drugs_parsed']

            try:
                result = arch.reverse_query(side_effect)
                predicted_drugs = result.get('drugs', [])

                # Calculate metrics for this query
                metrics = calculate_reverse_query_metrics(predicted_drugs, expected_drugs)

                results.append({
                    'side_effect': side_effect,
                    'predicted_drugs': predicted_drugs,
                    'expected_drugs': expected_drugs,
                    'metrics': metrics,
                    'result': result
                })

            except Exception as e:
                logger.error(f"Error processing {side_effect}: {e}")
                results.append({
                    'side_effect': side_effect,
                    'predicted_drugs': [],
                    'expected_drugs': expected_drugs,
                    'error': str(e)
                })

        # Calculate aggregate metrics
        elapsed_time = time.time() - start_time

        # Aggregate metrics across all queries
        all_metrics = [r['metrics'] for r in results if 'metrics' in r]

        if not all_metrics:
            logger.error("No valid results to aggregate")
            return None

        avg_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
            'coverage': np.mean([m['coverage'] for m in all_metrics]),
            'hallucination_rate': np.mean([m['hallucination_rate'] for m in all_metrics]),
            'exact_match_rate': np.mean([m['exact_match'] for m in all_metrics]) * 100,
            'avg_predicted_count': np.mean([m['predicted_count'] for m in all_metrics]),
            'avg_expected_count': np.mean([m['expected_count'] for m in all_metrics]),
            'avg_true_positives': np.mean([m['true_positives'] for m in all_metrics]),
            'avg_false_positives': np.mean([m['false_positives'] for m in all_metrics]),
            'avg_false_negatives': np.mean([m['false_negatives'] for m in all_metrics])
        }

        # Prepare result summary
        summary = {
            'architecture': architecture,
            'dataset': 'reverse_queries.csv',
            'total_queries': len(results),
            'test_size': test_size if test_size else len(dataset),
            'metrics': avg_metrics,
            'elapsed_time_s': elapsed_time,
            'queries_per_sec': len(results) / elapsed_time,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Print results
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"Architecture:    {architecture}")
        logger.info(f"Total Queries:   {len(results)}")
        logger.info(f"-"*80)
        logger.info("AVERAGE METRICS:")
        logger.info(f"Precision:       {avg_metrics['precision']:.4f} ({avg_metrics['precision']*100:.2f}%)")
        logger.info(f"Recall:          {avg_metrics['recall']:.4f} ({avg_metrics['recall']*100:.2f}%)")
        logger.info(f"F1 Score:        {avg_metrics['f1_score']:.4f}")
        logger.info(f"Coverage:        {avg_metrics['coverage']:.2f}%")
        logger.info(f"Hallucination:   {avg_metrics['hallucination_rate']:.2f}%")
        logger.info(f"Exact Match:     {avg_metrics['exact_match_rate']:.2f}%")
        logger.info(f"-"*80)
        logger.info(f"Avg Predicted:   {avg_metrics['avg_predicted_count']:.1f} drugs/query")
        logger.info(f"Avg Expected:    {avg_metrics['avg_expected_count']:.1f} drugs/query")
        logger.info(f"Avg True Pos:    {avg_metrics['avg_true_positives']:.1f}")
        logger.info(f"Avg False Pos:   {avg_metrics['avg_false_positives']:.1f}")
        logger.info(f"Avg False Neg:   {avg_metrics['avg_false_negatives']:.1f}")
        logger.info(f"-"*80)
        logger.info(f"Time:            {elapsed_time:.2f}s")
        logger.info(f"Speed:           {len(results)/elapsed_time:.2f} queries/sec")
        logger.info("="*80)

        # Add detailed results to summary
        summary['detailed_results'] = results[:10]  # Include first 10 for inspection

        return summary

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj

        results = convert_numpy(results)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reverse queries with DrugRAG architectures"
    )
    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        help='Architecture to evaluate (e.g., format_a_qwen, graphrag_qwen, pure_llm_qwen, all)'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=None,
        help='Number of queries to test (omit for all queries in dataset)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: results_reverse_queries_{architecture}.json)'
    )

    args = parser.parse_args()

    evaluator = ReverseQueryEvaluator()

    # Define all available architectures
    all_architectures = [
        'pure_llm_qwen',
        'pure_llm_llama3',
        'format_a_qwen',
        'format_a_llama3',
        'format_b_qwen',
        'format_b_llama3',
        'graphrag_qwen',
        'graphrag_llama3',
    ]

    # Evaluate
    if args.architecture == 'all':
        logger.info(f"📊 Running evaluation for ALL {len(all_architectures)} architectures")
        all_results = {}

        for arch in all_architectures:
            try:
                result = evaluator.evaluate(arch, args.test_size)
                if result:
                    all_results[arch] = result
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {arch}: {e}")
                all_results[arch] = {'error': str(e)}

        # Save combined results
        output_file = args.output or f"results_reverse_queries_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        evaluator.save_results(all_results, output_file)

        # Print summary comparison
        logger.info("\n" + "="*80)
        logger.info("SUMMARY COMPARISON")
        logger.info("="*80)
        logger.info(f"{'Architecture':<30} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'Time (s)':<12}")
        logger.info("-"*80)
        for arch, result in all_results.items():
            if 'error' not in result and 'metrics' in result:
                logger.info(
                    f"{arch:<30} "
                    f"{result['metrics']['f1_score']:.4f}       "
                    f"{result['metrics']['precision']:.4f}       "
                    f"{result['metrics']['recall']:.4f}       "
                    f"{result['elapsed_time_s']:.2f}"
                )
        logger.info("="*80)
    else:
        # Single architecture
        result = evaluator.evaluate(args.architecture, args.test_size)

        if result:
            # Save results
            output_file = args.output or f"results_reverse_queries_{args.architecture}.json"
            evaluator.save_results(result, output_file)


if __name__ == "__main__":
    main()

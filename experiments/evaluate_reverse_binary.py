#!/usr/bin/env python3
"""
Reverse Binary Query Evaluation for DrugRAG

Evaluates the reverse_queries_binary.csv dataset using all available architectures.
This dataset contains reverse queries: "Which drugs cause [side_effect]?" with binary labels.

Usage:
    python evaluate_reverse_binary.py --architecture format_a_qwen --test_size 100
    python evaluate_reverse_binary.py --architecture all --test_size 500
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
from typing import List, Dict, Any
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vllm_model import VLLMQwenModel, VLLMLLAMA3Model
from src.evaluation.metrics import calculate_binary_classification_metrics, print_metrics_summary
from src.architectures.rag_format_a import FormatARAG
from src.architectures.rag_format_b import FormatBRAG
from src.architectures.graphrag import GraphRAG
from src.architectures.enhanced_rag_format_b import EnhancedRAGFormatB
from src.architectures.enhanced_graphrag import EnhancedGraphRAG
from src.architectures.advanced_rag_format_b import AdvancedRAGFormatB

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


class ReverseBinaryEvaluator:
    """Evaluator for reverse binary queries (drug-side_effect-YES/NO)"""

    def __init__(self, config_path: str = "../config.json"):
        self.config_path = config_path
        self.dataset_path = "../data/processed/reverse_queries_binary.csv"

    def load_dataset(self, test_size: int = None):
        """Load reverse binary dataset with format conversion"""
        logger.info(f"📂 Loading dataset: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)

        # Convert label from YES/NO to 1/0 for metrics
        logger.info(f"   Converting labels: YES→1, NO→0")
        df['label_numeric'] = df['label'].apply(lambda x: 1 if str(x).upper() == 'YES' else 0)

        # Get balanced sample if test_size is specified
        if test_size and test_size < len(df):
            logger.info(f"   Sampling {test_size} balanced examples...")
            positive_samples = df[df['label_numeric'] == 1].sample(
                n=min(test_size//2, len(df[df['label_numeric'] == 1])),
                random_state=42
            )
            negative_samples = df[df['label_numeric'] == 0].sample(
                n=min(test_size//2, len(df[df['label_numeric'] == 0])),
                random_state=42
            )
            balanced_df = pd.concat([positive_samples, negative_samples])
            df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"✅ Loaded {len(df)} examples")
        logger.info(f"   YES: {(df['label_numeric'] == 1).sum()}")
        logger.info(f"   NO: {(df['label_numeric'] == 0).sum()}")
        logger.info(f"   Unique side effects: {df['side_effect'].nunique()}")
        logger.info(f"   Unique drugs: {df['drug'].nunique()}")

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
        elif architecture == 'enhanced_format_b_qwen':
            return EnhancedRAGFormatB(self.config_path, model="qwen")
        elif architecture == 'enhanced_format_b_llama3':
            return EnhancedRAGFormatB(self.config_path, model="llama3")
        elif architecture == 'enhanced_graphrag_qwen':
            return EnhancedGraphRAG(self.config_path, model="qwen")
        elif architecture == 'enhanced_graphrag_llama3':
            return EnhancedGraphRAG(self.config_path, model="llama3")
        elif architecture == 'advanced_rag_b_qwen':
            return AdvancedRAGFormatB(self.config_path, model="qwen")
        elif architecture == 'advanced_rag_b_llama3':
            return AdvancedRAGFormatB(self.config_path, model="llama3")
        elif architecture == 'pure_llm_qwen':
            return VLLMQwenModel(self.config_path)
        elif architecture == 'pure_llm_llama3':
            return VLLMLLAMA3Model(self.config_path)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def evaluate(self, architecture: str, test_size: int = None):
        """Run evaluation for a single architecture"""
        logger.info("="*80)
        logger.info(f"REVERSE BINARY EVALUATION")
        logger.info(f"Architecture: {architecture}")
        logger.info(f"Dataset: reverse_queries_binary.csv")
        logger.info("="*80)

        # Load dataset
        dataset = self.load_dataset(test_size)

        # Initialize architecture
        arch = self.initialize_architecture(architecture)

        # Start timing
        start_time = time.time()

        # Prepare queries
        queries = []
        for _, row in dataset.iterrows():
            queries.append({
                'drug': row['drug'],
                'side_effect': row['side_effect'],
                'query': row['query'],
                'label': row['label_numeric']
            })

        logger.info(f"\n🚀 Processing {len(queries)} queries...")

        # Use batch processing if available
        if hasattr(arch, 'query_batch'):
            logger.info("   ✅ Using batch processing")
            batch_start = time.time()
            results = arch.query_batch(queries)
            batch_time = time.time() - batch_start
            logger.info(f"   ⚡ Completed in {batch_time:.2f}s ({len(queries)/batch_time:.1f} queries/sec)")
        else:
            logger.info("   ⚠️  Using individual queries")
            results = []
            for q in tqdm(queries, desc="🔍 Processing", unit="query"):
                result = arch.query(q['drug'], q['side_effect'])
                results.append(result)

        # Calculate metrics
        elapsed_time = time.time() - start_time

        y_true = []
        y_pred = []
        correct = 0
        unknown_count = 0

        # Detailed logging of prompt-answer pairs
        detailed_logs = []

        for query, result in zip(queries, results):
            true_label = 'YES' if query['label'] == 1 else 'NO'
            predicted = result.get('answer', 'UNKNOWN')

            if predicted == 'UNKNOWN':
                unknown_count += 1
                # Treat UNKNOWN as incorrect (NO)
                predicted = 'NO'

            y_true.append(true_label)
            y_pred.append(predicted)

            if predicted == true_label:
                correct += 1

            # Log detailed prompt-answer pair
            detailed_log_entry = {
                'query_id': len(detailed_logs),
                'drug': query['drug'],
                'side_effect': query['side_effect'],
                'query': query['query'],
                'true_label': true_label,
                'predicted_label': predicted,
                'correct': predicted == true_label,
                'prompt': result.get('prompt', 'N/A'),
                'full_response': result.get('full_response', result.get('reasoning', 'N/A')),
                'confidence': result.get('confidence', 0.0),
                'evidence_count': result.get('evidence_count', 0),
                'model': result.get('model', 'unknown'),
                'architecture': architecture
            }
            detailed_logs.append(detailed_log_entry)

        # Calculate comprehensive metrics
        metrics = calculate_binary_classification_metrics(y_true, y_pred)

        # Prepare result summary
        summary = {
            'architecture': architecture,
            'dataset': 'reverse_queries_binary.csv',
            'total_queries': len(queries),
            'test_size': test_size if test_size else len(queries),
            'correct': correct,
            'unknown_count': unknown_count,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['sensitivity'],  # sensitivity = recall
            'specificity': metrics['specificity'],
            'f1_score': metrics['f1_score'],
            'true_positives': metrics['tp'],
            'true_negatives': metrics['tn'],
            'false_positives': metrics['fp'],
            'false_negatives': metrics['fn'],
            'elapsed_time_s': elapsed_time,
            'queries_per_sec': len(queries) / elapsed_time,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detailed_logs': detailed_logs  # Include detailed prompt-answer pairs
        }

        # Print results
        logger.info("\n" + "="*80)
        logger.info("EVALUATION RESULTS")
        logger.info("="*80)
        logger.info(f"Architecture:    {architecture}")
        logger.info(f"Total Queries:   {len(queries)}")
        logger.info(f"Correct:         {correct} ({correct/len(queries)*100:.1f}%)")
        logger.info(f"Unknown:         {unknown_count} ({unknown_count/len(queries)*100:.1f}%)")
        logger.info(f"-"*80)
        logger.info(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"Precision:       {metrics['precision']:.4f}")
        logger.info(f"Recall:          {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity:     {metrics['specificity']:.4f}")
        logger.info(f"F1 Score:        {metrics['f1_score']:.4f}")
        logger.info(f"-"*80)
        logger.info(f"True Positives:  {metrics['tp']}")
        logger.info(f"True Negatives:  {metrics['tn']}")
        logger.info(f"False Positives: {metrics['fp']}")
        logger.info(f"False Negatives: {metrics['fn']}")
        logger.info(f"-"*80)
        logger.info(f"Time:            {elapsed_time:.2f}s")
        logger.info(f"Speed:           {len(queries)/elapsed_time:.1f} queries/sec")
        logger.info("="*80)

        # Log information about detailed logging
        logger.info(f"\n📋 Detailed prompt-answer pairs saved ({len(detailed_logs)} entries)")
        logger.info(f"   Each entry includes: query, prompt, full_response, labels, confidence")

        return summary

    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✅ Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reverse binary queries with DrugRAG architectures"
    )
    parser.add_argument(
        '--architecture',
        type=str,
        required=True,
        help='Architecture to evaluate (e.g., format_a_qwen, graphrag_llama3, all)'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=None,
        help='Number of queries to test (omit for all 1,200 queries)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: results_reverse_binary_{architecture}.json)'
    )

    args = parser.parse_args()

    evaluator = ReverseBinaryEvaluator()

    # Define all available architectures
    all_architectures = [
        'pure_llm_qwen', 'pure_llm_llama3',
        'format_a_qwen', 'format_a_llama3',
        'format_b_qwen', 'format_b_llama3',
        'graphrag_qwen', 'graphrag_llama3',
        'enhanced_format_b_qwen', 'enhanced_format_b_llama3',
        'enhanced_graphrag_qwen', 'enhanced_graphrag_llama3',
    ]

    # Evaluate
    if args.architecture == 'all':
        logger.info(f"📊 Running evaluation for ALL {len(all_architectures)} architectures")
        all_results = {}

        for arch in all_architectures:
            try:
                result = evaluator.evaluate(arch, args.test_size)
                all_results[arch] = result
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {arch}: {e}")
                all_results[arch] = {'error': str(e)}

        # Save combined results
        output_file = args.output or f"results_reverse_binary_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        evaluator.save_results(all_results, output_file)

        # Print summary comparison
        logger.info("\n" + "="*80)
        logger.info("SUMMARY COMPARISON")
        logger.info("="*80)
        logger.info(f"{'Architecture':<30} {'F1 Score':<12} {'Accuracy':<12} {'Time (s)':<12}")
        logger.info("-"*80)
        for arch, result in all_results.items():
            if 'error' not in result:
                logger.info(
                    f"{arch:<30} "
                    f"{result['f1_score']:.4f}       "
                    f"{result['accuracy']:.4f}       "
                    f"{result['elapsed_time_s']:.2f}"
                )
        logger.info("="*80)
    else:
        # Single architecture
        result = evaluator.evaluate(args.architecture, args.test_size)

        # Save results
        output_file = args.output or f"results_reverse_binary_{args.architecture}.json"
        evaluator.save_results(result, output_file)


if __name__ == "__main__":
    main()

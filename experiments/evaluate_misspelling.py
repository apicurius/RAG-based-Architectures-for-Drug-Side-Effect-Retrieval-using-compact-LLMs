#!/usr/bin/env python3
"""
Misspelling Robustness Experiment

Tests how Pure LLM, RAG Format A, RAG Format B, and GraphRAG handle
misspelled drug names. Demonstrates semantic understanding superiority
over naive string matching.

Usage:
    python evaluate_misspelling.py --architectures all --models both
    python evaluate_misspelling.py --architectures pure_llm format_b --models qwen
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
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vllm_model import VLLMQwenModel, VLLMLLAMA3Model
from src.evaluation.metrics import calculate_binary_classification_metrics
from src.architectures.rag_format_a import FormatARAG
from src.architectures.rag_format_b import FormatBRAG
from src.architectures.graphrag import GraphRAG
from src.utils.misspelling_dataset_generator import MisspellingDatasetGenerator

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


class MisspellingEvaluator:
    """Evaluates robustness of different architectures to drug name misspellings."""

    def __init__(self, config_path: str = "/home/omeerdogan23/drugRAG/experiments/config.json", results_dir: str = "/home/omeerdogan23/drugRAG/results/misspelling_experiment"):
        """
        Initialize evaluator.

        Args:
            config_path: Path to configuration file
            results_dir: Directory to save results
        """
        self.config_path = config_path
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        logger.info(f"Results will be saved to: {results_dir}")

    def generate_datasets(self) -> Tuple[str, str]:
        """
        Generate or load correct and misspelled datasets.

        Returns:
            Tuple of (correct_path, misspelled_path)
        """
        correct_path = "/home/omeerdogan23/drugRAG/data/processed/misspelling_experiment_correct.csv"
        misspelled_path = "/home/omeerdogan23/drugRAG/data/processed/misspelling_experiment_misspelled.csv"

        # Check if datasets already exist
        if os.path.exists(correct_path) and os.path.exists(misspelled_path):
            logger.info("✓ Using existing misspelling datasets")
            return correct_path, misspelled_path

        # Generate datasets
        logger.info("Generating misspelling datasets...")
        generator = MisspellingDatasetGenerator()
        generator.generate_and_save()

        return correct_path, misspelled_path

    def load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from CSV."""
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded {len(df)} queries from {dataset_path}")
        return df

    def initialize_architecture(self, architecture: str, model: str):
        """
        Initialize architecture with specified model.

        Args:
            architecture: One of ['pure_llm', 'format_a', 'format_b', 'graphrag']
            model: One of ['qwen', 'llama3']

        Returns:
            Initialized architecture instance
        """
        arch_key = f"{architecture}_{model}"

        logger.info(f"Initializing {arch_key}...")

        if architecture == 'pure_llm':
            if model == 'qwen':
                return VLLMQwenModel(self.config_path)
            else:
                return VLLMLLAMA3Model(self.config_path)
        elif architecture == 'format_a':
            return FormatARAG(self.config_path, model=model)
        elif architecture == 'format_b':
            return FormatBRAG(self.config_path, model=model)
        elif architecture == 'graphrag':
            return GraphRAG(self.config_path, model=model)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def evaluate_dataset(self, arch_instance, dataset: pd.DataFrame, condition: str) -> Dict[str, Any]:
        """
        Evaluate architecture on a dataset.

        Args:
            arch_instance: Initialized architecture
            dataset: DataFrame with queries
            condition: 'correct' or 'misspelled'

        Returns:
            Dictionary with metrics and predictions
        """
        logger.info(f"Evaluating {condition} dataset ({len(dataset)} queries)...")

        # Prepare batch queries
        queries = []
        for _, row in dataset.iterrows():
            queries.append({
                'drug': row['drug'],
                'side_effect': row['side_effect'],
                'label': row.get('label', None)
            })

        # Run batch evaluation if available, otherwise individual
        start_time = time.time()

        if hasattr(arch_instance, 'query_batch'):
            logger.info("   Using batch processing...")
            results = arch_instance.query_batch(queries)
        else:
            logger.info("   Using individual query processing...")
            results = []
            for q in tqdm(queries, desc=f"Processing {condition}", unit="query"):
                result = arch_instance.query(q['drug'], q['side_effect'])
                results.append(result)

        elapsed_time = time.time() - start_time
        queries_per_sec = len(queries) / elapsed_time if elapsed_time > 0 else 0

        logger.info(f"   Completed in {elapsed_time:.2f}s ({queries_per_sec:.1f} queries/sec)")

        # Calculate metrics
        y_true = []
        y_pred = []
        detailed_results = []

        for i, (query, result) in enumerate(zip(queries, results)):
            if query['label'] is not None:
                true_answer = 'YES' if query['label'] == 1 else 'NO'
                predicted = result.get('answer', 'UNKNOWN')

                y_true.append(true_answer)
                y_pred.append(predicted)

                detailed_results.append({
                    'drug': query['drug'],
                    'side_effect': query['side_effect'],
                    'ground_truth': true_answer,
                    'predicted': predicted,
                    'is_correct': predicted == true_answer,
                    'confidence': result.get('confidence', 0.0)
                })

        # Calculate comprehensive metrics
        metrics = calculate_binary_classification_metrics(y_true, y_pred)

        return {
            'metrics': metrics,
            'detailed_results': detailed_results,
            'elapsed_time': elapsed_time,
            'queries_per_sec': queries_per_sec
        }

    def calculate_degradation(self, correct_metrics: Dict, misspelled_metrics: Dict) -> Dict:
        """
        Calculate degradation from correct to misspelled.

        Args:
            correct_metrics: Metrics from correct dataset
            misspelled_metrics: Metrics from misspelled dataset

        Returns:
            Dictionary with degradation metrics
        """
        degradation = {}

        for metric in ['accuracy', 'f1_score', 'precision', 'sensitivity', 'specificity']:
            correct_val = correct_metrics.get(metric, 0.0)
            misspelled_val = misspelled_metrics.get(metric, 0.0)

            # Absolute degradation
            abs_degradation = correct_val - misspelled_val

            # Percentage degradation
            pct_degradation = (abs_degradation / correct_val * 100) if correct_val > 0 else 0

            # Robustness score (higher is better)
            robustness = (misspelled_val / correct_val) if correct_val > 0 else 0

            degradation[metric] = {
                'correct': correct_val,
                'misspelled': misspelled_val,
                'absolute_degradation': abs_degradation,
                'percentage_degradation': pct_degradation,
                'robustness_score': robustness
            }

        return degradation

    def run_experiment(self, architectures: List[str], models: List[str]):
        """
        Run complete misspelling experiment.

        Args:
            architectures: List of architectures to test
            models: List of models to test
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("="*80)
        logger.info("MISSPELLING ROBUSTNESS EXPERIMENT")
        logger.info("="*80)
        logger.info(f"Architectures: {', '.join(architectures)}")
        logger.info(f"Models: {', '.join(models)}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info("="*80)

        # Generate datasets
        correct_path, misspelled_path = self.generate_datasets()
        correct_df = self.load_dataset(correct_path)
        misspelled_df = self.load_dataset(misspelled_path)

        # Store all results
        all_results = []
        comparison_data = []

        # Run evaluation for each architecture and model
        for architecture in architectures:
            for model in models:
                arch_key = f"{architecture}_{model}"

                logger.info("\n" + "="*80)
                logger.info(f"EVALUATING: {arch_key.upper()}")
                logger.info("="*80)

                try:
                    # Initialize architecture
                    arch_instance = self.initialize_architecture(architecture, model)

                    # Evaluate on correct dataset
                    logger.info(f"\n[{arch_key}] Testing with CORRECT spellings...")
                    correct_eval = self.evaluate_dataset(arch_instance, correct_df, "correct")

                    # Evaluate on misspelled dataset
                    logger.info(f"\n[{arch_key}] Testing with MISSPELLED drug names...")
                    misspelled_eval = self.evaluate_dataset(arch_instance, misspelled_df, "misspelled")

                    # Calculate degradation
                    degradation = self.calculate_degradation(
                        correct_eval['metrics'],
                        misspelled_eval['metrics']
                    )

                    # Store results
                    result = {
                        'architecture': architecture,
                        'model': model,
                        'arch_key': arch_key,
                        'correct_metrics': correct_eval['metrics'],
                        'misspelled_metrics': misspelled_eval['metrics'],
                        'degradation': degradation,
                        'correct_time': correct_eval['elapsed_time'],
                        'misspelled_time': misspelled_eval['elapsed_time']
                    }

                    all_results.append(result)

                    # Print summary
                    self.print_comparison_summary(arch_key, degradation)

                    # Prepare comparison data for CSV
                    for metric in ['accuracy', 'f1_score', 'precision', 'sensitivity', 'specificity']:
                        comparison_data.append({
                            'architecture': architecture,
                            'model': model,
                            'arch_key': arch_key,
                            'metric': metric,
                            'correct_value': degradation[metric]['correct'],
                            'misspelled_value': degradation[metric]['misspelled'],
                            'absolute_degradation': degradation[metric]['absolute_degradation'],
                            'percentage_degradation': degradation[metric]['percentage_degradation'],
                            'robustness_score': degradation[metric]['robustness_score']
                        })

                except Exception as e:
                    logger.error(f"Error evaluating {arch_key}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Save results
        self.save_results(all_results, comparison_data, timestamp)

        # Print final summary
        self.print_final_summary(all_results)

    def print_comparison_summary(self, arch_key: str, degradation: Dict):
        """Print comparison summary for one architecture."""
        print("\n" + "-"*80)
        print(f"SUMMARY: {arch_key.upper()}")
        print("-"*80)
        print(f"{'Metric':<15} | {'Correct':<10} | {'Misspelled':<10} | {'Degradation':<12} | {'Robustness':<10}")
        print("-"*80)

        for metric in ['accuracy', 'f1_score', 'precision', 'sensitivity']:
            deg = degradation[metric]
            print(f"{metric:<15} | {deg['correct']:<10.4f} | {deg['misspelled']:<10.4f} | "
                  f"{deg['percentage_degradation']:>10.2f}% | {deg['robustness_score']:<10.4f}")

        print("-"*80)

    def print_final_summary(self, all_results: List[Dict]):
        """Print final summary comparing all architectures."""
        print("\n" + "="*80)
        print("FINAL SUMMARY - F1 SCORE COMPARISON")
        print("="*80)
        print(f"{'Architecture':<30} | {'Correct F1':<12} | {'Misspelled F1':<12} | {'Degradation':<12} | {'Robustness':<10}")
        print("="*80)

        for result in sorted(all_results, key=lambda x: x['degradation']['f1_score']['percentage_degradation']):
            arch_key = result['arch_key']
            deg = result['degradation']['f1_score']
            print(f"{arch_key:<30} | {deg['correct']:<12.4f} | {deg['misspelled']:<12.4f} | "
                  f"{deg['percentage_degradation']:>10.2f}% | {deg['robustness_score']:<10.4f}")

        print("="*80)
        print("\nKEY INSIGHTS:")
        print("- Lower degradation % = More robust to misspellings")
        print("- Higher robustness score = Better semantic understanding")
        print("- Pure LLM expected to show lowest degradation (semantic understanding)")
        print("- GraphRAG expected to show highest degradation (exact string matching)")
        print("="*80)

    def save_results(self, all_results: List[Dict], comparison_data: List[Dict], timestamp: str):
        """Save all results to files."""
        # Save detailed results as JSON
        results_json_path = f"{self.results_dir}/detailed_results_{timestamp}.json"
        with open(results_json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"\nDetailed results saved to: {results_json_path}")

        # Save comparison as CSV
        comparison_csv_path = f"{self.results_dir}/comparison_{timestamp}.csv"
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(comparison_csv_path, index=False)
        logger.info(f"Comparison CSV saved to: {comparison_csv_path}")

        # Save summary report
        summary_path = f"{self.results_dir}/summary_report_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MISSPELLING ROBUSTNESS EXPERIMENT - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")

            for result in all_results:
                f.write(f"\n{result['arch_key'].upper()}\n")
                f.write("-"*80 + "\n")
                for metric in ['accuracy', 'f1_score', 'precision', 'sensitivity']:
                    deg = result['degradation'][metric]
                    f.write(f"{metric}: {deg['correct']:.4f} -> {deg['misspelled']:.4f} "
                           f"(degradation: {deg['percentage_degradation']:.2f}%)\n")

        logger.info(f"Summary report saved to: {summary_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Misspelling Robustness Experiment")

    parser.add_argument(
        '--architectures',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'pure_llm', 'format_a', 'format_b', 'graphrag'],
        help="Architectures to test (default: all)"
    )

    parser.add_argument(
        '--models',
        type=str,
        default='both',
        choices=['both', 'qwen', 'llama3'],
        help="Models to test (default: both)"
    )

    parser.add_argument(
        '--config',
        type=str,
        default='/home/omeerdogan23/drugRAG/experiments/config.json',
        help="Path to config file"
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        default='/home/omeerdogan23/drugRAG/results/misspelling_experiment',
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Parse architectures
    if 'all' in args.architectures:
        architectures = ['pure_llm', 'format_a', 'format_b', 'graphrag']
    else:
        architectures = args.architectures

    # Parse models
    models = ['qwen', 'llama3'] if args.models == 'both' else [args.models]

    # Run experiment
    evaluator = MisspellingEvaluator(
        config_path=args.config,
        results_dir=args.results_dir
    )

    evaluator.run_experiment(architectures, models)


if __name__ == "__main__":
    main()

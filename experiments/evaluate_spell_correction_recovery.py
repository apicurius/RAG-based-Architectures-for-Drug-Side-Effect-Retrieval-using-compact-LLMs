#!/usr/bin/env python3
"""
Part 2: Architecture Recovery with Spell Correction (Qwen 7B)

Tests if Qwen 7B spell correction can rescue Format B RAG and GraphRAG
from their 100% degradation with misspelled drug names.

Three-way comparison:
1. Perfect spelling (baseline performance)
2. Raw misspelled (expect 100% failure)
3. LLM-corrected (test recovery)

Usage:
    python evaluate_spell_correction_recovery.py --architectures format_b graphrag
    python evaluate_spell_correction_recovery.py --architectures format_b
"""

import argparse
import json
import logging
import pandas as pd
import os
import sys
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.spell_corrector import LLMSpellCorrector
from src.evaluation.metrics import calculate_binary_classification_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArchitectureRecoveryEvaluator:
    """Test if spell correction rescues Format B and GraphRAG"""

    def __init__(self, config_path: str = "config.json", results_dir: str = None):
        """
        Initialize evaluator

        Args:
            config_path: Path to configuration file
            results_dir: Directory to save results
        """
        self.config_path = config_path

        if results_dir is None:
            self.results_dir = "/home/omeerdogan23/drugRAG/results/spell_correction_recovery"
        else:
            self.results_dir = results_dir

        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize Qwen spell corrector
        self.corrector = LLMSpellCorrector(
            use_fewshot=True,
            temperature=0.0,
            config_path=config_path
        )

        logger.info(f"Results will be saved to: {self.results_dir}")

    def generate_corrected_dataset(self, misspelled_csv: str) -> str:
        """
        Create a new dataset with LLM-corrected drug names

        Args:
            misspelled_csv: Path to misspelled dataset

        Returns:
            Path to generated LLM-corrected dataset
        """
        logger.info("=" * 80)
        logger.info("GENERATING LLM-CORRECTED DATASET")
        logger.info("=" * 80)

        df = pd.read_csv(misspelled_csv)
        logger.info(f"Loaded {len(df)} queries from misspelled dataset")

        # Extract unique misspelled drug names
        unique_drugs = df['drug'].unique()
        logger.info(f"Found {len(unique_drugs)} unique misspelled drug names")

        # Batch correct them with Qwen
        logger.info("Running Qwen 7B spell correction...")
        correction_results = self.corrector.correct_batch(list(unique_drugs))

        # Create mapping: misspelled → corrected
        correction_map = {
            result.original: result.corrected
            for result in correction_results
        }

        # Log corrections
        logger.info("\nDrug name corrections:")
        logger.info("-" * 60)
        for misspelled, corrected in correction_map.items():
            changed = " ✓" if misspelled != corrected else ""
            logger.info(f"  {misspelled:<20} → {corrected:<20}{changed}")
        logger.info("-" * 60)

        # Apply corrections to dataset
        df_corrected = df.copy()
        df_corrected['drug_original_misspelled'] = df_corrected['drug']
        df_corrected['drug'] = df_corrected['drug'].map(correction_map)

        # Update query text too
        for misspelled, corrected in correction_map.items():
            mask = df_corrected['drug_original_misspelled'] == misspelled
            df_corrected.loc[mask, 'query'] = df_corrected.loc[mask, 'query'].str.replace(
                misspelled,
                corrected,
                case=False,
                regex=False
            )

        # Save
        output_path = "/home/omeerdogan23/drugRAG/data/processed/misspelling_experiment_llm_corrected.csv"
        df_corrected.to_csv(output_path, index=False)

        logger.info(f"\n✅ Saved LLM-corrected dataset to: {output_path}")
        logger.info(f"Total queries: {len(df_corrected)}")

        return output_path

    def initialize_architecture(self, architecture: str, model: str = 'qwen'):
        """
        Initialize architecture instance

        Args:
            architecture: 'pure_llm', 'format_a', 'format_b' or 'graphrag'
            model: 'qwen' (default)

        Returns:
            Architecture instance
        """
        logger.info(f"Initializing {architecture} with {model}...")

        if architecture == 'pure_llm':
            from src.models.vllm_model import VLLMQwenModel, VLLMLLAMA3Model
            # Create a wrapper that matches the query interface
            class PureLLMWrapper:
                def __init__(self, config_path, model):
                    if model == 'qwen':
                        self.llm = VLLMQwenModel(config_path)
                    else:
                        self.llm = VLLMLLAMA3Model(config_path)
                    self.model = model

                def query(self, drug: str, side_effect: str):
                    prompt = f"""You are asked to answer the following question with a single word: YES or NO.

Question: Is {side_effect} an adverse effect of {drug}?

Answer with only YES or NO:"""
                    response = self.llm.generate_response(prompt, max_tokens=10, temperature=0.0)
                    answer = 'YES' if 'YES' in response.upper() else 'NO' if 'NO' in response.upper() else 'UNKNOWN'
                    return {'answer': answer, 'drug': drug, 'side_effect': side_effect}

                def query_batch(self, queries):
                    prompts = []
                    for q in queries:
                        prompt = f"""You are asked to answer the following question with a single word: YES or NO.

Question: Is {q['side_effect']} an adverse effect of {q['drug']}?

Answer with only YES or NO:"""
                        prompts.append(prompt)

                    responses = self.llm.generate_batch(prompts, max_tokens=10, temperature=0.0)
                    results = []
                    for q, response in zip(queries, responses):
                        answer = 'YES' if 'YES' in response.upper() else 'NO' if 'NO' in response.upper() else 'UNKNOWN'
                        results.append({'answer': answer, 'drug': q['drug'], 'side_effect': q['side_effect']})
                    return results

            return PureLLMWrapper(self.config_path, model)
        elif architecture == 'format_a':
            from src.architectures.rag_format_a import FormatARAG
            return FormatARAG(self.config_path, model=model)
        elif architecture == 'format_b':
            from src.architectures.rag_format_b import FormatBRAG
            return FormatBRAG(self.config_path, model=model)
        elif architecture == 'graphrag':
            from src.architectures.graphrag import GraphRAG
            return GraphRAG(self.config_path, model=model)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def evaluate_dataset(self, arch_instance, dataset: pd.DataFrame, condition: str) -> Dict:
        """
        Evaluate architecture on a dataset

        Args:
            arch_instance: Initialized architecture
            dataset: DataFrame with queries
            condition: 'perfect', 'raw_misspelled', or 'llm_corrected'

        Returns:
            Dictionary with metrics
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

        # Run batch evaluation
        if hasattr(arch_instance, 'query_batch'):
            results = arch_instance.query_batch(queries)
        else:
            # Fallback to individual queries
            results = []
            for q in queries:
                result = arch_instance.query(q['drug'], q['side_effect'])
                results.append(result)

        # Calculate metrics
        y_true = []
        y_pred = []

        for query, result in zip(queries, results):
            if query['label'] is not None:
                true_answer = 'YES' if query['label'] == 1 else 'NO'
                predicted = result.get('answer', 'UNKNOWN')

                y_true.append(true_answer)
                y_pred.append(predicted)

        # Calculate comprehensive metrics
        metrics = calculate_binary_classification_metrics(y_true, y_pred)

        logger.info(f"   {condition} - F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}")

        return {'metrics': metrics}

    def calculate_recovery_metrics(self, results: Dict) -> Dict:
        """
        Calculate how much of the degradation was recovered

        Recovery % = (LLM_corrected_F1 - Raw_F1) / (Perfect_F1 - Raw_F1) × 100%

        Args:
            results: Dictionary with metrics for perfect, raw, and corrected conditions

        Returns:
            Recovery analysis dictionary
        """
        perfect_f1 = results['perfect']['metrics']['f1_score']
        raw_f1 = results['raw_misspelled']['metrics']['f1_score']
        corrected_f1 = results['llm_corrected']['metrics']['f1_score']

        if perfect_f1 == raw_f1:  # Avoid division by zero
            recovery_percentage = 100.0 if corrected_f1 == perfect_f1 else 0.0
        else:
            recovery_percentage = (corrected_f1 - raw_f1) / (perfect_f1 - raw_f1) * 100

        return {
            'perfect_f1': perfect_f1,
            'raw_misspelled_f1': raw_f1,
            'llm_corrected_f1': corrected_f1,
            'raw_degradation': perfect_f1 - raw_f1,
            'remaining_degradation': perfect_f1 - corrected_f1,
            'recovery_percentage': recovery_percentage,
            'full_recovery': recovery_percentage >= 95.0,
            'partial_recovery': 50.0 <= recovery_percentage < 95.0,
            'no_recovery': recovery_percentage < 50.0
        }

    def evaluate_three_way_comparison(self, architecture: str, model: str = 'qwen') -> Dict:
        """
        Three-way comparison for one architecture

        Args:
            architecture: 'pure_llm', 'format_a', 'format_b' or 'graphrag'
            model: 'qwen'

        Returns:
            Complete results dictionary
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"THREE-WAY COMPARISON: {architecture.upper()}")
        logger.info("=" * 80)

        # Load datasets
        correct_path = "/home/omeerdogan23/drugRAG/data/processed/misspelling_experiment_correct.csv"
        misspelled_path = "/home/omeerdogan23/drugRAG/data/processed/misspelling_experiment_misspelled.csv"

        # Generate corrected dataset
        corrected_path = self.generate_corrected_dataset(misspelled_path)

        # Initialize architecture
        arch_instance = self.initialize_architecture(architecture, model)

        # Evaluate on all three conditions
        results = {}

        # 1. Perfect spelling (ceiling)
        logger.info(f"\n[{architecture}] Condition 1: PERFECT SPELLINGS...")
        results['perfect'] = self.evaluate_dataset(
            arch_instance,
            pd.read_csv(correct_path),
            "perfect"
        )

        # 2. Raw misspelled (expect failure)
        logger.info(f"\n[{architecture}] Condition 2: RAW MISSPELLINGS...")
        results['raw_misspelled'] = self.evaluate_dataset(
            arch_instance,
            pd.read_csv(misspelled_path),
            "raw_misspelled"
        )

        # 3. LLM-corrected (test recovery)
        logger.info(f"\n[{architecture}] Condition 3: QWEN-CORRECTED...")
        results['llm_corrected'] = self.evaluate_dataset(
            arch_instance,
            pd.read_csv(corrected_path),
            "llm_corrected"
        )

        # Calculate recovery metrics
        recovery_analysis = self.calculate_recovery_metrics(results)

        # Print summary
        self.print_recovery_summary(architecture, recovery_analysis)

        return {
            'architecture': architecture,
            'model': model,
            'results': results,
            'recovery_analysis': recovery_analysis
        }

    def print_recovery_summary(self, architecture: str, recovery: Dict):
        """Print recovery summary"""
        logger.info("\n" + "-" * 80)
        logger.info(f"RECOVERY SUMMARY: {architecture.upper()}")
        logger.info("-" * 80)
        logger.info(f"Perfect F1:        {recovery['perfect_f1']:.4f}")
        logger.info(f"Raw Misspelled F1: {recovery['raw_misspelled_f1']:.4f} (degradation: {recovery['raw_degradation']:.4f})")
        logger.info(f"LLM-Corrected F1:  {recovery['llm_corrected_f1']:.4f}")
        logger.info(f"Recovery:          {recovery['recovery_percentage']:.1f}%")

        if recovery['full_recovery']:
            logger.info("Status: ✓ FULL RECOVERY (≥95%)")
        elif recovery['partial_recovery']:
            logger.info("Status: ⚠ PARTIAL RECOVERY (50-95%)")
        else:
            logger.info("Status: ✗ NO RECOVERY (<50%)")

        logger.info("-" * 80)

    def save_results(self, all_results: List[Dict], timestamp: str):
        """Save all results to files"""
        # Save detailed JSON
        results_json_path = f"{self.results_dir}/recovery_results_{timestamp}.json"
        with open(results_json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"\nDetailed results saved to: {results_json_path}")

        # Save comparison CSV
        comparison_data = []
        for result in all_results:
            arch = result['architecture']
            rec = result['recovery_analysis']

            comparison_data.append({
                'architecture': arch,
                'model': result['model'],
                'perfect_f1': rec['perfect_f1'],
                'raw_misspelled_f1': rec['raw_misspelled_f1'],
                'llm_corrected_f1': rec['llm_corrected_f1'],
                'raw_degradation': rec['raw_degradation'],
                'remaining_degradation': rec['remaining_degradation'],
                'recovery_percentage': rec['recovery_percentage'],
                'status': 'Full' if rec['full_recovery'] else 'Partial' if rec['partial_recovery'] else 'None'
            })

        comparison_csv_path = f"{self.results_dir}/three_way_comparison_{timestamp}.csv"
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison.to_csv(comparison_csv_path, index=False)

        logger.info(f"Comparison CSV saved to: {comparison_csv_path}")

        # Save summary report
        summary_path = f"{self.results_dir}/recovery_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SPELL CORRECTION RECOVERY EXPERIMENT - SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write("Spell Corrector: Qwen2.5-7B-Instruct (vLLM)\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            for result in all_results:
                arch = result['architecture']
                rec = result['recovery_analysis']

                f.write(f"\n{arch.upper()}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Perfect F1:        {rec['perfect_f1']:.4f}\n")
                f.write(f"Raw Misspelled F1: {rec['raw_misspelled_f1']:.4f}\n")
                f.write(f"LLM-Corrected F1:  {rec['llm_corrected_f1']:.4f}\n")
                f.write(f"Recovery:          {rec['recovery_percentage']:.1f}%\n")
                f.write(f"Status: {'Full' if rec['full_recovery'] else 'Partial' if rec['partial_recovery'] else 'None'}\n")

        logger.info(f"Summary report saved to: {summary_path}")

    def run_full_evaluation(self, architectures: List[str]):
        """Run complete recovery evaluation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("\n" + "=" * 80)
        logger.info("ARCHITECTURE RECOVERY EVALUATION - STARTING")
        logger.info("=" * 80)
        logger.info(f"Architectures: {', '.join(architectures)}")
        logger.info(f"Spell Corrector: Qwen2.5-7B-Instruct")
        logger.info(f"Timestamp: {timestamp}")
        logger.info("=" * 80)

        # Run evaluation for each architecture
        all_results = []

        for architecture in architectures:
            try:
                result = self.evaluate_three_way_comparison(architecture, model='qwen')
                all_results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating {architecture}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Save results
        if all_results:
            self.save_results(all_results, timestamp)

        logger.info("\n" + "=" * 80)
        logger.info("ARCHITECTURE RECOVERY EVALUATION - COMPLETE")
        logger.info("=" * 80)

        return all_results


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Test architecture recovery with Qwen spell correction")

    parser.add_argument(
        '--architectures',
        type=str,
        nargs='+',
        default=['pure_llm', 'format_a', 'format_b', 'graphrag'],
        choices=['pure_llm', 'format_a', 'format_b', 'graphrag'],
        help="Architectures to test (default: all four)"
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
        default='/home/omeerdogan23/drugRAG/results/spell_correction_recovery',
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = ArchitectureRecoveryEvaluator(
        config_path=args.config,
        results_dir=args.results_dir
    )

    evaluator.run_full_evaluation(architectures=args.architectures)


if __name__ == "__main__":
    main()

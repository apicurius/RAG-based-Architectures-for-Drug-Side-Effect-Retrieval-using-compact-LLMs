#!/usr/bin/env python3
"""
Part 1: Spell Correction Accuracy Evaluation (Qwen 7B)

Tests Qwen 7B's ability to correct misspelled drug names.

Metrics:
- Exact match accuracy: corrected == ground_truth
- Average edit distance from ground truth
- Unchanged rate: % of misspellings not corrected
- Over-correction rate: % of correct names wrongly changed

Usage:
    python evaluate_spell_correction.py
    python evaluate_spell_correction.py --no-fewshot
    python evaluate_spell_correction.py --temperature 0.3
"""

import argparse
import json
import logging
import pandas as pd
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.spell_corrector import LLMSpellCorrector, levenshtein_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SpellCorrectionEvaluator:
    """Evaluate Qwen 7B spell correction accuracy on drug names"""

    def __init__(self, config_path: str = "config.json", results_dir: str = None):
        """
        Initialize evaluator

        Args:
            config_path: Path to configuration file
            results_dir: Directory to save results
        """
        self.config_path = config_path

        if results_dir is None:
            self.results_dir = "/home/omeerdogan23/drugRAG/results/spell_correction_experiment"
        else:
            self.results_dir = results_dir

        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {self.results_dir}")

    def load_misspellings(self) -> List[Tuple[str, str]]:
        """
        Load (correct, misspelled) pairs from misspellings.csv

        Returns:
            List of (correct_drug, misspelled_drug) tuples
        """
        csv_path = "/home/omeerdogan23/drugRAG/experiments/misspellings.csv"

        logger.info(f"Loading misspellings from: {csv_path}")

        df = pd.read_csv(csv_path)

        # Remove BOM if present and strip whitespace
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()

        pairs = []
        for _, row in df.iterrows():
            correct = str(row['Original']).strip().lower()
            misspelled = str(row['Spelling error']).strip().lower()
            pairs.append((correct, misspelled))

        logger.info(f"Loaded {len(pairs)} misspelling pairs")
        return pairs

    def evaluate_correction_accuracy(
        self,
        corrector: LLMSpellCorrector,
        test_pairs: List[Tuple[str, str]]
    ) -> Dict:
        """
        Test correction accuracy

        Args:
            corrector: LLMSpellCorrector instance
            test_pairs: List of (correct, misspelled) tuples

        Returns:
            Dictionary with evaluation results
        """
        logger.info("=" * 80)
        logger.info("SPELL CORRECTION ACCURACY EVALUATION (Qwen 7B)")
        logger.info("=" * 80)

        # Extract misspelled names
        misspelled_names = [pair[1] for pair in test_pairs]
        ground_truth_map = {pair[1]: pair[0] for pair in test_pairs}

        # Run batch correction
        logger.info(f"Testing correction on {len(misspelled_names)} misspelled drug names...")
        correction_results = corrector.correct_batch(misspelled_names)

        # Calculate metrics
        exact_matches = 0
        total_edit_distance = 0
        unchanged_count = 0

        detailed_results = []

        logger.info("\nDetailed Correction Results:")
        logger.info("-" * 80)
        logger.info(f"{'Misspelled':<20} {'Qwen Correction':<20} {'Ground Truth':<20} {'Status':<15} {'Edit Dist'}")
        logger.info("-" * 80)

        for result in correction_results:
            original_misspelled = result.original
            corrected = result.corrected
            expected = ground_truth_map[original_misspelled]

            is_correct = corrected.lower() == expected.lower()
            edit_dist_from_truth = levenshtein_distance(corrected, expected)
            is_unchanged = original_misspelled.lower() == corrected.lower()

            if is_correct:
                exact_matches += 1
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"

            if is_unchanged:
                unchanged_count += 1

            total_edit_distance += edit_dist_from_truth

            # Log each correction
            logger.info(f"{original_misspelled:<20} {corrected:<20} {expected:<20} {status:<15} {edit_dist_from_truth}")

            detailed_results.append({
                'misspelled_input': original_misspelled,
                'qwen_correction': corrected,
                'ground_truth': expected,
                'is_correct': is_correct,
                'edit_distance_from_truth': edit_dist_from_truth,
                'edit_distance_from_input': result.edit_distance,
                'unchanged': is_unchanged,
                'confidence': result.confidence,
                'raw_response': result.raw_response
            })

        logger.info("-" * 80)

        # Calculate aggregate metrics
        accuracy = exact_matches / len(test_pairs) if len(test_pairs) > 0 else 0.0
        avg_edit_distance = total_edit_distance / len(test_pairs) if len(test_pairs) > 0 else 0.0
        unchanged_rate = unchanged_count / len(test_pairs) if len(test_pairs) > 0 else 0.0

        # Analyze failure modes
        failures = [r for r in detailed_results if not r['is_correct']]

        results = {
            'accuracy': accuracy,
            'exact_matches': exact_matches,
            'total_tested': len(test_pairs),
            'avg_edit_distance_from_truth': avg_edit_distance,
            'unchanged_rate': unchanged_rate,
            'unchanged_count': unchanged_count,
            'failure_count': len(failures),
            'detailed_results': detailed_results,
            'failures': failures
        }

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("ACCURACY SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Exact Match Accuracy:      {accuracy:.2%} ({exact_matches}/{len(test_pairs)})")
        logger.info(f"Avg Edit Distance (truth): {avg_edit_distance:.2f}")
        logger.info(f"Unchanged Rate:            {unchanged_rate:.2%} ({unchanged_count}/{len(test_pairs)})")
        logger.info(f"Failures:                  {len(failures)}/{len(test_pairs)}")
        logger.info("=" * 80)

        return results

    def test_overcorrection(self, corrector: LLMSpellCorrector, sample_size: int = 50) -> Dict:
        """
        Test if corrector wrongly changes already-correct drug names

        Args:
            corrector: LLMSpellCorrector instance
            sample_size: Number of correct drug names to test

        Returns:
            Dictionary with over-correction results
        """
        logger.info("\n" + "=" * 80)
        logger.info("OVER-CORRECTION TEST")
        logger.info("=" * 80)
        logger.info("Testing if Qwen wrongly changes already-correct drug names...")

        # Load correct drug names from evaluation dataset
        eval_path = "/home/omeerdogan23/drugRAG/data/processed/evaluation_dataset.csv"

        if not os.path.exists(eval_path):
            logger.warning(f"Evaluation dataset not found: {eval_path}")
            logger.warning("Skipping over-correction test")
            return {
                'over_correction_rate': 0.0,
                'false_changes': [],
                'tested_count': 0,
                'skipped': True
            }

        eval_df = pd.read_csv(eval_path)
        correct_drugs = eval_df['drug'].unique()[:sample_size]

        logger.info(f"Testing on {len(correct_drugs)} correct drug names...")

        # Run correction
        correction_results = corrector.correct_batch(list(correct_drugs))

        # Analyze over-corrections
        over_corrections = 0
        false_changes = []

        for result in correction_results:
            if result.changed:
                over_corrections += 1
                false_changes.append({
                    'correct_original': result.original,
                    'wrong_correction': result.corrected,
                    'confidence': result.confidence,
                    'edit_distance': result.edit_distance
                })

        over_correction_rate = over_corrections / len(correct_drugs) if len(correct_drugs) > 0 else 0.0

        logger.info(f"\nOver-correction Rate: {over_correction_rate:.2%} ({over_corrections}/{len(correct_drugs)})")

        if false_changes:
            logger.info("\nFalse Changes (Over-corrections):")
            for fc in false_changes[:10]:  # Show first 10
                logger.info(f"  {fc['correct_original']} → {fc['wrong_correction']} (confidence: {fc['confidence']:.2f})")

        return {
            'over_correction_rate': over_correction_rate,
            'over_correction_count': over_corrections,
            'false_changes': false_changes,
            'tested_count': len(correct_drugs),
            'skipped': False
        }

    def save_results(self, results: Dict, overcorrection_results: Dict, model_info: Dict, timestamp: str):
        """Save evaluation results to files"""
        # Save detailed results as JSON
        results_json_path = f"{self.results_dir}/accuracy_report_{timestamp}.json"

        full_report = {
            'model': 'Qwen 7B',
            'model_info': model_info,
            'correction_accuracy': results,
            'overcorrection_test': overcorrection_results,
            'timestamp': timestamp
        }

        with open(results_json_path, 'w') as f:
            json.dump(full_report, f, indent=2, default=str)

        logger.info(f"\nDetailed results saved to: {results_json_path}")

        # Save detailed corrections as CSV
        corrections_csv_path = f"{self.results_dir}/detailed_corrections_{timestamp}.csv"
        df_corrections = pd.DataFrame(results['detailed_results'])
        df_corrections.to_csv(corrections_csv_path, index=False)

        logger.info(f"Detailed corrections saved to: {corrections_csv_path}")

        # Save summary report
        summary_path = f"{self.results_dir}/summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SPELL CORRECTION ACCURACY EVALUATION - SUMMARY (Qwen 7B)\n")
            f.write("=" * 80 + "\n\n")

            f.write("MODEL: Qwen2.5-7B-Instruct (vLLM)\n")
            f.write(f"Few-shot: {model_info['use_fewshot']}\n")
            f.write(f"Temperature: {model_info['temperature']}\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            f.write("CORRECTION ACCURACY:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Exact Match Accuracy:      {results['accuracy']:.2%}\n")
            f.write(f"Exact Matches:             {results['exact_matches']}/{results['total_tested']}\n")
            f.write(f"Avg Edit Distance (truth): {results['avg_edit_distance_from_truth']:.2f}\n")
            f.write(f"Unchanged Rate:            {results['unchanged_rate']:.2%}\n")
            f.write(f"Failures:                  {results['failure_count']}\n\n")

            f.write("OVER-CORRECTION TEST:\n")
            f.write("-" * 80 + "\n")
            if not overcorrection_results['skipped']:
                f.write(f"Over-correction Rate:      {overcorrection_results['over_correction_rate']:.2%}\n")
                f.write(f"False Changes:             {overcorrection_results['over_correction_count']}/{overcorrection_results['tested_count']}\n")
            else:
                f.write("Skipped (evaluation dataset not found)\n")

        logger.info(f"Summary report saved to: {summary_path}")

    def run_full_evaluation(self, use_fewshot: bool = True, temperature: float = 0.0):
        """Run complete spell correction evaluation with Qwen 7B"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info("\n" + "=" * 80)
        logger.info("SPELL CORRECTION EVALUATION - STARTING")
        logger.info("=" * 80)
        logger.info("Model: Qwen2.5-7B-Instruct (vLLM)")
        logger.info(f"Few-shot: {use_fewshot}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info("=" * 80 + "\n")

        # Initialize corrector with Qwen 7B
        corrector = LLMSpellCorrector(
            use_fewshot=use_fewshot,
            temperature=temperature,
            config_path=self.config_path
        )

        model_info = {
            'use_fewshot': use_fewshot,
            'temperature': temperature
        }

        # Load test data
        test_pairs = self.load_misspellings()

        # Part 1: Correction accuracy
        correction_results = self.evaluate_correction_accuracy(corrector, test_pairs)

        # Part 2: Over-correction test
        overcorrection_results = self.test_overcorrection(corrector, sample_size=50)

        # Save all results
        self.save_results(correction_results, overcorrection_results, model_info, timestamp)

        logger.info("\n" + "=" * 80)
        logger.info("SPELL CORRECTION EVALUATION - COMPLETE")
        logger.info("=" * 80)

        return {
            'correction_accuracy': correction_results,
            'overcorrection': overcorrection_results,
            'model_info': model_info,
            'timestamp': timestamp
        }


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Evaluate Qwen 7B spell correction accuracy")

    parser.add_argument(
        '--no-fewshot',
        action='store_true',
        help="Disable few-shot examples (default: enabled)"
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for deterministic)"
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
        default='/home/omeerdogan23/drugRAG/results/spell_correction_experiment',
        help="Directory to save results"
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = SpellCorrectionEvaluator(
        config_path=args.config,
        results_dir=args.results_dir
    )

    evaluator.run_full_evaluation(
        use_fewshot=not args.no_fewshot,
        temperature=args.temperature
    )


if __name__ == "__main__":
    main()

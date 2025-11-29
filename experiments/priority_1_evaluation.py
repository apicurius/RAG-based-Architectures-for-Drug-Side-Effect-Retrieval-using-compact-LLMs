#!/usr/bin/env python3
"""
Priority 1: Critical Baseline Evaluation
Tests top 5 most common side effects with 3 architectures

Runtime: ~45 minutes (sequential) or ~15 minutes (with optimization)
Purpose: Quick validation that chunked strategy > monolithic
"""

import json
import time
import logging
from datetime import datetime
import sys
import os
from typing import Dict, Any, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.architectures.rag_format_b import FormatBRAG
from src.architectures.graphrag import GraphRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


class Priority1Evaluator:
    """Efficient evaluator using cached ground truth"""

    def __init__(self, config_path='../config.json', ground_truth_path='../data/processed/neo4j_ground_truth.json'):
        self.config_path = config_path
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.architectures = {}  # Lazy loading

    def _load_ground_truth(self, path):
        """Load pre-generated ground truth (instant)"""
        logger.info(f"📂 Loading ground truth from {path}")
        try:
            with open(path, 'r') as f:
                gt = json.load(f)
            logger.info(f"✅ Loaded ground truth for {len(gt)} side effects")
            return gt
        except FileNotFoundError:
            logger.error(f"❌ Ground truth not found at {path}")
            logger.error("   Run: python scripts/generate_ground_truth_neo4j.py")
            sys.exit(1)

    def get_architecture(self, name: str):
        """Lazy initialization - only load when needed"""
        if name not in self.architectures:
            logger.info(f"🔧 Initializing {name}...")

            if name == 'graphrag':
                self.architectures[name] = GraphRAG(self.config_path, model='qwen')

            elif name == 'format_b_chunked':
                rag = FormatBRAG(self.config_path, model='qwen')
                # Wrapper to add strategy parameter
                class ChunkedWrapper:
                    def __init__(self, rag):
                        self.rag = rag
                    def reverse_query(self, se):
                        return self.rag.reverse_query(se, strategy='chunked')
                self.architectures[name] = ChunkedWrapper(rag)

            elif name == 'format_b_monolithic':
                rag = FormatBRAG(self.config_path, model='qwen')
                # Wrapper for monolithic
                class MonolithicWrapper:
                    def __init__(self, rag):
                        self.rag = rag
                    def reverse_query(self, se):
                        return self.rag.reverse_query(se, strategy='monolithic')
                self.architectures[name] = MonolithicWrapper(rag)

            logger.info(f"✅ {name} initialized")

        return self.architectures[name]

    def evaluate_single_query(self, side_effect: str, architecture_name: str) -> Dict[str, Any]:
        """
        Evaluate single reverse query using cached ground truth

        Args:
            side_effect: Side effect to query
            architecture_name: Which architecture to test

        Returns:
            Dict with metrics and results
        """
        # Get ground truth (instant lookup from cached data)
        expected_drugs = set(d.lower().strip() for d in self.ground_truth.get(side_effect, []))
        expected_count = len(expected_drugs)

        logger.info(f"\n{'='*80}")
        logger.info(f"Query: Which drugs cause {side_effect}?")
        logger.info(f"Expected: {expected_count} drugs")
        logger.info(f"Architecture: {architecture_name}")
        logger.info(f"{'='*80}")

        # Run architecture query
        arch = self.get_architecture(architecture_name)
        start_time = time.time()

        try:
            result = arch.reverse_query(side_effect)
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            return {
                'side_effect': side_effect,
                'architecture': architecture_name,
                'error': str(e),
                'expected_count': expected_count
            }

        elapsed = time.time() - start_time

        # Extract and normalize results
        extracted_drugs = set(d.lower().strip() for d in result.get('drugs', []))
        extracted_count = len(extracted_drugs)

        # Calculate metrics (instant with cached ground truth)
        tp = len(expected_drugs & extracted_drugs)  # True positives
        fp = len(extracted_drugs - expected_drugs)  # False positives
        fn = len(expected_drugs - extracted_drugs)  # False negatives

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Pair coverage (how many of retrieved pairs were extracted)
        retrieved_pairs = result.get('retrieved_pairs', 0)
        pair_coverage = extracted_count / retrieved_pairs if retrieved_pairs > 0 else 0

        # Log results
        logger.info(f"\n📊 RESULTS:")
        logger.info(f"   Retrieved pairs: {retrieved_pairs}")

        if 'chunks_processed' in result:
            logger.info(f"   Chunks processed: {result['chunks_processed']} (size: {result.get('chunk_size', 'N/A')})")

        logger.info(f"   Extracted drugs: {extracted_count}")
        logger.info(f"   Expected drugs: {expected_count}")
        logger.info(f"\n📈 METRICS:")
        logger.info(f"   True Positives:  {tp}")
        logger.info(f"   False Positives: {fp}")
        logger.info(f"   False Negatives: {fn}")
        logger.info(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        logger.info(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
        logger.info(f"   F1 Score:  {f1:.4f}")
        logger.info(f"   Pair Coverage: {pair_coverage:.4f} ({pair_coverage*100:.2f}%)")
        logger.info(f"   Time: {elapsed:.2f}s")

        return {
            'side_effect': side_effect,
            'architecture': architecture_name,
            'expected_count': expected_count,
            'extracted_count': extracted_count,
            'retrieved_pairs': retrieved_pairs,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'pair_coverage': pair_coverage,
            'elapsed_time': elapsed,
            'chunks_processed': result.get('chunks_processed'),
            'chunk_size': result.get('chunk_size'),
            'timestamp': datetime.now().isoformat()
        }

    def run_priority_1_test(self, critical_ses: List[str], architectures: List[str]):
        """
        Run Priority 1 test: 5 critical SEs × 3 architectures = 15 evaluations

        Args:
            critical_ses: List of critical side effects (top 5)
            architectures: List of architecture names to test

        Returns:
            Dict with all results and summary
        """
        logger.info("\n" + "="*80)
        logger.info("PRIORITY 1: CRITICAL BASELINE EVALUATION")
        logger.info("="*80)
        logger.info(f"Side effects: {len(critical_ses)}")
        logger.info(f"Architectures: {len(architectures)}")
        logger.info(f"Total evaluations: {len(critical_ses) * len(architectures)}")
        logger.info("="*80 + "\n")

        all_results = {}
        start_time = time.time()

        for arch_name in architectures:
            logger.info(f"\n🧪 TESTING ARCHITECTURE: {arch_name.upper()}")
            logger.info("="*80)

            arch_results = []
            for se in critical_ses:
                result = self.evaluate_single_query(se, arch_name)
                arch_results.append(result)

            all_results[arch_name] = arch_results

            # Calculate aggregate metrics for this architecture
            avg_precision = sum(r['precision'] for r in arch_results) / len(arch_results)
            avg_recall = sum(r['recall'] for r in arch_results) / len(arch_results)
            avg_f1 = sum(r['f1_score'] for r in arch_results) / len(arch_results)
            avg_time = sum(r['elapsed_time'] for r in arch_results) / len(arch_results)
            total_time = sum(r['elapsed_time'] for r in arch_results)

            logger.info(f"\n📊 AGGREGATE METRICS FOR {arch_name}:")
            logger.info(f"   Average Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
            logger.info(f"   Average Recall:    {avg_recall:.4f} ({avg_recall*100:.2f}%)")
            logger.info(f"   Average F1 Score:  {avg_f1:.4f}")
            logger.info(f"   Average Time:      {avg_time:.2f}s")
            logger.info(f"   Total Time:        {total_time:.2f}s")

        total_elapsed = time.time() - start_time

        # Print comparison summary
        self._print_comparison_summary(all_results, total_elapsed)

        # Save results
        output = {
            'test_name': 'Priority 1 - Critical Baseline',
            'side_effects': critical_ses,
            'architectures': architectures,
            'results': all_results,
            'total_elapsed_time': total_elapsed,
            'timestamp': datetime.now().isoformat()
        }

        output_file = f"results_priority_1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\n💾 Results saved to: {output_file}")

        return output

    def _print_comparison_summary(self, all_results: Dict, total_elapsed: float):
        """Print side-by-side comparison of all architectures"""

        logger.info("\n" + "="*80)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*80)

        # Create comparison table
        arch_names = list(all_results.keys())
        headers = ['Side Effect'] + arch_names

        # Print header
        header_line = f"{'Side Effect':<25}"
        for arch in arch_names:
            header_line += f" {arch:<20}"
        logger.info(header_line)
        logger.info("-" * 80)

        # Get side effects (assume all architectures tested same SEs)
        first_arch = arch_names[0]
        side_effects = [r['side_effect'] for r in all_results[first_arch]]

        # Print results per SE
        for se in side_effects:
            line = f"{se:<25}"
            for arch in arch_names:
                # Find result for this SE and arch
                result = next(r for r in all_results[arch] if r['side_effect'] == se)
                recall = result['recall']
                line += f" {recall*100:>5.1f}% recall     "
            logger.info(line)

        logger.info("-" * 80)

        # Print averages
        avg_line = f"{'AVERAGE':<25}"
        for arch in arch_names:
            avg_recall = sum(r['recall'] for r in all_results[arch]) / len(all_results[arch])
            avg_line += f" {avg_recall*100:>5.1f}% recall     "
        logger.info(avg_line)

        logger.info("-" * 80)

        # Calculate improvements
        if 'format_b_monolithic' in arch_names and 'format_b_chunked' in arch_names:
            mono_avg = sum(r['recall'] for r in all_results['format_b_monolithic']) / len(all_results['format_b_monolithic'])
            chunk_avg = sum(r['recall'] for r in all_results['format_b_chunked']) / len(all_results['format_b_chunked'])
            improvement = ((chunk_avg - mono_avg) / mono_avg) * 100 if mono_avg > 0 else 0

            logger.info(f"\n🎯 KEY FINDING:")
            logger.info(f"   Monolithic Avg Recall: {mono_avg*100:.2f}%")
            logger.info(f"   Chunked Avg Recall:    {chunk_avg*100:.2f}%")
            logger.info(f"   Improvement:           {improvement:+.1f}%")

        if 'graphrag' in arch_names and 'format_b_chunked' in arch_names:
            graph_avg = sum(r['recall'] for r in all_results['graphrag']) / len(all_results['graphrag'])
            chunk_avg = sum(r['recall'] for r in all_results['format_b_chunked']) / len(all_results['format_b_chunked'])
            ratio = (chunk_avg / graph_avg) * 100 if graph_avg > 0 else 0

            logger.info(f"\n📊 vs GraphRAG Baseline:")
            logger.info(f"   GraphRAG Recall:    {graph_avg*100:.2f}%")
            logger.info(f"   Chunked Recall:     {chunk_avg*100:.2f}%")
            logger.info(f"   Chunked achieves {ratio:.1f}% of GraphRAG performance")

        logger.info(f"\n⏱️  Total Evaluation Time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} min)")
        logger.info("="*80 + "\n")


def main():
    """Main execution"""
    logger.info("="*80)
    logger.info("PRIORITY 1 EVALUATION")
    logger.info("="*80 + "\n")

    # Initialize evaluator
    evaluator = Priority1Evaluator()

    # Load critical test set (top 5 side effects)
    try:
        with open('../data/processed/critical_test_set.json', 'r') as f:
            critical_data = json.load(f)
            critical_ses = critical_data['side_effects']
        logger.info(f"✅ Loaded critical test set: {critical_ses}")
    except FileNotFoundError:
        logger.warning("⚠️  Critical test set not found, using default top 5")
        # Fallback to known common side effects
        critical_ses = ['nausea', 'dizziness', 'headache', 'dry mouth', 'thrombocytopenia']

    # Architectures to test - all three for comprehensive comparison
    architectures = [
        'graphrag',
        'format_b_chunked',
        'format_b_monolithic'
    ]

    # Run evaluation
    results = evaluator.run_priority_1_test(critical_ses, architectures)

    logger.info("✅ Priority 1 evaluation complete!")
    logger.info("\nNext steps:")
    logger.info("1. Review results in the generated JSON file")
    logger.info("2. If chunked > monolithic confirmed, proceed to Priority 2 (large queries)")
    logger.info("3. Run: python experiments/priority_2_evaluation.py")


if __name__ == "__main__":
    main()

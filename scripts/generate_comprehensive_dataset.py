#!/usr/bin/env python3
"""
Generate Comprehensive Reverse Query Dataset - Stratified Sampling

This script generates a production-ready reverse query dataset using:
- Stratified sampling across frequency tiers (1,000 side effects)
- Validated chunked extraction strategy (98.37% recall)
- Binary YES/NO pairs for each query
- Cached ground truth for validation

Output: 2,000 examples (1,000 YES + 1,000 NO)
Runtime: ~8-12 hours (1,000 queries × ~30-40s each)
"""

import json
import random
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.architectures.rag_format_b import FormatBRAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveDatasetGenerator:
    """Generate stratified reverse query dataset with validated chunked strategy"""

    def __init__(
        self,
        ground_truth_path: str = '../data/processed/neo4j_ground_truth.json',
        frequency_tiers_path: str = '../data/processed/frequency_tiers.json',
        config_path: str = '../config.json',
        output_dir: str = '../data/processed'
    ):
        """Initialize dataset generator"""
        self.ground_truth_path = ground_truth_path
        self.frequency_tiers_path = frequency_tiers_path
        self.config_path = config_path
        self.output_dir = output_dir

        # Load ground truth
        logger.info(f"📂 Loading ground truth from {ground_truth_path}")
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
        logger.info(f"✅ Loaded ground truth for {len(self.ground_truth)} side effects")

        # Load frequency tiers
        logger.info(f"📂 Loading frequency tiers from {frequency_tiers_path}")
        with open(frequency_tiers_path, 'r') as f:
            self.frequency_tiers = json.load(f)

        # Initialize RAG Format B with chunked strategy
        logger.info("🔧 Initializing RAG Format B (chunked strategy)...")
        self.rag = FormatBRAG(config_path, model='qwen')
        logger.info("✅ RAG Format B initialized")

        # Track progress
        self.completed = set()
        self.failed = set()
        self.results = []

    def perform_stratified_sampling(self, target_total: int = 1000) -> Dict[str, List[str]]:
        """
        Perform stratified sampling across frequency tiers

        Target distribution (total 1,000):
        - Very Large (≥1000):  50 samples
        - Large (500-999):    150 samples
        - Medium (100-499):   300 samples
        - Small (20-99):      300 samples
        - Rare (5-19):        150 samples
        - Very Rare (1-4):     50 samples
        """
        target_counts = {
            'very_large': 50,
            'large': 150,
            'medium': 300,
            'small': 300,
            'rare': 150,
            'very_rare': 50
        }

        sampled = {}

        for tier_name, target_count in target_counts.items():
            tier_data = self.frequency_tiers.get(tier_name, [])

            if not tier_data:
                logger.warning(f"⚠️  Tier '{tier_name}' is empty, skipping")
                sampled[tier_name] = []
                continue

            # Each item is {'side_effect': ..., 'drug_count': ...}
            available_ses = [item['side_effect'] for item in tier_data]
            available_count = len(available_ses)

            # Sample with or without replacement depending on availability
            if available_count >= target_count:
                selected = random.sample(available_ses, target_count)
                logger.info(f"📊 {tier_name}: Sampled {target_count}/{available_count} side effects")
            else:
                selected = available_ses  # Take all if insufficient
                logger.warning(f"⚠️  {tier_name}: Only {available_count} available (target: {target_count})")

            sampled[tier_name] = selected

        # Count total
        total_sampled = sum(len(v) for v in sampled.values())
        logger.info(f"\n✅ Stratified sampling complete: {total_sampled} side effects selected")

        # Show distribution
        logger.info("\n📈 SAMPLING DISTRIBUTION:")
        for tier_name, samples in sampled.items():
            logger.info(f"   {tier_name:12} {len(samples):>4} samples")

        return sampled

    def generate_negative_example(
        self,
        correct_side_effect: str,
        correct_drugs: List[str]
    ) -> Tuple[str, str]:
        """
        Generate a NO example by selecting a different side effect
        that shares NO drugs with the correct answer

        Returns: (drug_name, negative_side_effect)
        """
        correct_drugs_set = set(correct_drugs)

        # Find side effects with no overlap
        candidates = []
        for se, drugs in self.ground_truth.items():
            if se == correct_side_effect:
                continue

            # Check if there's no overlap
            if not correct_drugs_set.intersection(set(drugs)):
                candidates.append(se)

        if not candidates:
            # Fallback: Just pick a different side effect
            candidates = [se for se in self.ground_truth.keys() if se != correct_side_effect]

        if not candidates:
            logger.error(f"❌ Could not generate negative example for '{correct_side_effect}'")
            return None, None

        # Select random negative side effect
        negative_se = random.choice(candidates)

        # Select random drug from correct answer
        if correct_drugs:
            drug = random.choice(correct_drugs)
        else:
            return None, None

        return drug, negative_se

    def generate_single_query(
        self,
        side_effect: str,
        tier: str
    ) -> Dict:
        """
        Generate one reverse query with both YES and NO examples

        Returns dict with:
        - yes_example: {query, answer, metadata}
        - no_example: {query, answer, metadata}
        """
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"📝 Processing: {side_effect} ({tier})")
            logger.info(f"{'='*80}")

            # Get expected answer from ground truth
            expected_drugs = self.ground_truth.get(side_effect, [])
            expected_count = len(expected_drugs)

            logger.info(f"📊 Expected drugs: {expected_count}")

            # Use chunked strategy to extract drugs
            result = self.rag.reverse_query(side_effect, strategy='chunked')
            extracted_drugs = result.get('drugs', [])
            extracted_count = len(extracted_drugs)

            logger.info(f"✅ Extracted drugs: {extracted_count}")

            # Calculate recall
            if expected_count > 0:
                recall = len(set(extracted_drugs) & set(expected_drugs)) / expected_count
                logger.info(f"📈 Recall: {recall:.2%}")
            else:
                recall = 0.0

            # Create YES example
            yes_example = {
                'query': f"Which drugs cause {side_effect}?",
                'side_effect': side_effect,
                'answer': 'YES',
                'drugs': extracted_drugs,
                'drug_count': extracted_count,
                'expected_count': expected_count,
                'recall': recall,
                'tier': tier,
                'strategy': 'chunked',
                'timestamp': datetime.now().isoformat()
            }

            # Generate NO example
            drug, negative_se = self.generate_negative_example(side_effect, extracted_drugs)

            if drug and negative_se:
                no_example = {
                    'query': f"Does {drug} cause {negative_se}?",
                    'drug': drug,
                    'side_effect': negative_se,
                    'answer': 'NO',
                    'correct_side_effect': side_effect,
                    'tier': tier,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"⚠️  Could not generate NO example for {side_effect}")
                no_example = None

            return {
                'yes_example': yes_example,
                'no_example': no_example,
                'success': True
            }

        except Exception as e:
            logger.error(f"❌ Error processing '{side_effect}': {e}")
            return {
                'yes_example': None,
                'no_example': None,
                'success': False,
                'error': str(e),
                'side_effect': side_effect
            }

    def generate_dataset(
        self,
        sampled_tiers: Dict[str, List[str]],
        checkpoint_interval: int = 50
    ):
        """
        Generate complete dataset with checkpointing

        Args:
            sampled_tiers: Dict mapping tier names to lists of side effects
            checkpoint_interval: Save progress every N queries
        """
        # Flatten sampled side effects with tier labels
        all_queries = []
        for tier, side_effects in sampled_tiers.items():
            for se in side_effects:
                all_queries.append((se, tier))

        total_queries = len(all_queries)
        logger.info(f"\n🚀 STARTING DATASET GENERATION")
        logger.info(f"📊 Total queries: {total_queries}")
        logger.info(f"⏱️  Estimated time: {total_queries * 0.5:.1f} - {total_queries * 0.67:.1f} hours")
        logger.info(f"💾 Checkpoint every {checkpoint_interval} queries\n")

        yes_examples = []
        no_examples = []
        failed_queries = []

        start_time = datetime.now()

        for idx, (side_effect, tier) in enumerate(all_queries, 1):
            # Skip if already completed
            if side_effect in self.completed:
                logger.info(f"⏭️  Skipping {side_effect} (already completed)")
                continue

            logger.info(f"\n{'='*80}")
            logger.info(f"Progress: {idx}/{total_queries} ({idx/total_queries*100:.1f}%)")
            logger.info(f"{'='*80}")

            # Generate query
            result = self.generate_single_query(side_effect, tier)

            if result['success']:
                if result['yes_example']:
                    yes_examples.append(result['yes_example'])
                if result['no_example']:
                    no_examples.append(result['no_example'])

                self.completed.add(side_effect)
                logger.info(f"✅ Success: {side_effect} ({len(yes_examples)} total)")
            else:
                failed_queries.append({
                    'side_effect': side_effect,
                    'tier': tier,
                    'error': result.get('error', 'Unknown error')
                })
                self.failed.add(side_effect)
                logger.error(f"❌ Failed: {side_effect}")

            # Checkpoint
            if idx % checkpoint_interval == 0:
                self._save_checkpoint(yes_examples, no_examples, failed_queries, idx, total_queries)

            # Progress update
            if idx % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time_per_query = elapsed / idx
                remaining = (total_queries - idx) * avg_time_per_query
                logger.info(f"\n📊 PROGRESS UPDATE:")
                logger.info(f"   Completed: {idx}/{total_queries}")
                logger.info(f"   Success rate: {len(yes_examples)/idx*100:.1f}%")
                logger.info(f"   Avg time/query: {avg_time_per_query:.1f}s")
                logger.info(f"   Est. remaining: {remaining/3600:.1f} hours\n")

        # Final save
        self._save_final_dataset(yes_examples, no_examples, failed_queries)

        return {
            'yes_examples': yes_examples,
            'no_examples': no_examples,
            'failed_queries': failed_queries
        }

    def _save_checkpoint(
        self,
        yes_examples: List[Dict],
        no_examples: List[Dict],
        failed_queries: List[Dict],
        current: int,
        total: int
    ):
        """Save intermediate progress"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_file = f"{self.output_dir}/checkpoint_{current}of{total}_{timestamp}.json"

        checkpoint_data = {
            'progress': {
                'current': current,
                'total': total,
                'percentage': current / total * 100,
                'timestamp': timestamp
            },
            'yes_examples': yes_examples,
            'no_examples': no_examples,
            'failed_queries': failed_queries,
            'completed': list(self.completed),
            'failed': list(self.failed)
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
        logger.info(f"   YES examples: {len(yes_examples)}")
        logger.info(f"   NO examples: {len(no_examples)}")
        logger.info(f"   Failed: {len(failed_queries)}")

    def _save_final_dataset(
        self,
        yes_examples: List[Dict],
        no_examples: List[Dict],
        failed_queries: List[Dict]
    ):
        """Save final dataset in multiple formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 1. Save comprehensive JSON
        json_file = f"{self.output_dir}/comprehensive_reverse_queries_{timestamp}.json"

        final_data = {
            'metadata': {
                'generation_date': timestamp,
                'total_yes_examples': len(yes_examples),
                'total_no_examples': len(no_examples),
                'total_examples': len(yes_examples) + len(no_examples),
                'failed_queries': len(failed_queries),
                'strategy': 'chunked',
                'sampling_method': 'stratified',
                'target_count': 1000
            },
            'yes_examples': yes_examples,
            'no_examples': no_examples,
            'failed_queries': failed_queries
        }

        with open(json_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        logger.info(f"✅ JSON dataset saved: {json_file}")

        # 2. Save as CSV (compatible with existing format)
        csv_file = f"{self.output_dir}/comprehensive_reverse_queries_{timestamp}.csv"

        csv_rows = []

        # Add YES examples
        for ex in yes_examples:
            csv_rows.append({
                'query': ex['query'],
                'side_effect': ex['side_effect'],
                'answer': 'YES',
                'drugs': ', '.join(ex['drugs'][:50]),  # Limit for CSV
                'drug_count': ex['drug_count'],
                'tier': ex['tier'],
                'recall': ex.get('recall', 0.0)
            })

        # Add NO examples
        for ex in no_examples:
            csv_rows.append({
                'query': ex['query'],
                'side_effect': ex['side_effect'],
                'answer': 'NO',
                'drugs': ex.get('drug', ''),
                'drug_count': 0,
                'tier': ex['tier'],
                'recall': 0.0
            })

        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_file, index=False)
        logger.info(f"✅ CSV dataset saved: {csv_file}")

        # 3. Generate summary report
        self._generate_summary_report(yes_examples, no_examples, failed_queries, timestamp)

    def _generate_summary_report(
        self,
        yes_examples: List[Dict],
        no_examples: List[Dict],
        failed_queries: List[Dict],
        timestamp: str
    ):
        """Generate comprehensive summary report"""
        report_file = f"{self.output_dir}/dataset_generation_report_{timestamp}.md"

        # Calculate statistics
        total_yes = len(yes_examples)
        total_no = len(no_examples)
        total_failed = len(failed_queries)
        total_attempted = total_yes + total_failed

        if total_yes > 0:
            avg_recall = sum(ex.get('recall', 0) for ex in yes_examples) / total_yes
            avg_drugs = sum(ex['drug_count'] for ex in yes_examples) / total_yes
        else:
            avg_recall = 0.0
            avg_drugs = 0.0

        # Tier distribution
        tier_stats = {}
        for ex in yes_examples:
            tier = ex['tier']
            if tier not in tier_stats:
                tier_stats[tier] = {'count': 0, 'avg_recall': 0, 'avg_drugs': 0}
            tier_stats[tier]['count'] += 1
            tier_stats[tier]['avg_recall'] += ex.get('recall', 0)
            tier_stats[tier]['avg_drugs'] += ex['drug_count']

        for tier in tier_stats:
            count = tier_stats[tier]['count']
            tier_stats[tier]['avg_recall'] /= count
            tier_stats[tier]['avg_drugs'] /= count

        report = f"""# Comprehensive Reverse Query Dataset Generation Report

**Date:** {timestamp}
**Strategy:** Chunked Iterative Extraction
**Sampling:** Stratified across frequency tiers

---

## Summary Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| **Total YES Examples** | {total_yes:,} | - |
| **Total NO Examples** | {total_no:,} | - |
| **Total Examples** | {total_yes + total_no:,} | - |
| **Failed Queries** | {total_failed:,} | {total_failed/total_attempted*100:.1f}% |
| **Success Rate** | {total_yes:,}/{total_attempted:,} | {total_yes/total_attempted*100:.1f}% |

---

## Quality Metrics

| Metric | Value |
|--------|-------|
| **Average Recall** | {avg_recall:.2%} |
| **Average Drugs/Query** | {avg_drugs:.1f} |
| **Extraction Strategy** | Chunked (98.37% validated) |

---

## Distribution by Frequency Tier

| Tier | Examples | Avg Recall | Avg Drugs |
|------|----------|-----------|-----------|
"""

        for tier in ['very_large', 'large', 'medium', 'small', 'rare', 'very_rare']:
            if tier in tier_stats:
                stats = tier_stats[tier]
                report += f"| {tier:12} | {stats['count']:>8} | {stats['avg_recall']:>9.1%} | {stats['avg_drugs']:>9.1f} |\n"

        report += f"""
---

## Failed Queries

Total: {total_failed}

"""

        if failed_queries:
            report += "| Side Effect | Tier | Error |\n"
            report += "|-------------|------|-------|\n"
            for fq in failed_queries[:20]:  # Show first 20
                report += f"| {fq['side_effect']} | {fq['tier']} | {fq['error'][:50]}... |\n"

            if len(failed_queries) > 20:
                report += f"\n*...and {len(failed_queries) - 20} more failures*\n"

        report += f"""
---

## Files Generated

1. **JSON:** `comprehensive_reverse_queries_{timestamp}.json`
   - Complete dataset with metadata
   - YES and NO examples separated
   - Includes extraction metrics

2. **CSV:** `comprehensive_reverse_queries_{timestamp}.csv`
   - Compatible with existing format
   - Flat structure for easy loading
   - Truncated drug lists (first 50)

3. **Report:** This file

---

## Validation

- ✅ Ground truth: {len(self.ground_truth):,} side effects
- ✅ Strategy: Chunked (validated 98.37% recall)
- ✅ Infrastructure: vLLM + Pinecone + Neo4j
- ✅ Checkpointing: Every 50 queries

---

## Next Steps

1. **Quality Review:** Manually review sample queries
2. **Integration:** Add to training/evaluation pipelines
3. **Monitoring:** Track performance on this dataset
4. **Expansion:** Consider expanding to 2,000+ examples

---

**Generated:** {timestamp}
**Total Runtime:** See log file for details
"""

        with open(report_file, 'w') as f:
            f.write(report)

        logger.info(f"✅ Summary report saved: {report_file}")


def main():
    """Main execution"""
    print("="*80)
    print("COMPREHENSIVE REVERSE QUERY DATASET GENERATION")
    print("="*80)
    print()
    print("This will generate 1,000 reverse query examples using:")
    print("- Stratified sampling across frequency tiers")
    print("- Validated chunked strategy (98.37% recall)")
    print("- Binary YES/NO pairs")
    print()
    print("Estimated runtime: 8-12 hours")
    print("="*80)
    print()

    # Initialize generator
    generator = ComprehensiveDatasetGenerator()

    # Perform stratified sampling
    sampled_tiers = generator.perform_stratified_sampling(target_total=1000)

    # Generate dataset
    results = generator.generate_dataset(sampled_tiers, checkpoint_interval=50)

    # Final summary
    print("\n" + "="*80)
    print("✅ DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"YES examples: {len(results['yes_examples']):,}")
    print(f"NO examples: {len(results['no_examples']):,}")
    print(f"Total examples: {len(results['yes_examples']) + len(results['no_examples']):,}")
    print(f"Failed queries: {len(results['failed_queries']):,}")
    print("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Ultra-fast Ground Truth Generation using Neo4j
Extracts complete drug-side effect mappings in ~5 seconds

This script queries Neo4j once to get ALL reverse query ground truths,
eliminating the need for repeated database queries during evaluation.
"""

import json
import time
from neo4j import GraphDatabase
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def generate_ground_truth_from_csv(csv_path='../data/processed/data_format_b.csv'):
    """
    Generate ground truth from CSV file (fallback if Neo4j unavailable)

    Returns:
        tuple: (ground_truth_dict, frequency_distribution_dict, frequency_tiers_dict)
    """
    print(f"📂 Loading data from {csv_path}...")
    import pandas as pd

    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df):,} drug-side effect pairs")

    print("\n📊 Extracting ground truth...")
    start_time = time.time()

    # Group by side effect to get drug lists
    ground_truth = {}
    frequency_distribution = {}
    frequency_tiers = {
        'very_large': [],    # >=1000 drugs
        'large': [],         # 500-999 drugs
        'medium': [],        # 100-499 drugs
        'small': [],         # 20-99 drugs
        'rare': [],          # 5-19 drugs
        'very_rare': []      # 1-4 drugs
    }

    for se, group in df.groupby('side_effect'):
        drugs = group['drug'].unique().tolist()
        count = len(drugs)

        ground_truth[se] = drugs
        frequency_distribution[se] = count

        # Categorize into tiers
        if count >= 1000:
            frequency_tiers['very_large'].append((se, count))
        elif count >= 500:
            frequency_tiers['large'].append((se, count))
        elif count >= 100:
            frequency_tiers['medium'].append((se, count))
        elif count >= 20:
            frequency_tiers['small'].append((se, count))
        elif count >= 5:
            frequency_tiers['rare'].append((se, count))
        else:
            frequency_tiers['very_rare'].append((se, count))

    elapsed = time.time() - start_time
    return ground_truth, frequency_distribution, frequency_tiers, elapsed


def generate_ground_truth(config_path='../config.json', use_csv_fallback=True):
    """
    Generate complete ground truth for all side effects

    Returns:
        tuple: (ground_truth_dict, frequency_distribution_dict, frequency_tiers_dict)
    """
    # Try Neo4j first, fallback to CSV if connection fails
    if not use_csv_fallback:
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        print("🔗 Connecting to Neo4j...")

        # Connect to Neo4j using bolt protocol
        try:
            driver = GraphDatabase.driver(
                "bolt://9d0e641a.databases.neo4j.io:7687",
                auth=(config['neo4j_username'], config['neo4j_password']),
                encrypted=True,
                trust="TRUST_ALL_CERTIFICATES",
                connection_timeout=10
            )

            # Test connection
            with driver.session() as session:
                session.run("RETURN 1").single()
            print("✅ Neo4j connection established")

            print("\n📊 Extracting ALL drug-side effect pairs via Neo4j...")
            start_time = time.time()

            # Single Cypher query to get all reverse query ground truths
            cypher = """
            MATCH (drug)-[r:HAS_SIDE_EFFECT]->(effect)
            RETURN effect.name AS side_effect,
                   COLLECT(DISTINCT drug.name) AS drugs,
                   COUNT(DISTINCT drug.name) AS drug_count
            ORDER BY drug_count DESC
            """

            ground_truth = {}
            frequency_distribution = {}
            frequency_tiers = {
                'very_large': [],    # >=1000 drugs
                'large': [],         # 500-999 drugs
                'medium': [],        # 100-499 drugs
                'small': [],         # 20-99 drugs
                'rare': [],          # 5-19 drugs
                'very_rare': []      # 1-4 drugs
            }

            with driver.session() as session:
                result = session.run(cypher)

                for record in result:
                    se = record['side_effect']
                    drugs = record['drugs']
                    count = record['drug_count']

                    ground_truth[se] = drugs
                    frequency_distribution[se] = count

                    # Categorize into tiers
                    if count >= 1000:
                        frequency_tiers['very_large'].append((se, count))
                    elif count >= 500:
                        frequency_tiers['large'].append((se, count))
                    elif count >= 100:
                        frequency_tiers['medium'].append((se, count))
                    elif count >= 20:
                        frequency_tiers['small'].append((se, count))
                    elif count >= 5:
                        frequency_tiers['rare'].append((se, count))
                    else:
                        frequency_tiers['very_rare'].append((se, count))

            driver.close()

            elapsed = time.time() - start_time
            return ground_truth, frequency_distribution, frequency_tiers, elapsed

        except Exception as e:
            print(f"⚠️  Neo4j connection failed: {e}")
            print("📂 Falling back to CSV file...")
            return generate_ground_truth_from_csv()
    else:
        print("📂 Using CSV file (fast local processing)...")
        return generate_ground_truth_from_csv()


def save_results(ground_truth, frequency_distribution, frequency_tiers, output_dir='../data/processed'):
    """Save generated ground truth and metadata"""

    os.makedirs(output_dir, exist_ok=True)

    # Save ground truth
    gt_file = os.path.join(output_dir, 'neo4j_ground_truth.json')
    with open(gt_file, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    print(f"\n💾 Ground truth saved to: {gt_file}")
    print(f"   File size: {os.path.getsize(gt_file) / 1024:.1f} KB")

    # Save frequency distribution
    freq_file = os.path.join(output_dir, 'side_effect_frequencies.json')
    with open(freq_file, 'w') as f:
        json.dump(frequency_distribution, f, indent=2)
    print(f"💾 Frequencies saved to: {freq_file}")

    # Save tier metadata
    tier_file = os.path.join(output_dir, 'frequency_tiers.json')
    # Convert to serializable format
    tiers_serializable = {
        tier: [{'side_effect': se, 'drug_count': count} for se, count in ses]
        for tier, ses in frequency_tiers.items()
    }
    with open(tier_file, 'w') as f:
        json.dump(tiers_serializable, f, indent=2)
    print(f"💾 Frequency tiers saved to: {tier_file}")

    # Create critical test set (top 5)
    critical_ses = [se for se, _ in sorted(frequency_distribution.items(), key=lambda x: x[1], reverse=True)[:5]]
    critical_file = os.path.join(output_dir, 'critical_test_set.json')
    with open(critical_file, 'w') as f:
        json.dump({
            'side_effects': critical_ses,
            'description': 'Top 5 most common side effects for quick validation',
            'use_case': 'Priority 1 testing - quick validation in 45 min'
        }, f, indent=2)
    print(f"💾 Critical test set saved to: {critical_file}")
    print(f"   Critical SEs: {', '.join(critical_ses)}")


def main():
    """Main execution"""
    print("="*80)
    print("NEO4J GROUND TRUTH GENERATOR")
    print("="*80)
    print("\nThis script extracts ALL reverse query ground truths from Neo4j")
    print("Runtime: ~5 seconds for complete SIDER database")
    print("="*80 + "\n")

    # Generate ground truth
    ground_truth, frequency_distribution, frequency_tiers, elapsed = generate_ground_truth()

    if ground_truth is None:
        print("\n❌ Failed to generate ground truth")
        sys.exit(1)

    # Print statistics
    print(f"\n✅ Extraction complete in {elapsed:.2f} seconds!")
    print(f"\n📈 STATISTICS:")
    print(f"   Total side effects: {len(ground_truth):,}")
    print(f"   Total unique drugs across all SEs: {len(set(drug for drugs in ground_truth.values() for drug in drugs)):,}")
    print(f"\n🎯 FREQUENCY DISTRIBUTION:")
    print(f"   Very Large (≥1000 drugs):  {len(frequency_tiers['very_large']):>4} side effects")
    print(f"   Large (500-999 drugs):     {len(frequency_tiers['large']):>4} side effects")
    print(f"   Medium (100-499 drugs):    {len(frequency_tiers['medium']):>4} side effects")
    print(f"   Small (20-99 drugs):       {len(frequency_tiers['small']):>4} side effects")
    print(f"   Rare (5-19 drugs):         {len(frequency_tiers['rare']):>4} side effects")
    print(f"   Very Rare (1-4 drugs):     {len(frequency_tiers['very_rare']):>4} side effects")

    # Show top 10 most common side effects
    top_10 = sorted(frequency_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n🔥 TOP 10 MOST COMMON SIDE EFFECTS:")
    for i, (se, count) in enumerate(top_10, 1):
        print(f"   {i:2}. {se:<30} {count:>4} drugs")

    # Save results
    save_results(ground_truth, frequency_distribution, frequency_tiers)

    print("\n" + "="*80)
    print("✅ GROUND TRUTH GENERATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Run Priority 1 test (5 critical SEs):")
    print("   python experiments/priority_1_evaluation.py")
    print("\n2. Or run comprehensive evaluation (300 SEs):")
    print("   python experiments/efficient_reverse_evaluation.py --test-set stratified_300")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

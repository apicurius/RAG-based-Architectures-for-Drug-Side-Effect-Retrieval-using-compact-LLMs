#!/usr/bin/env python3
"""
Create combined CSV files for llama and qwen models
Similar format to results_RAG_models_ALLdrugs_10randomADRs_llama38B.xlsx
"""
import json
import csv
from pathlib import Path
from collections import defaultdict

def load_results(json_file):
    """Load results from JSON file and return as dict keyed by (drug, side_effect)"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    results_dict = {}
    for record in data['detailed_results']:
        key = (record['drug'], record['side_effect'])
        # Convert YES/NO to 1/0
        prediction = 1 if record['predicted'] == 'YES' else 0
        results_dict[key] = {
            'true_label': record['true_label'],
            'predicted': prediction,
            'confidence': record.get('confidence', None)
        }

    return results_dict

def create_combined_csv(model_name, experiments_dir, output_path):
    """
    Create combined CSV for a specific model (llama3 or qwen)

    Args:
        model_name: 'llama3' or 'qwen'
        experiments_dir: Path to experiments directory
        output_path: Path for output CSV file
    """
    experiments_dir = Path(experiments_dir)

    # Load results from all architectures
    print(f"\nProcessing {model_name} results...")

    # File patterns for each architecture
    architectures = {
        'pure_llm': f'results_vllm_pure_llm_{model_name}_19520_*.json',
        'format_a': f'results_vllm_format_a_{model_name}_19520_*.json',
        'format_b': f'results_vllm_format_b_{model_name}_19520_*.json',
        'graphrag': f'results_vllm_graphrag_{model_name}_19520_*.json'
    }

    # Load all results
    all_results = {}
    for arch_name, pattern in architectures.items():
        files = list(experiments_dir.glob(pattern))
        if files:
            json_file = files[0]  # Take the first match
            print(f"  Loading {arch_name}: {json_file.name}")
            all_results[arch_name] = load_results(json_file)
        else:
            print(f"  Warning: No file found for {arch_name} with pattern {pattern}")
            all_results[arch_name] = {}

    # Get all unique drug-side_effect pairs
    all_keys = set()
    for results in all_results.values():
        all_keys.update(results.keys())

    all_keys = sorted(all_keys)  # Sort for consistent output

    print(f"  Total unique drug-side_effect pairs: {len(all_keys)}")

    # Create combined data
    combined_data = []
    for idx, (drug, side_effect) in enumerate(all_keys):
        row = {
            'index': idx,
            'drug': drug,
            'side_effect': side_effect,
        }

        # Get true label (should be same across all architectures)
        true_label = None
        for arch_results in all_results.values():
            if (drug, side_effect) in arch_results:
                true_label = arch_results[(drug, side_effect)]['true_label']
                break

        row['label'] = true_label

        # Add predictions from each architecture
        for arch_name in ['pure_llm', 'format_a', 'format_b', 'graphrag']:
            if (drug, side_effect) in all_results[arch_name]:
                row[f'output_{arch_name}'] = all_results[arch_name][(drug, side_effect)]['predicted']
            else:
                row[f'output_{arch_name}'] = None  # Missing data

        combined_data.append(row)

    # Write to CSV
    output_path = Path(output_path)
    fieldnames = ['index', 'drug', 'side_effect', 'label',
                  'output_pure_llm', 'output_format_a', 'output_format_b', 'output_graphrag']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_data)

    print(f"  Created: {output_path}")
    print(f"  Rows: {len(combined_data)}")

    return output_path

def main():
    experiments_dir = '/home/omeerdogan23/drugRAG/experiments'

    # Create combined CSV for llama3
    llama_output = create_combined_csv(
        'llama3',
        experiments_dir,
        '/home/omeerdogan23/drugRAG/experiments/results_combined_llama3_all_architectures.csv'
    )

    # Create combined CSV for qwen
    qwen_output = create_combined_csv(
        'qwen',
        experiments_dir,
        '/home/omeerdogan23/drugRAG/experiments/results_combined_qwen_all_architectures.csv'
    )

    print("\n" + "="*60)
    print("Combined CSV files created successfully!")
    print("="*60)

if __name__ == '__main__':
    main()

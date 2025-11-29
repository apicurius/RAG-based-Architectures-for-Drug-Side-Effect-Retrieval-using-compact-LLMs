#!/usr/bin/env python3
"""
Convert JSON experiment results to CSV files
"""
import json
import csv
from pathlib import Path

def convert_json_to_csv(json_file_path, output_dir=None):
    """
    Convert JSON results file to CSV format

    Args:
        json_file_path: Path to the JSON results file
        output_dir: Directory to save CSV files (default: same as input)
    """
    json_path = Path(json_file_path)

    # Read JSON file
    print(f"Processing: {json_path.name}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract detailed results
    detailed_results = data.get('detailed_results', [])

    if not detailed_results:
        print(f"Warning: No detailed_results found in {json_path.name}")
        return None

    # Create output filename
    # Extract key parts from filename: model type and architecture
    filename_parts = json_path.stem.split('_')

    # Build descriptive name
    if 'pure_llm' in json_path.stem:
        arch_type = 'pure_llm'
    elif 'graphrag' in json_path.stem:
        arch_type = 'graphrag'
    elif 'format_a' in json_path.stem:
        arch_type = 'format_a'
    elif 'format_b' in json_path.stem:
        arch_type = 'format_b'
    else:
        arch_type = 'unknown'

    # Detect model
    if 'llama3' in json_path.stem or 'llama' in json_path.stem:
        model = 'llama3'
    elif 'qwen' in json_path.stem:
        model = 'qwen'
    else:
        model = 'unknown'

    output_filename = f"results_{arch_type}_{model}_binary_eval.csv"

    # Set output directory
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_filename

    # Save to CSV using csv module
    if detailed_results:
        # Get all unique keys from all results
        fieldnames = list(detailed_results[0].keys())

        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)

        print(f"  -> Created: {output_path.name}")
        print(f"     Rows: {len(detailed_results)}")

    # Also save metrics summary
    metrics = data.get('metrics', {})
    if metrics:
        metrics_filename = f"metrics_{arch_type}_{model}_summary.csv"
        metrics_path = output_dir / metrics_filename

        # Write metrics to CSV (single row)
        with open(metrics_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)

        print(f"  -> Created: {metrics_filename}")

    return output_path

def main():
    # Define experiment directory
    experiments_dir = Path('/home/omeerdogan23/drugRAG/experiments')

    # Find all result JSON files
    result_files = sorted(experiments_dir.glob('results_vllm_*_19520_*.json'))

    print(f"Found {len(result_files)} result files to process\n")

    # Process each file
    for json_file in result_files:
        convert_json_to_csv(json_file)
        print()

    print("All files processed successfully!")

if __name__ == '__main__':
    main()

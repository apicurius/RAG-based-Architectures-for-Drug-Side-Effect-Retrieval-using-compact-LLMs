#!/usr/bin/env python3
"""
Reorganize reverse queries to include one TRUE and one FALSE example per side effect.
This creates a binary classification dataset for reverse lookup queries.
"""

import pandas as pd
import ast
import random

# Set random seed for reproducibility
random.seed(42)

# Load the data
print("Loading data...")
reverse_df = pd.read_csv('data/processed/reverse_queries.csv')
all_drugs_df = pd.read_csv('data/processed/drug_names.csv')

# Get list of all drugs
all_drugs = set(all_drugs_df['drug_name'].str.strip().str.lower())
print(f"Total drugs available: {len(all_drugs)}")

# Create new dataset with one TRUE and one FALSE example per side effect
new_data = []

print("Processing side effects...")
for idx, row in reverse_df.iterrows():
    side_effect = row['side_effect']
    query = row['query']

    # Parse the expected_drugs list (it's stored as a string representation of a list)
    try:
        expected_drugs_list = ast.literal_eval(row['expected_drugs'])
        expected_drugs = set([drug.strip().lower() for drug in expected_drugs_list])
    except:
        print(f"Warning: Could not parse expected_drugs for {side_effect}")
        continue

    # Get one TRUE example (drug that causes the side effect)
    if expected_drugs:
        true_drug = random.choice(list(expected_drugs))

        # Add TRUE example
        new_data.append({
            'side_effect': side_effect,
            'query': query,
            'drug': true_drug,
            'label': 'YES'  # Using YES/NO to match binary evaluation format
        })

    # Get one FALSE example (drug that does NOT cause the side effect)
    false_candidates = all_drugs - expected_drugs
    if false_candidates:
        false_drug = random.choice(list(false_candidates))

        # Add FALSE example
        new_data.append({
            'side_effect': side_effect,
            'query': query,
            'drug': false_drug,
            'label': 'NO'  # Using YES/NO to match binary evaluation format
        })

# Create new DataFrame
new_df = pd.DataFrame(new_data)

# Save to CSV
output_file = 'data/processed/reverse_queries_binary.csv'
new_df.to_csv(output_file, index=False)

print(f"\nCreated new dataset with {len(new_df)} examples")
print(f"YES examples: {len(new_df[new_df['label'] == 'YES'])}")
print(f"NO examples: {len(new_df[new_df['label'] == 'NO'])}")
print(f"Saved to: {output_file}")

# Display sample rows
print("\nSample rows from the new dataset:")
print(new_df.head(10))

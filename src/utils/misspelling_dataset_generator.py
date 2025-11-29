"""
Misspelling Dataset Generator

Creates paired datasets (correct vs misspelled) for testing robustness
of different architectures to spelling errors in drug names.
"""

import pandas as pd
import os
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MisspellingDatasetGenerator:
    """Generates correct and misspelled variants of evaluation dataset."""

    def __init__(
        self,
        misspelling_csv_path: str = "/home/omeerdogan23/drugRAG/experiments/misspellings.csv",
        evaluation_csv_path: str = "/home/omeerdogan23/drugRAG/data/processed/evaluation_dataset.csv",
        output_dir: str = "/home/omeerdogan23/drugRAG/data/processed"
    ):
        """
        Initialize the generator.

        Args:
            misspelling_csv_path: Path to CSV with correct/misspelled drug pairs
            evaluation_csv_path: Path to main evaluation dataset
            output_dir: Directory to save generated datasets
        """
        self.misspelling_csv_path = misspelling_csv_path
        self.evaluation_csv_path = evaluation_csv_path
        self.output_dir = output_dir

        # Load misspelling pairs
        self.misspelling_pairs = self._load_misspelling_pairs()
        logger.info(f"Loaded {len(self.misspelling_pairs)} misspelling pairs")

    def _load_misspelling_pairs(self) -> Dict[str, str]:
        """
        Load correct -> misspelled drug name mappings.

        Returns:
            Dictionary mapping correct drug names to misspelled versions
        """
        df = pd.read_csv(self.misspelling_csv_path)

        # Remove BOM if present and strip whitespace
        df.columns = df.columns.str.replace('\ufeff', '').str.strip()

        # Create mapping: correct -> misspelled
        pairs = {}
        for _, row in df.iterrows():
            correct = str(row['Original']).strip().lower()
            misspelled = str(row['Spelling error']).strip().lower()
            pairs[correct] = misspelled

        return pairs

    def generate_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate correct and misspelled datasets.

        Returns:
            Tuple of (correct_df, misspelled_df)
        """
        # Load evaluation dataset
        eval_df = pd.read_csv(self.evaluation_csv_path)
        logger.info(f"Loaded evaluation dataset with {len(eval_df)} queries")

        # Filter to only include drugs that have misspellings
        drugs_to_test = list(self.misspelling_pairs.keys())
        correct_df = eval_df[eval_df['drug'].str.lower().isin(drugs_to_test)].copy()

        logger.info(f"Filtered to {len(correct_df)} queries for {len(drugs_to_test)} drugs")

        # Log distribution
        drug_counts = correct_df['drug'].str.lower().value_counts()
        logger.info(f"Query distribution per drug:\n{drug_counts}")

        # Create misspelled version
        misspelled_df = correct_df.copy()

        # Replace drug names in both 'drug' column and 'query' column
        for correct_drug, misspelled_drug in self.misspelling_pairs.items():
            # Update drug column
            mask = misspelled_df['drug'].str.lower() == correct_drug
            misspelled_df.loc[mask, 'drug'] = misspelled_drug

            # Update query text (case-insensitive replacement)
            misspelled_df.loc[mask, 'query'] = misspelled_df.loc[mask, 'query'].str.replace(
                correct_drug,
                misspelled_drug,
                case=False,
                regex=False
            )

            logger.info(f"Replaced '{correct_drug}' -> '{misspelled_drug}' in {mask.sum()} queries")

        return correct_df, misspelled_df

    def save_datasets(self, correct_df: pd.DataFrame, misspelled_df: pd.DataFrame):
        """
        Save generated datasets to CSV files.

        Args:
            correct_df: Dataset with correct spellings
            misspelled_df: Dataset with misspelled drug names
        """
        os.makedirs(self.output_dir, exist_ok=True)

        correct_path = os.path.join(self.output_dir, "misspelling_experiment_correct.csv")
        misspelled_path = os.path.join(self.output_dir, "misspelling_experiment_misspelled.csv")

        correct_df.to_csv(correct_path, index=False)
        misspelled_df.to_csv(misspelled_path, index=False)

        logger.info(f"Saved correct dataset to: {correct_path}")
        logger.info(f"Saved misspelled dataset to: {misspelled_path}")

        # Verify datasets
        self._verify_datasets(correct_df, misspelled_df)

    def _verify_datasets(self, correct_df: pd.DataFrame, misspelled_df: pd.DataFrame):
        """Verify that datasets were created correctly."""
        assert len(correct_df) == len(misspelled_df), "Datasets must have same length"

        # Check that side effects and labels are identical
        assert (correct_df['side_effect'].values == misspelled_df['side_effect'].values).all(), \
            "Side effects must be identical"
        assert (correct_df['label'].values == misspelled_df['label'].values).all(), \
            "Labels must be identical"

        # Check that drug names are different
        drug_differences = (correct_df['drug'].str.lower() != misspelled_df['drug'].str.lower()).sum()
        assert drug_differences == len(correct_df), \
            f"All drug names should be different, but only {drug_differences}/{len(correct_df)} differ"

        logger.info("✓ Dataset verification passed")

        # Log sample comparison
        logger.info("\nSample comparison (first 3 queries):")
        for i in range(min(3, len(correct_df))):
            logger.info(f"\nQuery {i+1}:")
            logger.info(f"  Correct:    {correct_df.iloc[i]['query']}")
            logger.info(f"  Misspelled: {misspelled_df.iloc[i]['query']}")

    def generate_and_save(self):
        """Main method to generate and save datasets."""
        logger.info("Starting misspelling dataset generation...")

        correct_df, misspelled_df = self.generate_datasets()
        self.save_datasets(correct_df, misspelled_df)

        logger.info(f"\n{'='*60}")
        logger.info("Dataset generation complete!")
        logger.info(f"Total queries per dataset: {len(correct_df)}")
        logger.info(f"Number of drugs tested: {len(self.misspelling_pairs)}")
        logger.info(f"Label distribution: {correct_df['label'].value_counts().to_dict()}")
        logger.info(f"{'='*60}\n")

        return correct_df, misspelled_df


def main():
    """CLI entry point."""
    generator = MisspellingDatasetGenerator()
    generator.generate_and_save()


if __name__ == "__main__":
    main()

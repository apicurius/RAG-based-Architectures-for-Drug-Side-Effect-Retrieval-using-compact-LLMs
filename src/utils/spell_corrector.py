#!/usr/bin/env python3
"""
LLM-Based Spell Corrector for Drug Names

Provides spell correction using Qwen 7B or GPT-4 for misspelled drug names.
Designed for robustness experiments in DrugRAG system.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import Levenshtein
except ImportError:
    logging.warning("python-Levenshtein not installed. Using fallback edit distance calculation.")
    Levenshtein = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrectionResult:
    """Result of a spell correction attempt"""
    original: str
    corrected: str
    confidence: float
    changed: bool
    raw_response: str
    edit_distance: int

    def to_dict(self) -> Dict:
        return {
            'original': self.original,
            'corrected': self.corrected,
            'confidence': self.confidence,
            'changed': self.changed,
            'raw_response': self.raw_response,
            'edit_distance': self.edit_distance
        }


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.
    Uses python-Levenshtein if available, otherwise fallback implementation.
    """
    if Levenshtein:
        return Levenshtein.distance(s1.lower(), s2.lower())

    # Fallback implementation
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class LLMSpellCorrector:
    """LLM-based spell corrector for drug names"""

    # Simple prompt template
    SIMPLE_PROMPT_TEMPLATE = """You are a medical terminology expert specializing in pharmaceutical drug names.

TASK: Correct the spelling of the following drug name if it contains errors. If the drug name is already correct, return it unchanged.

DRUG NAME: {drug_name}

INSTRUCTIONS:
- If the drug name is misspelled, return ONLY the correct spelling
- If the drug name is already correct, return it unchanged
- Return ONLY the drug name, no explanations or additional text
- Do not add prefixes like "Corrected:" or "Answer:"
- If you cannot identify the drug with high confidence, return the original name unchanged

CORRECTED DRUG NAME:"""

    # Few-shot prompt template with examples
    FEWSHOT_PROMPT_TEMPLATE = """You are a medical terminology expert specializing in pharmaceutical drug names.

TASK: Correct the spelling of the following drug name if it contains errors.

EXAMPLES:
Drug: "floxetine" → "fluoxetine"
Drug: "ropirinole" → "ropinirole"
Drug: "grisefulvin" → "griseofulvin"
Drug: "metfarmin" → "metformin"
Drug: "aspirn" → "aspirin"
Drug: "ibuproffen" → "ibuprofen"
Drug: "azithromyacin" → "azithromycin"
Drug: "prednisone" → "prednisone" (already correct)

DRUG NAME: {drug_name}

Return ONLY the corrected drug name (or original if correct), no explanation:"""

    def __init__(
        self,
        use_fewshot: bool = True,
        temperature: float = 0.0,
        config_path: str = "/home/omeerdogan23/drugRAG/experiments/config.json"
    ):
        """
        Initialize spell corrector with Qwen 7B

        Args:
            use_fewshot: Use few-shot examples in prompt (improves accuracy)
            temperature: 0.0 for deterministic, 0.3 for multi-sample consensus
            config_path: Path to configuration file
        """
        self.use_fewshot = use_fewshot
        self.temperature = temperature
        self.config_path = config_path

        # Initialize Qwen 7B via vLLM
        from src.models.vllm_model import VLLMQwenModel
        self.llm = VLLMQwenModel(config_path)
        logger.info("SUCCESS: Spell corrector initialized with Qwen 7B (vLLM)")

        self.prompt_template = self.FEWSHOT_PROMPT_TEMPLATE if use_fewshot else self.SIMPLE_PROMPT_TEMPLATE

    def correct_single(self, drug_name: str) -> CorrectionResult:
        """
        Correct a single drug name

        Args:
            drug_name: The potentially misspelled drug name

        Returns:
            CorrectionResult with correction details
        """
        # Build prompt
        prompt = self._build_prompt(drug_name)

        # Generate correction
        try:
            raw_response = self.llm.generate_response(
                prompt,
                max_tokens=30,  # Drug names are short
                temperature=self.temperature
            )
        except Exception as e:
            logger.error(f"LLM generation failed for '{drug_name}': {e}")
            raw_response = drug_name  # Fallback to original

        # Parse correction
        corrected = self._parse_correction(raw_response)

        # Calculate metrics
        changed = drug_name.lower() != corrected.lower()
        edit_distance = levenshtein_distance(drug_name, corrected)
        confidence = self._calculate_confidence(drug_name, corrected, edit_distance)

        return CorrectionResult(
            original=drug_name,
            corrected=corrected,
            confidence=confidence,
            changed=changed,
            raw_response=raw_response,
            edit_distance=edit_distance
        )

    def correct_batch(self, drug_names: List[str]) -> List[CorrectionResult]:
        """
        Batch correction with parallel processing

        Args:
            drug_names: List of potentially misspelled drug names

        Returns:
            List of CorrectionResults
        """
        logger.info(f"🔤 Correcting {len(drug_names)} drug names with Qwen 7B...")

        # Build prompts for all drugs
        prompts = [self._build_prompt(drug) for drug in drug_names]

        # Batch generation using vLLM parallel processing
        try:
            if hasattr(self.llm, 'generate_batch'):
                raw_responses = self.llm.generate_batch(
                    prompts,
                    max_tokens=30,
                    temperature=self.temperature
                )
            else:
                # Fallback to individual generation
                logger.warning("Batch generation not supported, using individual calls")
                raw_responses = [
                    self.llm.generate_response(prompt, max_tokens=30, temperature=self.temperature)
                    for prompt in prompts
                ]
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            # Fallback to originals
            raw_responses = drug_names

        # Parse all corrections
        results = []
        for original, raw_response in zip(drug_names, raw_responses):
            corrected = self._parse_correction(raw_response)
            changed = original.lower() != corrected.lower()
            edit_distance = levenshtein_distance(original, corrected)
            confidence = self._calculate_confidence(original, corrected, edit_distance)

            results.append(CorrectionResult(
                original=original,
                corrected=corrected,
                confidence=confidence,
                changed=changed,
                raw_response=raw_response,
                edit_distance=edit_distance
            ))

        # Log summary
        changed_count = sum(1 for r in results if r.changed)
        avg_confidence = sum(r.confidence for r in results) / len(results)

        logger.info(f"SUCCESS: Correction complete: {changed_count}/{len(results)} changed, avg confidence: {avg_confidence:.2f}")

        return results

    def _build_prompt(self, drug_name: str) -> str:
        """Build correction prompt for a drug name"""
        return self.prompt_template.format(drug_name=drug_name)

    def _parse_correction(self, raw_response: str) -> str:
        """
        Extract corrected drug name from LLM response

        Handles various output formats:
        - Clean: "fluoxetine"
        - Prefixed: "Corrected: fluoxetine"
        - Explained: "The correct spelling is fluoxetine"
        """
        response = raw_response.strip()

        # Remove common prefixes
        prefixes = [
            "corrected:", "corrected drug name:", "answer:",
            "the correct spelling is", "the drug name is",
            "correct spelling:", "corrected name:"
        ]

        response_lower = response.lower()
        for prefix in prefixes:
            if response_lower.startswith(prefix):
                response = response[len(prefix):].strip()
                break

        # Remove quotes
        response = response.strip('"\'')

        # Take first word (drug names are typically one word)
        # But preserve hyphenated names
        words = response.split()
        if words:
            return words[0]

        # Fallback: return cleaned response
        return response

    def _calculate_confidence(self, original: str, corrected: str, edit_distance: int) -> float:
        """
        Estimate confidence in correction

        Confidence factors:
        - Unchanged: high confidence (0.95) - model thinks it's correct
        - Edit distance <= 2: medium-high confidence (0.75-0.85)
        - Edit distance 3-4: medium confidence (0.60-0.70)
        - Edit distance > 4: low confidence (0.30) - suspicious large change
        - Length ratio: penalize if length changes drastically
        """
        if original.lower() == corrected.lower():
            return 0.95  # Unchanged - high confidence it's already correct

        # Edit distance factor
        if edit_distance == 1:
            conf_edit = 0.85  # Single char change - very plausible
        elif edit_distance == 2:
            conf_edit = 0.75  # Two char changes - plausible
        elif edit_distance <= 4:
            conf_edit = 0.65  # Moderate change
        else:
            conf_edit = 0.30  # Large change - suspicious

        # Length ratio factor (penalize drastic length changes)
        length_ratio = len(corrected) / max(len(original), 1)
        if 0.7 <= length_ratio <= 1.3:
            conf_length = 0.9  # Similar length - good sign
        elif 0.5 <= length_ratio <= 1.5:
            conf_length = 0.7  # Moderate length change
        else:
            conf_length = 0.4  # Drastic length change - suspicious

        # Weighted average (edit distance more important)
        final_confidence = (conf_edit * 0.7) + (conf_length * 0.3)

        return final_confidence


def main():
    """CLI test of spell corrector"""
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM spell corrector (Qwen 7B)")
    parser.add_argument('--drug', type=str, help="Drug name to correct")
    parser.add_argument('--drugs', type=str, nargs='+', help="Multiple drug names")
    parser.add_argument('--no-fewshot', action='store_true', help="Disable few-shot examples")
    parser.add_argument('--temperature', type=float, default=0.0)

    args = parser.parse_args()

    # Initialize corrector with Qwen 7B
    corrector = LLMSpellCorrector(
        use_fewshot=not args.no_fewshot,
        temperature=args.temperature
    )

    # Test single or batch
    if args.drug:
        result = corrector.correct_single(args.drug)
        print(f"\nOriginal: {result.original}")
        print(f"Corrected: {result.corrected}")
        print(f"Changed: {result.changed}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Edit Distance: {result.edit_distance}")

    elif args.drugs:
        results = corrector.correct_batch(args.drugs)
        print(f"\n{'Original':<20} {'Corrected':<20} {'Changed':<10} {'Confidence':<12} {'Edit Dist'}")
        print("-" * 80)
        for r in results:
            print(f"{r.original:<20} {r.corrected:<20} {str(r.changed):<10} {r.confidence:<12.2f} {r.edit_distance}")

    else:
        # Test with example misspellings
        test_drugs = [
            "floxetine", "ropirinole", "grisefulvin",
            "lormetazerpam", "lercanipidine", "fluoxetine"  # Last one is correct
        ]

        print(f"\nTesting with {len(test_drugs)} example misspellings...")
        results = corrector.correct_batch(test_drugs)

        print(f"\n{'Original':<20} {'Corrected':<20} {'Changed':<10} {'Confidence':<12} {'Edit Dist'}")
        print("-" * 80)
        for r in results:
            print(f"{r.original:<20} {r.corrected:<20} {str(r.changed):<10} {r.confidence:<12.2f} {r.edit_distance}")


if __name__ == "__main__":
    main()

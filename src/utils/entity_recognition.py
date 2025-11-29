#!/usr/bin/env python3
"""
Entity Recognition Module - As shown in RAG diagram
Extracts drug and side effect entities from natural language queries
"""

import re
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityRecognizer:
    """
    Entity recognition module from RAG architecture diagram (section d)

    This implements the parallel "Entity Recognition" path shown in the diagram:
    User Query → Entity Recognition → [drug, side_effect]
    """

    def __init__(self):
        """Initialize entity recognizer with common patterns"""
        # Common patterns for drug-side effect queries
        self.patterns = [
            # Pattern 1: "Is [SE] an adverse effect of [DRUG]?"
            (
                r"Is\s+(.+?)\s+an?\s+adverse\s+(?:effect|reaction|event)\s+of\s+(.+?)[?\.]?",
                'se_drug'
            ),
            # Pattern 2: "Does [DRUG] cause [SE]?"
            (
                r"Does\s+(.+?)\s+cause\s+(.+?)[?\.]?",
                'drug_se'
            ),
            # Pattern 3: "Can [DRUG] lead to [SE]?"
            (
                r"Can\s+(.+?)\s+(?:lead\s+to|result\s+in)\s+(.+?)[?\.]?",
                'drug_se'
            ),
            # Pattern 4: "[DRUG] causes [SE]"
            (
                r"(.+?)\s+causes?\s+(.+?)[?\.]?",
                'drug_se'
            ),
            # Pattern 5: "[SE] is caused by [DRUG]"
            (
                r"(.+?)\s+is\s+caused\s+by\s+(.+?)[?\.]?",
                'se_drug'
            ),
            # Pattern 6: "Is [SE] a side effect of [DRUG]?"
            (
                r"Is\s+(.+?)\s+a\s+side\s+effect\s+of\s+(.+?)[?\.]?",
                'se_drug'
            ),
        ]

        logger.info("✅ Entity Recognition module initialized")

    def extract_entities(self, query: str) -> Dict[str, Optional[str]]:
        """
        Extract drug and side effect from natural language query

        Args:
            query: Natural language query (e.g., "Is nausea an adverse effect of aspirin?")

        Returns:
            Dict with 'drug' and 'side_effect' keys (None if not found)

        Examples:
            >>> recognizer.extract_entities("Is nausea an adverse effect of aspirin?")
            {'drug': 'aspirin', 'side_effect': 'nausea'}

            >>> recognizer.extract_entities("Does metformin cause headaches?")
            {'drug': 'metformin', 'side_effect': 'headaches'}
        """
        query = query.strip()

        # Try each pattern
        for pattern, order in self.patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entity1 = match.group(1).strip()
                entity2 = match.group(2).strip()

                # Clean up entities (remove articles, trailing punctuation)
                entity1 = self._clean_entity(entity1)
                entity2 = self._clean_entity(entity2)

                # Return based on pattern order
                if order == 'se_drug':
                    result = {'side_effect': entity1, 'drug': entity2}
                else:  # drug_se
                    result = {'drug': entity1, 'side_effect': entity2}

                logger.debug(f"Entity extraction: '{query}' → {result}")
                return result

        # No pattern matched
        logger.warning(f"Could not extract entities from: '{query}'")
        return {'drug': None, 'side_effect': None}

    def _clean_entity(self, entity: str) -> str:
        """Clean extracted entity (remove articles, punctuation)"""
        # Remove leading articles
        entity = re.sub(r'^(?:the|a|an)\s+', '', entity, flags=re.IGNORECASE)

        # Remove trailing punctuation (except hyphens and spaces)
        entity = re.sub(r'[.,!?;:]+$', '', entity)

        # Normalize whitespace
        entity = ' '.join(entity.split())

        return entity.strip()

    def validate_entities(self, drug: Optional[str], side_effect: Optional[str]) -> Tuple[bool, str]:
        """
        Validate extracted entities

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not drug or not side_effect:
            missing = []
            if not drug:
                missing.append('drug')
            if not side_effect:
                missing.append('side effect')

            return False, f"Could not extract {' and '.join(missing)} from query"

        # Check if entities are reasonable (not too short/long)
        if len(drug) < 2:
            return False, f"Drug name too short: '{drug}'"

        if len(side_effect) < 2:
            return False, f"Side effect name too short: '{side_effect}'"

        if len(drug) > 100:
            return False, f"Drug name too long: '{drug[:50]}...'"

        if len(side_effect) > 100:
            return False, f"Side effect name too long: '{side_effect[:50]}...'"

        return True, ""

    def query_from_natural_language(self, rag_system, natural_query: str) -> Dict:
        """
        Process natural language query using RAG system

        This implements the full two-path architecture from the diagram:
        1. Extract entities from query
        2. Pass to RAG system's query() method

        Args:
            rag_system: FormatARAG or FormatBRAG instance
            natural_query: Natural language query

        Returns:
            RAG query result or error dict
        """
        # Extract entities (parallel path in diagram)
        entities = self.extract_entities(natural_query)

        # Validate entities
        is_valid, error_msg = self.validate_entities(
            entities.get('drug'),
            entities.get('side_effect')
        )

        if not is_valid:
            return {
                'answer': 'ERROR',
                'confidence': 0.0,
                'error': error_msg,
                'query': natural_query
            }

        # Query RAG system with extracted entities
        logger.info(f"Natural language query: '{natural_query}'")
        logger.info(f"Extracted entities: drug='{entities['drug']}', side_effect='{entities['side_effect']}'")

        return rag_system.query(entities['drug'], entities['side_effect'])


# Convenience function for quick usage
def extract_entities_from_query(query: str) -> Dict[str, Optional[str]]:
    """
    Convenience function to extract entities without creating recognizer instance

    Args:
        query: Natural language query

    Returns:
        Dict with 'drug' and 'side_effect' keys
    """
    recognizer = EntityRecognizer()
    return recognizer.extract_entities(query)


if __name__ == "__main__":
    # Test entity recognition
    recognizer = EntityRecognizer()

    test_queries = [
        "Is nausea an adverse effect of aspirin?",
        "Does metformin cause headaches?",
        "Can ibuprofen lead to stomach pain?",
        "Is dizziness a side effect of lisinopril?",
        "Does acetaminophen cause liver damage?",
    ]

    print("Testing Entity Recognition Module\n")
    print("=" * 70)

    for query in test_queries:
        entities = recognizer.extract_entities(query)
        print(f"\nQuery: {query}")
        print(f"  → Drug: {entities['drug']}")
        print(f"  → Side Effect: {entities['side_effect']}")

        is_valid, error = recognizer.validate_entities(entities['drug'], entities['side_effect'])
        if is_valid:
            print(f"  ✅ Valid entities")
        else:
            print(f"  ❌ {error}")

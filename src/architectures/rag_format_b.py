#!/usr/bin/env python3
"""
Format B RAG Implementation - vLLM ONLY
Retrieval: Pinecone vector store (drug-effect pairs)
Reasoning: vLLM (Qwen or LLAMA3)
"""

import json
import logging
from typing import Dict, Any, List
from pinecone import Pinecone
import openai
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.embedding_client import create_embedding_client
from src.utils.token_manager import create_token_manager
from src.utils.binary_parser import parse_binary_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FormatBRAG:
    """Format B: Individual drug-side effect pairs with vLLM reasoning"""

    def __init__(self, config_path: str = "config.json", model: str = "qwen"):
        """
        Initialize Format B RAG with vLLM

        Args:
            config_path: Path to configuration file
            model: vLLM model ("qwen" or "llama3")

        Note: Embeds full query like notebook: "Is [SE] an adverse effect of [DRUG]?"
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Initialize Pinecone for retrieval
        self.pc = Pinecone(api_key=self.config['pinecone_api_key'])
        self.index = self.pc.Index(self.config['pinecone_index_name'])
        self.namespace = "drug-side-effects-formatB"

        # Initialize robust embedding client
        self.embedding_client = create_embedding_client(
            api_key=self.config['openai_api_key'],
            model="text-embedding-ada-002"
        )

        # Initialize vLLM for reasoning
        self.model = model
        if model == "qwen":
            from src.models.vllm_model import VLLMQwenModel
            self.llm = VLLMQwenModel(config_path)
        elif model == "llama3":
            from src.models.vllm_model import VLLMLLAMA3Model
            self.llm = VLLMLLAMA3Model(config_path)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'qwen' or 'llama3'")

        # Initialize token manager for context truncation
        self.token_manager = create_token_manager(model_type=model)

        logger.info(f"Format B RAG initialized with {model} via vLLM and token management")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using robust client that handles 400 errors"""
        return self.embedding_client.get_embedding(text)

    def _filter_by_entities(self, results, drug: str, side_effect: str) -> List[str]:
        """
        Filtering Module - As shown in RAG diagram (section d)

        This implements the critical filtering module that:
        1. Takes top-k results from Vector DB
        2. Takes [drug, side_effect] from entity recognition
        3. Keeps only pairs where BOTH entities match

        This is equivalent to the notebook's filter_rag() function for Format B.

        Args:
            results: Pinecone query results
            drug: Drug name entity
            side_effect: Side effect entity

        Returns:
            List of filtered pair strings
        """
        filtered_pairs = []

        for match in results.matches:
            if match.metadata and match.score > 0.5:  # Score threshold
                pair_drug = match.metadata.get('drug', '')
                pair_effect = match.metadata.get('side_effect', '')

                if pair_drug and pair_effect:
                    # CRITICAL: Check if BOTH entities match
                    # Drug must match AND side_effect must match
                    drug_matches = drug.lower() in pair_drug.lower()
                    side_effect_matches = side_effect.lower() in pair_effect.lower()

                    if drug_matches and side_effect_matches:
                        filtered_pairs.append(f"The drug {pair_drug} may cause {pair_effect} as an adverse effect.")
                        logger.debug(f"Filter PASS: {pair_drug} -> {pair_effect}")
                    else:
                        missing = []
                        if not drug_matches:
                            missing.append('drug')
                        if not side_effect_matches:
                            missing.append('side_effect')
                        logger.debug(f"Filter REJECT: {pair_drug} → {pair_effect} - missing {', '.join(missing)}")

        return filtered_pairs

    def query(self, drug: str, side_effect: str) -> Dict[str, Any]:
        """Binary query with vLLM reasoning"""
        # Retrieve from Pinecone
        # Embed full query (notebook-aligned): "Is [SE] an adverse effect of [DRUG]?"
        query_text = f"Is {side_effect} an adverse effect of {drug}?"
        logger.debug(f"Embedding full query: '{query_text}'")

        query_embedding = self.get_embedding(query_text)

        if not query_embedding:
            return {'answer': 'ERROR', 'confidence': 0.0}

        results = self.index.query(
            vector=query_embedding,
            top_k=10,  # Increased for better context
            namespace=self.namespace,
            include_metadata=True
        )

        # FILTERING MODULE - As shown in RAG diagram
        # Apply entity-based filtering (checks if BOTH drug AND side_effect match)
        filtered_pairs = self._filter_by_entities(results, drug, side_effect)

        # Create base prompt template for token counting
        base_prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the RAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {side_effect} an adverse effect of {drug}?

### RAG Results:

{{context}}"""

        # Generate negative statement if no filtered results (notebook-aligned)
        if not filtered_pairs:
            context = f"No, the side effect {side_effect} is not listed as an adverse effect, adverse reaction or side effect of the drug {drug}"
            logger.info(f"Format B: No filtered results for {drug}-{side_effect}, using negative statement")
        else:
            # Use token manager to intelligently truncate context
            context, pairs_included = self.token_manager.truncate_context_pairs(filtered_pairs, base_prompt)
            if pairs_included < len(filtered_pairs):
                logger.debug(f"Token limit: included {pairs_included}/{len(filtered_pairs)} filtered pairs for {drug}-{side_effect}")
            logger.info(f"Format B: {len(filtered_pairs)} pairs passed filtering for {drug}-{side_effect}")

        # Build final prompt with truncated context
        prompt = base_prompt.format(context=context)

        try:
            # Use temperature=0.1 for RAG deterministic outputs
            response = self.llm.generate_response(prompt, max_tokens=100, temperature=0.1)
            # Use standardized binary parser (notebook-compatible)
            answer = parse_binary_response(response)

            return {
                'answer': answer,
                'confidence': 0.9 if answer != 'UNKNOWN' else 0.3,
                'drug': drug,
                'side_effect': side_effect,
                'format': 'B',
                'model': f'vllm_{self.model}',
                'reasoning': response[:200],
                'evidence_count': len(filtered_pairs),
                'prompt': prompt,  # Full prompt for detailed logging
                'full_response': response  # Full response for detailed logging
            }

        except Exception as e:
            logger.error(f"vLLM reasoning error: {e}")
            return {
                'answer': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'format': 'B',
                'model': f'vllm_{self.model}'
            }

    def query_natural_language(self, natural_query: str) -> Dict[str, Any]:
        """
        Process natural language query using two-path architecture from diagram

        This implements the complete RAG architecture from the diagram:
        Path 1: Query → Embedding → Vector Search
        Path 2: Query → Entity Recognition → [drug, side_effect]
        Convergence: Vector results + Entities → Filtering Module → LLM

        Args:
            natural_query: Natural language query (e.g., "Is nausea an adverse effect of aspirin?")

        Returns:
            Same format as query() method

        Examples:
            >>> rag.query_natural_language("Is nausea an adverse effect of aspirin?")
            >>> rag.query_natural_language("Does metformin cause headaches?")
        """
        from src.utils.entity_recognition import EntityRecognizer

        # Entity Recognition Path (parallel to embedding in diagram)
        recognizer = EntityRecognizer()
        entities = recognizer.extract_entities(natural_query)

        # Validate entities
        is_valid, error_msg = recognizer.validate_entities(
            entities.get('drug'),
            entities.get('side_effect')
        )

        if not is_valid:
            return {
                'answer': 'ERROR',
                'confidence': 0.0,
                'error': error_msg,
                'query': natural_query,
                'format': 'B',
                'model': f'vllm_{self.model}'
            }

        logger.info(f"Natural language query: '{natural_query}'")
        logger.info(f"Extracted entities: drug='{entities['drug']}', side_effect='{entities['side_effect']}'")

        # Process using standard query path
        return self.query(entities['drug'], entities['side_effect'])

    def query_batch(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple queries with FULL BATCH OPTIMIZATION
        This provides dramatic speedup over individual query processing
        """
        if not queries:
            return []

        logger.info(f"FORMAT B BATCH PROCESSING: {len(queries)} queries with optimized embeddings + retrieval + vLLM")

        # Step 1: Batch embedding generation (MAJOR SPEEDUP)
        # Embed full queries (notebook-aligned)
        query_texts = [f"Is {q['side_effect']} an adverse effect of {q['drug']}?" for q in queries]
        logger.info(f"Generating {len(query_texts)} embeddings in batch (full query mode)...")

        embeddings = self.embedding_client.get_embeddings_batch(
            query_texts,
            batch_size=20  # Conservative batch size for large datasets
        )

        # Step 2: Concurrent Pinecone retrieval with progress tracking
        logger.info(f"Performing {len(embeddings)} Pinecone queries (concurrent)...")
        all_contexts = [None] * len(queries)  # Pre-allocate to maintain order

        def process_single_query(idx_query_embedding):
            """Process a single Pinecone query"""
            idx, query, embedding = idx_query_embedding

            if embedding is None:
                return idx, f"No specific pairs found for {query['drug']} and {query['side_effect']}"

            try:
                results = self.index.query(
                    vector=embedding,
                    top_k=10,  # Increased for better context
                    namespace=self.namespace,
                    include_metadata=True
                )

                # FILTERING MODULE - Apply entity-based filtering (BOTH drug AND side_effect)
                filtered_pairs = self._filter_by_entities(results, query['drug'], query['side_effect'])

                # Use token manager for context truncation
                base_prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the RAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {query['side_effect']} an adverse effect of {query['drug']}?

### RAG Results:

{{context}}"""

                # Generate negative statement if no filtered results
                if not filtered_pairs:
                    context = f"No, the side effect {query['side_effect']} is not listed as an adverse effect, adverse reaction or side effect of the drug {query['drug']}"
                else:
                    context, pairs_included = self.token_manager.truncate_context_pairs(filtered_pairs, base_prompt)

                return idx, context

            except Exception as e:
                logger.error(f"Pinecone query failed for {query['drug']}: {e}")
                return idx, f"No specific pairs found for {query['drug']} and {query['side_effect']}"

        # Process Pinecone queries in chunks to prevent memory buildup (SLURM 10GB limit)
        CHUNK_SIZE = 1000  # Process 1000 queries at a time
        query_data = [(i, query, embedding) for i, (query, embedding) in enumerate(zip(queries, embeddings))]

        logger.info(f"Processing Pinecone queries in chunks of {CHUNK_SIZE} (memory-safe mode)")

        with tqdm(total=len(queries), desc="Pinecone", unit="query", ncols=100) as pbar:
            for chunk_start in range(0, len(query_data), CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, len(query_data))
                chunk_data = query_data[chunk_start:chunk_end]

                # Use ThreadPoolExecutor for concurrent Pinecone queries
                max_workers = min(20, len(chunk_data))  # Optimized: 20 concurrent queries for faster retrieval

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit chunk queries
                    future_to_idx = {executor.submit(process_single_query, qd): qd[0] for qd in chunk_data}

                    # Collect results
                    for future in as_completed(future_to_idx):
                        try:
                            idx, context = future.result(timeout=30)
                            all_contexts[idx] = context
                            pbar.update(1)
                        except Exception as e:
                            idx = future_to_idx[future]
                            logger.error(f"Query {idx} failed: {e}")
                            all_contexts[idx] = f"No specific pairs found for {queries[idx]['drug']} and {queries[idx]['side_effect']}"
                            pbar.update(1)

        # Ensure all contexts are filled (safety check)
        for i, context in enumerate(all_contexts):
            if context is None:
                all_contexts[i] = f"No specific pairs found for {queries[i]['drug']} and {queries[i]['side_effect']}"

        # Step 3: Prepare prompts for batch vLLM processing
        logger.info(f"Preparing {len(queries)} prompts for batch vLLM inference...")
        prompts = []

        for query, context in zip(queries, all_contexts):
            prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the RAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {query['side_effect']} an adverse effect of {query['drug']}?

### RAG Results:

{context}"""
            prompts.append(prompt)

        # Step 4: Batch vLLM inference (OPTIMIZED)
        logger.info(f"Running batch vLLM inference...")
        try:
            # Use temperature=0.1 for RAG deterministic outputs
            responses = self.llm.generate_batch(prompts, max_tokens=100, temperature=0.1)
        except Exception as e:
            logger.error(f"Batch vLLM failed: {e}")
            # Fallback to individual processing
            return [self.query(q['drug'], q['side_effect']) for q in queries]

        # Step 5: Parse results
        results = []
        for query, response, context in zip(queries, responses, all_contexts):
            # Use standardized binary parser (notebook-compatible)
            answer = parse_binary_response(response)

            # Count evidence pairs (count "adverse effect" occurrences)
            evidence_count = context.count('adverse effect') if context else 0

            results.append({
                'answer': answer,
                'confidence': 0.9 if answer != 'UNKNOWN' else 0.3,
                'drug': query['drug'],
                'side_effect': query['side_effect'],
                'format': 'B',
                'model': f'vllm_{self.model}_batch_optimized',
                'reasoning': response[:200],
                'full_response': response,
                'evidence_count': evidence_count,
                'retrieval_context': context[:200]
            })

        success_rate = sum(1 for r in results if r['answer'] != 'UNKNOWN') / len(results) * 100
        logger.info(f"FORMAT B BATCH COMPLETE: {success_rate:.1f}% successful, {len(results)} total results")

        return results

    def reverse_query(self, side_effect: str, strategy: str = "chunked") -> Dict[str, Any]:
        """
        Reverse lookup: Find all drugs that cause a specific side effect

        Args:
            side_effect: The adverse effect to query
            strategy: Extraction strategy - "chunked" (default) or "monolithic"
                     - chunked: Process in chunks iteratively (DEFAULT - 98.37% recall, validated by Priority 1 evaluation)
                     - monolithic: Process all pairs at once (DEPRECATED - only 42.15% recall on large queries)

        Returns:
            Dict with 'drugs' list and metadata

        Note:
            As of November 2025, chunked strategy is the default based on Priority 1 evaluation results:
            - Chunked: 98.37% recall, 99.81% precision
            - Monolithic: 42.15% recall (fails catastrophically on queries >800 pairs)
            - See docs/PRIORITY_1_EVALUATION_RESULTS.md for full analysis

            Monolithic strategy is DEPRECATED for queries with >100 expected pairs due to
            "lost in the middle" attention degradation problem.
        """
        if strategy == "monolithic":
            logger.warning("Using monolithic strategy (DEPRECATED). Chunked strategy recommended for >100 pairs.")
            logger.warning("   Priority 1 evaluation: chunked 98.37% recall vs monolithic 42.15%")
            return self._reverse_query_monolithic(side_effect)
        else:
            return self._reverse_query_chunked(side_effect)

    def _reverse_query_monolithic(self, side_effect: str) -> Dict[str, Any]:
        """
        Original monolithic approach: Process all pairs at once through LLM

        Pros: Faster, simpler
        Cons: Lower recall for large result sets (>600 pairs) due to attention degradation
        """
        # Step 1: Generate embedding for side effect
        query_embedding = self.get_embedding(side_effect)

        if not query_embedding:
            return {'drugs': [], 'error': 'Failed to generate embedding'}

        # Step 2: Retrieve drug-side effect pairs from Pinecone with metadata filtering
        # IMPROVED: Use Pinecone's native metadata filter for exact matching
        # This retrieves ALL pairs with the exact side effect (not limited by similarity ranking)
        results = self.index.query(
            vector=query_embedding,
            top_k=10000,  # High limit to retrieve all matching pairs
            namespace=self.namespace,
            filter={'side_effect': {'$eq': side_effect.lower()}},  # Exact metadata match
            include_metadata=True
        )

        # Step 3: Extract unique drugs (all results already filtered by Pinecone)
        matching_drugs = set()
        context_pairs = []

        for match in results.matches:
            if match.metadata:
                pair_drug = match.metadata.get('drug', '')
                pair_effect = match.metadata.get('side_effect', '')

                if pair_drug and pair_effect:
                    matching_drugs.add(pair_drug)
                    context_pairs.append(f"The drug {pair_drug} may cause {pair_effect} as an adverse effect.")

        # Step 4: Use LLM to extract and verify drug list from retrieved pairs
        # SMART APPROACH: Use token manager to fit as many pairs as possible within context limit
        # Base prompt template for token counting
        base_prompt = f"""The RAG Results below show drug-side effect relationships.

### RAG Results:

{{context}}

### Question:
Based on these pairs, which drugs cause {side_effect}?

### Instructions:
- Extract all unique drug names that are paired with {side_effect}
- List only the drug names, separated by commas
- Do not include duplicates
- Base your answer strictly on the pairs shown above

Answer:"""

        # Use token manager to intelligently truncate context to fit within 8192 token limit
        # The token manager automatically reserves space for output
        if context_pairs:
            context, pairs_included = self.token_manager.truncate_context_pairs(
                context_pairs,
                base_prompt
            )
            if pairs_included < len(context_pairs):
                logger.warning(f"Format B: Context truncated to {pairs_included}/{len(context_pairs)} pairs for '{side_effect}' due to token limit")
            else:
                logger.info(f"Format B: Showing all {pairs_included} pairs to LLM for '{side_effect}'")
        else:
            context = f"No drug-side effect pairs found for {side_effect}"

        # Build final prompt with truncated context
        prompt = base_prompt.format(context=context)

        try:
            # Use temperature=0.1 for deterministic extraction
            # Dynamic max_tokens based on estimated drug count
            # With 32K context, we can use much larger output tokens
            # Estimate: ~3 tokens per drug name + separators
            # For 915 drugs (nausea), need ~3000 tokens
            max_output_tokens = max(2000, len(context_pairs) * 3)  # Minimum 2000, scale with pairs
            response = self.llm.generate_response(prompt, max_tokens=max_output_tokens, temperature=0.1)

            # Parse drug list from response
            drugs = self._parse_drug_list(response)

            logger.info(f"Format B Reverse Query: LLM extracted {len(drugs)} drugs from {len(context_pairs)} pairs for '{side_effect}'")

            return {
                'side_effect': side_effect,
                'drugs': drugs,
                'drug_count': len(drugs),
                'architecture': 'format_b',
                'model': f'vllm_{self.model}',
                'retrieved_pairs': len(context_pairs),
                'llm_response': response[:500]  # Truncate for logging
            }

        except Exception as e:
            logger.error(f"Reverse query error: {e}")
            return {
                'side_effect': side_effect,
                'drugs': [],
                'error': str(e),
                'architecture': 'format_b',
                'model': f'vllm_{self.model}'
            }

    def _parse_drug_list(self, response: str) -> List[str]:
        """
        Extract drug names from LLM response

        Handles various formats:
        - Comma-separated: "drug1, drug2, drug3"
        - Bulleted: "- drug1\n- drug2\n- drug3"
        - Numbered: "1. drug1\n2. drug2\n3. drug3"
        """
        import re

        # Remove common prefixes
        response = re.sub(r'^(Answer:|Drugs:|Drug list:)', '', response, flags=re.IGNORECASE).strip()

        # Try comma-separated first
        if ',' in response:
            drugs = [d.strip() for d in response.split(',')]
            return [d for d in drugs if d and len(d) > 1]

        # Try line-separated (bullets, numbers)
        lines = response.split('\n')
        drugs = []
        for line in lines:
            # Remove bullets, numbers, dashes
            cleaned = re.sub(r'^[\s\-\*\d\.\)]+', '', line).strip()
            if cleaned and len(cleaned) > 1 and not cleaned.startswith('#'):
                drugs.append(cleaned)

        return drugs

    def _reverse_query_chunked(self, side_effect: str, chunk_size: int = 200) -> Dict[str, Any]:
        """
        Chunked iterative extraction approach: Process pairs in smaller chunks

        This addresses the "lost in the middle" problem where LLMs show attention
        degradation on long contexts, even when they fit within the context window.

        Research shows LLMs perform better on shorter contexts. By chunking:
        - 142 pairs → 87% recall (monolithic)
        - 915 pairs → 49% recall (monolithic) WARNING:
        - 915 pairs → ~85-90% recall expected (chunked) SUCCESS:

        Pros: Higher recall for large result sets (>200 pairs)
        Cons: Slower (multiple LLM calls), higher token usage

        Args:
            side_effect: The adverse effect to query
            chunk_size: Number of pairs to process per chunk (default 200 for optimal recall)

        Returns:
            Dict with 'drugs' list and metadata
        """
        # Step 1: Generate embedding for side effect
        query_embedding = self.get_embedding(side_effect)

        if not query_embedding:
            return {'drugs': [], 'error': 'Failed to generate embedding'}

        # Step 2: Retrieve ALL matching pairs from Pinecone with metadata filtering
        results = self.index.query(
            vector=query_embedding,
            top_k=10000,  # High limit to retrieve all matching pairs
            namespace=self.namespace,
            filter={'side_effect': {'$eq': side_effect.lower()}},  # Exact metadata match
            include_metadata=True
        )

        # Step 3: Build context pairs
        context_pairs = []
        for match in results.matches:
            if match.metadata:
                pair_drug = match.metadata.get('drug', '')
                pair_effect = match.metadata.get('side_effect', '')
                if pair_drug and pair_effect:
                    context_pairs.append(f"The drug {pair_drug} may cause {pair_effect} as an adverse effect.")

        if not context_pairs:
            logger.info(f"Format B Chunked: No pairs found for '{side_effect}'")
            return {
                'side_effect': side_effect,
                'drugs': [],
                'drug_count': 0,
                'architecture': 'format_b_chunked',
                'model': f'vllm_{self.model}',
                'retrieved_pairs': 0,
                'chunks_processed': 0
            }

        # Step 4: Split pairs into chunks
        chunks = [context_pairs[i:i+chunk_size] for i in range(0, len(context_pairs), chunk_size)]
        total_chunks = len(chunks)

        logger.info(f"Format B Chunked: Processing {len(context_pairs)} pairs in {total_chunks} chunks of {chunk_size} for '{side_effect}'")

        # Step 5: Process each chunk independently and merge results
        all_drugs = set()

        for chunk_idx, chunk in enumerate(chunks, 1):
            # Build prompt for this chunk
            context = "\n".join(chunk)

            prompt = f"""The RAG Results below show drug-side effect relationships.

### RAG Results (Chunk {chunk_idx}/{total_chunks}):

{context}

### Question:
Based on these pairs, which drugs cause {side_effect}?

### Instructions:
- Extract all unique drug names that are paired with {side_effect}
- List only the drug names, separated by commas
- Do not include duplicates
- Base your answer strictly on the pairs shown above

Answer:"""

            try:
                # Use temperature=0.1 for deterministic extraction
                # For chunks of ~200 pairs, we need ~600-800 tokens for output
                max_output_tokens = max(1000, len(chunk) * 3)
                response = self.llm.generate_response(prompt, max_tokens=max_output_tokens, temperature=0.1)

                # Parse drug list from chunk response
                chunk_drugs = self._parse_drug_list(response)
                all_drugs.update(chunk_drugs)

                logger.debug(f"Format B Chunked: Chunk {chunk_idx}/{total_chunks} extracted {len(chunk_drugs)} drugs (total: {len(all_drugs)})")

            except Exception as e:
                logger.error(f"Format B Chunked: Error processing chunk {chunk_idx}/{total_chunks} for '{side_effect}': {e}")
                continue

        # Step 6: Return merged results
        final_drugs = sorted(list(all_drugs))

        logger.info(f"Format B Chunked: Extracted {len(final_drugs)} unique drugs from {len(context_pairs)} pairs across {total_chunks} chunks for '{side_effect}'")

        return {
            'side_effect': side_effect,
            'drugs': final_drugs,
            'drug_count': len(final_drugs),
            'architecture': 'format_b_chunked',
            'model': f'vllm_{self.model}',
            'retrieved_pairs': len(context_pairs),
            'chunks_processed': total_chunks,
            'chunk_size': chunk_size
        }
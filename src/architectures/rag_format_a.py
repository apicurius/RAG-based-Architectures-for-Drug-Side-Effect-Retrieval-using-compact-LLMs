#!/usr/bin/env python3
"""
Format A RAG Implementation - vLLM ONLY
Retrieval: Pinecone vector store
Reasoning: vLLM (Qwen or LLAMA3)
"""

import json
import logging
import numpy as np
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


class FormatARAG:
    """Format A: Drug -> [side effects] with vLLM reasoning"""

    def __init__(self, config_path: str = "config.json", model: str = "qwen"):
        """
        Initialize Format A RAG with vLLM

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
        self.namespace = "drug-side-effects-formatA"

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

        logger.info(f"✅ Format A RAG initialized with {model} via vLLM and token management")

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding using robust client that handles 400 errors"""
        return self.embedding_client.get_embedding(text)

    def _filter_by_entities(self, results, drug: str, side_effect: str) -> List[str]:
        """
        Filtering Module - As shown in RAG diagram (section d)

        This implements the critical filtering module that:
        1. Takes top-k results from Vector DB
        2. Takes [drug, side_effect] from entity recognition
        3. Keeps only documents where BOTH entities appear

        This is equivalent to the notebook's filter_rag() function.

        Args:
            results: Pinecone query results
            drug: Drug name entity
            side_effect: Side effect entity

        Returns:
            List of filtered document texts
        """
        filtered_documents = []

        for match in results.matches:
            if match.metadata and match.score > 0.5:  # Score threshold
                drug_name = match.metadata.get('drug', '')
                drug_text = match.metadata.get('text', '')

                if drug_name and drug_text:
                    # CRITICAL: Check if BOTH entities appear in text
                    # This is the filtering module logic from the diagram
                    drug_in_text = drug.lower() in drug_text.lower()
                    side_effect_in_text = side_effect.lower() in drug_text.lower()

                    if drug_in_text and side_effect_in_text:
                        filtered_documents.append(f"Drug: {drug_name}\n{drug_text}")
                        logger.debug(f"Filter PASS: {drug_name} - found both '{drug}' and '{side_effect}'")
                    else:
                        missing = []
                        if not drug_in_text:
                            missing.append('drug')
                        if not side_effect_in_text:
                            missing.append('side_effect')
                        logger.debug(f"Filter REJECT: {drug_name} - missing {', '.join(missing)}")

        return filtered_documents

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
        # Apply entity-based filtering (checks if BOTH drug AND side_effect appear)
        filtered_documents = self._filter_by_entities(results, drug, side_effect)

        # Create base prompt template for token counting
        base_prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the RAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {side_effect} an adverse effect of {drug}?

### RAG Results:

{{context}}"""

        # Generate negative statement if no filtered results (notebook-aligned)
        if not filtered_documents:
            context = f"No, the side effect {side_effect} is not listed as an adverse effect, adverse reaction or side effect of the drug {drug}"
            logger.info(f"Format A: No filtered results for {drug}-{side_effect}, using negative statement")
        else:
            # Use token manager to intelligently truncate context
            context, docs_included = self.token_manager.truncate_context_documents(filtered_documents, base_prompt)
            if docs_included < len(filtered_documents):
                logger.debug(f"Format A token limit: included {docs_included}/{len(filtered_documents)} filtered documents for {drug}-{side_effect}")
            logger.info(f"Format A: {len(filtered_documents)} documents passed filtering for {drug}-{side_effect}")

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
                'format': 'A',
                'model': f'vllm_{self.model}',
                'reasoning': response[:200],
                'prompt': prompt,  # Full prompt for detailed logging
                'full_response': response  # Full response for detailed logging
            }

        except Exception as e:
            logger.error(f"vLLM reasoning error: {e}")
            return {
                'answer': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'format': 'A',
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
                'format': 'A',
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

        logger.info(f"🚀 FORMAT A BATCH PROCESSING: {len(queries)} queries with optimized embeddings + retrieval + vLLM")

        # Step 1: Batch embedding generation (MAJOR SPEEDUP)
        # Embed full queries (notebook-aligned)
        query_texts = [f"Is {q['side_effect']} an adverse effect of {q['drug']}?" for q in queries]
        logger.info(f"📝 Generating {len(query_texts)} embeddings in batch (full query mode)...")

        embeddings = self.embedding_client.get_embeddings_batch(
            query_texts,
            batch_size=20  # Conservative batch size for large datasets
        )

        # Step 2: Concurrent Pinecone retrieval with progress tracking
        logger.info(f"🔍 Performing {len(embeddings)} Pinecone queries (concurrent)...")
        all_contexts = [None] * len(queries)  # Pre-allocate to maintain order

        def process_single_query(idx_query_embedding):
            """Process a single Pinecone query"""
            idx, query, embedding = idx_query_embedding

            if embedding is None:
                return idx, f"No data found for {query['drug']}"

            try:
                results = self.index.query(
                    vector=embedding,
                    top_k=10,  # Increased for better context
                    namespace=self.namespace,
                    include_metadata=True
                )

                # FILTERING MODULE - Apply entity-based filtering
                filtered_documents = self._filter_by_entities(results, query['drug'], query['side_effect'])

                # Use token manager for context truncation
                base_prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the RAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {query['side_effect']} an adverse effect of {query['drug']}?

### RAG Results:

{{context}}"""

                # Generate negative statement if no filtered results
                if not filtered_documents:
                    context = f"No, the side effect {query['side_effect']} is not listed as an adverse effect, adverse reaction or side effect of the drug {query['drug']}"
                else:
                    context, docs_included = self.token_manager.truncate_context_documents(filtered_documents, base_prompt)

                return idx, context

            except Exception as e:
                logger.error(f"Pinecone query failed for {query['drug']}: {e}")
                return idx, f"No data found for {query['drug']}"

        # Process Pinecone queries in chunks to prevent memory buildup (SLURM 10GB limit)
        CHUNK_SIZE = 1000  # Process 1000 queries at a time
        query_data = [(i, query, embedding) for i, (query, embedding) in enumerate(zip(queries, embeddings))]

        logger.info(f"🔍 Processing Pinecone queries in chunks of {CHUNK_SIZE} (memory-safe mode)")

        with tqdm(total=len(queries), desc="🔍 Pinecone", unit="query", ncols=100) as pbar:
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
                            all_contexts[idx] = f"No data found for {queries[idx]['drug']}"
                            pbar.update(1)

        # Ensure all contexts are filled (safety check)
        for i, context in enumerate(all_contexts):
            if context is None:
                all_contexts[i] = f"No data found for {queries[i]['drug']}"

        # Step 3: Prepare prompts for batch vLLM processing
        logger.info(f"🧠 Preparing {len(queries)} prompts for batch vLLM inference...")
        prompts = []

        for query, context in zip(queries, all_contexts):
            prompt = f"""You are asked to answer the following question with a single word: YES or NO. Base your answer strictly on the RAG Results provided below. After your YES or NO answer, briefly explain your reasoning using the information from the RAG Results. Do not infer or speculate beyond the provided information.

### Question:

Is {query['side_effect']} an adverse effect of {query['drug']}?

### RAG Results:

{context}"""
            prompts.append(prompt)

        # Step 4: Batch vLLM inference (OPTIMIZED)
        logger.info(f"⚡ Running batch vLLM inference...")
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

            results.append({
                'answer': answer,
                'confidence': 0.9 if answer != 'UNKNOWN' else 0.3,
                'drug': query['drug'],
                'side_effect': query['side_effect'],
                'format': 'A',
                'model': f'vllm_{self.model}_batch_optimized',
                'reasoning': response[:200],
                'full_response': response,
                'retrieval_context': context[:200]
            })

        success_rate = sum(1 for r in results if r['answer'] != 'UNKNOWN') / len(results) * 100
        logger.info(f"✅ FORMAT A BATCH COMPLETE: {success_rate:.1f}% successful, {len(results)} total results")

        return results

    def reverse_query(self, side_effect: str) -> Dict[str, Any]:
        """
        Reverse lookup: Find all drugs that cause a specific side effect

        Args:
            side_effect: The adverse effect to query

        Returns:
            Dict with 'drugs' list and metadata
        """
        # Step 1: Generate embedding for side effect
        query_embedding = self.get_embedding(side_effect)

        if not query_embedding:
            return {'drugs': [], 'error': 'Failed to generate embedding'}

        # Step 2: Retrieve drug documents from Pinecone
        # IMPROVED: Retrieve many more documents with lower threshold to combat embedding dilution
        results = self.index.query(
            vector=query_embedding,
            top_k=2000,  # Increased from 100 to 2000 (20x more documents)
            namespace=self.namespace,
            include_metadata=True
        )

        # Step 3: Filter and extract relevant drugs
        context_drugs = []
        context_texts = []

        for match in results.matches:
            if match.metadata and match.score > 0.3:  # Lowered from 0.6 to 0.3 for better recall
                drug_name = match.metadata.get('drug', '')
                drug_text = match.metadata.get('text', '')

                # Check if side effect mentioned in drug description
                if drug_name and drug_text and side_effect.lower() in drug_text.lower():
                    context_drugs.append(drug_name)
                    context_texts.append(f"Drug: {drug_name}\n{drug_text[:300]}")

        # Step 4: Use LLM to extract and verify drug list
        if context_texts:
            # Increased context limit with 8192 token model
            # Reserve ~500 tokens for prompt template + output
            # Use up to ~6000 tokens for context
            max_context_chars = 20000  # Increased from 10000 to support more documents
            context = "\n\n".join(context_texts[:100])  # Increased from 30 to 100 documents
            if len(context) > max_context_chars:
                context = context[:max_context_chars] + "\n... (truncated)"
        else:
            context = f"No drugs found associated with {side_effect}"

        prompt = f"""Based on the RAG Results below, identify all drugs that can cause {side_effect} as an adverse effect.

### RAG Results:

{context}

### Question:
Which drugs from the above results can cause {side_effect}?

### Instructions:
- Extract and list ONLY the drug names that are associated with {side_effect}
- Provide drug names separated by commas
- Only include drugs explicitly mentioned in the RAG Results
- Do not infer or speculate beyond the provided information

Answer:"""

        try:
            # Use temperature=0.1 for deterministic extraction
            response = self.llm.generate_response(prompt, max_tokens=500, temperature=0.1)

            # Parse drug list from response
            drugs = self._parse_drug_list(response)

            return {
                'side_effect': side_effect,
                'drugs': drugs,
                'drug_count': len(drugs),
                'architecture': 'format_a',
                'model': f'vllm_{self.model}',
                'retrieved_docs': len(context_drugs),
                'llm_response': response
            }

        except Exception as e:
            logger.error(f"Reverse query error: {e}")
            return {
                'side_effect': side_effect,
                'drugs': [],
                'error': str(e),
                'architecture': 'format_a',
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

        # Defensive check for None or empty responses
        if not response or response is None:
            logger.warning("Empty or None response received in _parse_drug_list")
            return []

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
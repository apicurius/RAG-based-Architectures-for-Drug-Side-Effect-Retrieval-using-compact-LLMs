#!/usr/bin/env python3
"""
vLLM-based Model Client - ULTRA FAST with 4 GPU Tensor Parallelism
This connects to the vLLM server running with tensor-parallel-size=4
"""

import json
import os
import logging
from typing import Dict, Any, List
import requests
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
from tqdm import tqdm
import sys

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.binary_parser import parse_binary_response

# Suppress HTTP request logs for clean terminal output
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMModel:
    """vLLM client for ultra-fast inference with 4-GPU tensor parallelism"""

    def __init__(self, config_path: str = "config.json", base_url: str = "http://localhost:8002", model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        """
        Initialize vLLM client

        Args:
            config_path: Path to config file
            base_url: vLLM server URL
            model_name: Model name for the server
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.base_url = base_url
        self.model_name = model_name

        # Test connection
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                logger.info(f"SUCCESS: Connected to vLLM server at {self.base_url}")
                logger.info(f"   Server is using 4 GPUs with tensor parallelism")
            else:
                logger.warning(f"vLLM server not responding. Start it with: ./start_vllm_server.sh")
        except:
            logger.warning(f"vLLM server not running. Start it with: ./start_vllm_server.sh")

    def generate_response(self, prompt: str, max_tokens: int = 150, temperature: float = None) -> str:
        """Generate response using vLLM server - ULTRA FAST

        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.1 for RAG deterministic, 0.3 for creative)
        """
        prompt_logger = logging.getLogger('prompts')
        response_logger = logging.getLogger('responses')

        # Log the prompt
        prompt_logger.debug(f"PROMPT (length: {len(prompt)} chars):\n{'-'*50}\n{prompt}\n{'-'*50}")

        try:
            # Use chat completions API for instruct models
            if "Instruct" in self.model_name or "instruct" in self.model_name.lower():
                # Use chat completions endpoint for instruct models
                request_data = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that answers questions accurately and concisely."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature is not None else 0.1,  # Default 0.1 for instruct models with RAG
                    "top_p": 0.9
                }

                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request_data
                )

                if response.status_code == 200:
                    result = response.json()
                    raw_response = result["choices"][0]["message"]["content"].strip()

                    # Log the response
                    response_logger.debug(f"RAW RESPONSE (status: 200):\n{'-'*50}\n{raw_response}\n{'-'*50}")

                    return raw_response
                else:
                    # Handle non-200 responses for Instruct models
                    error_msg = f"vLLM server error: {response.status_code}"
                    if response.text:
                        error_msg += f" - {response.text}"
                    logger.error(error_msg)
                    response_logger.debug(f"ERROR RESPONSE (status: {response.status_code}):\n{'-'*50}\n{response.text}\n{'-'*50}")
                    return ""
            else:
                # Use completions endpoint for non-instruct models
                request_data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature if temperature is not None else 0.3,  # Keep 0.3 for non-instruct models
                    "top_p": 0.9
                }

                response = requests.post(
                    f"{self.base_url}/v1/completions",
                    json=request_data
                )

                if response.status_code == 200:
                    result = response.json()
                    raw_response = result["choices"][0]["text"].strip()

                    # Log the response
                    response_logger.debug(f"RAW RESPONSE (status: 200):\n{'-'*50}\n{raw_response}\n{'-'*50}")

                    return raw_response
                else:
                    error_msg = f"vLLM server error: {response.status_code}"
                    if response.text:
                        error_msg += f" - {response.text}"
                    logger.error(error_msg)
                    response_logger.debug(f"ERROR RESPONSE (status: {response.status_code}):\n{'-'*50}\n{response.text}\n{'-'*50}")
                    return ""

        except Exception as e:
            error_msg = f"Error calling vLLM: {e}"
            logger.error(error_msg)
            response_logger.debug(f"EXCEPTION:\n{'-'*50}\n{error_msg}\n{'-'*50}")
            return ""

    def generate_batch(self, prompts: List[str], max_tokens: int = 150, temperature: float = None) -> List[str]:
        """
        Generate responses for multiple prompts using CHUNKED PARALLEL PROCESSING.
        Processes prompts in chunks of 2000 to prevent vLLM queue overflow and OOM.
        """
        if not prompts:
            return []

        logger.info(f"PROCESSING: vLLM CHUNKED PROCESSING: {len(prompts)} prompts in chunks of 2000")

        # Process in chunks to prevent vLLM queue overflow
        CHUNK_SIZE = 2000
        all_responses = []

        for chunk_start in range(0, len(prompts), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(prompts))
            chunk_prompts = prompts[chunk_start:chunk_end]
            chunk_num = (chunk_start // CHUNK_SIZE) + 1
            total_chunks = (len(prompts) + CHUNK_SIZE - 1) // CHUNK_SIZE

            logger.info(f"📦 Processing chunk {chunk_num}/{total_chunks} ({len(chunk_prompts)} prompts)")

            # Process this chunk with optimized concurrency
            # Increased from 1 to 20 after reducing context (16K) and with 128 max_num_seqs server config
            max_workers = min(len(chunk_prompts), 20)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all requests for this chunk
                future_to_index = {}
                for i, prompt in enumerate(chunk_prompts):
                    future = executor.submit(self._safe_generate_response, prompt, max_tokens, temperature)
                    future_to_index[future] = i

                # Collect results in order with progress bar
                chunk_responses = [None] * len(chunk_prompts)

                with tqdm(total=len(chunk_prompts), desc=f"🧠 Chunk {chunk_num}/{total_chunks}", unit="prompt", ncols=100) as pbar:
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        try:
                            response = future.result(timeout=30)
                            chunk_responses[index] = response
                        except Exception as e:
                            logger.error(f"Request {index} in chunk {chunk_num} failed: {e}")
                            chunk_responses[index] = ""

                        pbar.update(1)
                        # Update postfix with success rate
                        successful = len([r for r in chunk_responses if r is not None and r != ""])
                        pbar.set_postfix({'success': f'{successful}/{pbar.n}'})

                # Fill any None values (shouldn't happen but safety check)
                chunk_responses = [r if r is not None else "" for r in chunk_responses]
                all_responses.extend(chunk_responses)

            chunk_success = len([r for r in chunk_responses if r])
            logger.info(f"SUCCESS: Chunk {chunk_num}/{total_chunks} complete: {chunk_success}/{len(chunk_prompts)} successful")

        success_count = len([r for r in all_responses if r])
        logger.info(f"SUCCESS: ALL vLLM processing complete: {success_count}/{len(prompts)} successful ({success_count/len(prompts)*100:.1f}%)")
        return all_responses

    def _safe_generate_response(self, prompt: str, max_tokens: int, temperature: float = None) -> str:
        """
        Generate response with proper truncation and error handling for 400 errors.
        """
        # Truncate prompt if too long to avoid 400 errors
        max_prompt_length = 2000  # Conservative limit
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length] + "..."
            logger.debug(f"Truncated prompt to {max_prompt_length} characters")

        try:
            return self.generate_response(prompt, max_tokens, temperature)
        except Exception as e:
            error_str = str(e).lower()
            if "400" in error_str or "bad request" in error_str:
                # Try with shorter prompt
                logger.warning(f"400 error, retrying with shorter prompt")
                short_prompt = prompt[:1000] + "..." if len(prompt) > 1000 else prompt
                try:
                    return self.generate_response(short_prompt, max_tokens, temperature)
                except Exception:
                    logger.error(f"Failed even with shortened prompt")
                    return ""
            else:
                logger.error(f"Non-400 error: {e}")
                return ""

    def query(self, drug: str, side_effect: str) -> Dict[str, Any]:
        """Binary query using vLLM - ULTRA FAST"""
        analysis_logger = logging.getLogger('analysis')

        prompt = f"""You are asked to answer the following question with a single word: YES or NO. After your YES or NO answer, briefly explain your reasoning.

### Question:

Is {side_effect} an adverse effect of {drug}?"""

        analysis_logger.debug(f"QUERY START: {drug} -> {side_effect}")

        try:
            # Use temperature=0.3 for pure LLM (non-RAG) queries
            full_response = self.generate_response(prompt, max_tokens=100, temperature=0.3)

            analysis_logger.debug(f"RESPONSE ANALYSIS for {drug} -> {side_effect}:")
            analysis_logger.debug(f"  Raw response length: {len(full_response)}")

            # Use standardized binary parser (notebook-compatible)
            final_answer = parse_binary_response(full_response)

            confidence = 0.95 if final_answer in ['YES', 'NO'] else 0.3

            analysis_logger.debug(f"FINAL EXTRACTION: {final_answer} (confidence: {confidence})")

            result = {
                'answer': final_answer,
                'confidence': confidence,
                'drug': drug,
                'side_effect': side_effect,
                'model': 'vllm-qwen2.5-7b',
                'reasoning': full_response[:100],
                'prompt': prompt,  # Full prompt for detailed logging
                'full_response': full_response  # Keep full response for detailed logging
            }

            analysis_logger.debug(f"QUERY COMPLETE: {drug} -> {side_effect} = {final_answer}")
            return result

        except Exception as e:
            error_msg = f"Error querying vLLM: {e}"
            logger.error(error_msg)
            analysis_logger.debug(f"QUERY ERROR: {drug} -> {side_effect} - {error_msg}")
            return {
                'answer': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'model': 'vllm-qwen2.5-7b'
            }

    def query_batch(self, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch using TRUE BATCH PROCESSING

        Args:
            queries: List of dicts with 'drug' and 'side_effect' keys

        Returns:
            List of query results
        """
        if not queries:
            return []

        logger.info(f"PROCESSING: Processing {len(queries)} queries with optimized batch processing")

        # Prepare prompts with truncation to avoid 400 errors
        prompts = []
        for q in queries:
            # Shorter prompt to avoid truncation issues
            prompt = f"""Question: Is {q['side_effect']} an adverse effect of {q['drug']}?

Answer with YES or NO only.

FINAL ANSWER:"""
            prompts.append(prompt)

        # Generate all responses in batch with proper error handling
        try:
            # Use temperature=0.3 for pure LLM batch queries
            responses = self.generate_batch(prompts, max_tokens=50, temperature=0.3)  # Shorter responses
        except Exception as e:
            logger.error(f"Batch processing failed: {e}. Using fallback.")
            # Fallback to individual processing
            responses = []
            for prompt in prompts:
                try:
                    response = self.generate_response(prompt, max_tokens=50, temperature=0.3)
                    responses.append(response)
                except Exception:
                    responses.append("UNKNOWN")

        # Parse results with improved extraction
        results = []
        for i, (query, response) in enumerate(zip(queries, responses)):
            # Use standardized binary parser (notebook-compatible)
            final_answer = parse_binary_response(response)
            confidence = 0.95 if final_answer in ['YES', 'NO'] else 0.3

            results.append({
                'answer': final_answer,
                'confidence': confidence,
                'drug': query['drug'],
                'side_effect': query['side_effect'],
                'model': self.model_name,
                'reasoning': response[:50],  # Keep short reasoning
                'full_response': response
            })

        success_rate = sum(1 for r in results if r['answer'] != 'UNKNOWN') / len(results) * 100
        logger.info(f"SUCCESS: Batch processing complete: {success_rate:.1f}% successful extractions")

        return results

    def reverse_query(self, side_effect: str) -> Dict[str, Any]:
        """
        Reverse lookup: Find all drugs that cause a specific side effect
        Uses pure LLM parametric knowledge (no retrieval)

        Args:
            side_effect: The adverse effect to query

        Returns:
            Dict with 'drugs' list and metadata
        """
        prompt = f"""You are a medical knowledge expert. Answer the following question accurately based on your training data.

Question: Which drugs are known to cause {side_effect}?

Instructions:
- List all drugs that can cause {side_effect} as an adverse effect or side effect
- Provide only drug names separated by commas
- If you're not certain, only list drugs you're confident about
- Do not include explanations or descriptions, just the drug list

Answer:"""

        try:
            # Use temperature=0.3 for slightly higher recall
            response = self.generate_response(prompt, max_tokens=500, temperature=0.3)

            # Parse drug list from response
            drugs = self._parse_drug_list(response)

            return {
                'side_effect': side_effect,
                'drugs': drugs,
                'drug_count': len(drugs),
                'architecture': 'pure_llm',
                'model': self.model_name,
                'llm_response': response
            }

        except Exception as e:
            logger.error(f"Reverse query error: {e}")
            return {
                'side_effect': side_effect,
                'drugs': [],
                'error': str(e),
                'architecture': 'pure_llm',
                'model': self.model_name
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


# Model-specific classes
class VLLMQwenModel(VLLMModel):
    """vLLM client for Qwen model"""
    def __init__(self, config_path: str = "config.json"):
        super().__init__(
            config_path=config_path,
            base_url=os.getenv("VLLM_QWEN_BASE_URL", "http://localhost:8002"),
            model_name=os.getenv("VLLM_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        )


class VLLMLLAMA3Model(VLLMModel):
    """vLLM client for LLAMA3 model - NO AWS BEDROCK"""
    def __init__(self, config_path: str = "config.json"):
        super().__init__(
            config_path=config_path,
            base_url=os.getenv("VLLM_LLAMA3_BASE_URL", "http://localhost:8003"),
            model_name=os.getenv("VLLM_LLAMA3_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        )
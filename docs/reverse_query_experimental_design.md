# Reverse Query Experimental Design for DrugRAG

## Overview

This document outlines the experimental design for evaluating reverse lookup queries using the `data/processed/reverse_queries.csv` dataset. Reverse queries ask "Which drugs cause [side_effect]?" instead of the binary classification task "Does [drug] cause [side_effect]?".

---

## Dataset Structure

### File: `data/processed/reverse_queries.csv`

**Columns:**
- `side_effect`: The adverse effect being queried (e.g., "dizziness", "nausea")
- `query`: Natural language query (e.g., "Which drugs cause dizziness?")
- `expected_drugs`: List of drugs known to cause this side effect (ground truth)
- `drug_count`: Number of expected drugs
- `frequency`: Occurrence frequency in the original data

**Example Entry:**
```csv
side_effect,query,expected_drugs,drug_count,frequency
dizziness,"Which drugs cause dizziness?","['carnitine', 'leucovorin', 'pge2', ...]",988,2826
```

**Dataset Characteristics:**
- Query Type: Reverse lookup (side effect → drugs)
- Expected Output: List of drug names
- Evaluation: Set-based metrics (precision, recall, F1)
- Challenge: High cardinality (hundreds of drugs per side effect)

---

## Experimental Approaches

### 1. Pure LLM (Baseline)

**Description:**
Query the language model directly without retrieval, relying solely on parametric knowledge.

**Query Format:**
```
Which drugs are known to cause {side_effect}?

Please list all drugs that can cause this adverse effect.
```

**Prompt Template:**
```python
prompt = f"""You are a medical knowledge expert. Answer the following question accurately based on your training data.

Question: Which drugs are known to cause {side_effect}?

Instructions:
- List all drugs that can cause {side_effect} as an adverse effect or side effect
- Provide only drug names separated by commas
- If you're not certain, only list drugs you're confident about
- Do not include explanations or descriptions, just the drug list

Answer:"""
```

**Parameters:**
- **Temperature**: `0.3` (slightly higher than binary tasks to encourage recall)
- **Max Tokens**: `500` (sufficient for listing multiple drug names)
- **Top-p**: `0.9`
- **Model Options**: Qwen2.5-7B-Instruct or LLAMA3-8B-Instruct

**Expected Behavior:**
- Relies on pre-training knowledge
- May hallucinate drugs not in dataset
- Limited recall for rare side effects
- Fast inference (no retrieval overhead)

**Advantages:**
- No retrieval infrastructure needed
- Fast query processing
- Simple implementation

**Disadvantages:**
- Knowledge cutoff limitations
- Potential hallucinations
- Lower recall for comprehensive drug lists
- No source attribution

---

### 2. RAG Format A (Drug-Centric Documents)

**Description:**
Retrieve drug documents from Pinecone vector store and extract drugs associated with the target side effect.

**Retrieval Strategy:**

**Step 1: Generate Query Embedding**
```python
query_embedding = embedding_client.get_embedding(side_effect)
```

**Step 2: Retrieve Drug Documents**
```python
results = index.query(
    vector=query_embedding,
    top_k=100,  # Retrieve many documents for reverse lookup
    namespace="drug-side-effects-formatA",
    include_metadata=True
)
```

**Step 3: Filter and Extract Relevant Drugs**
```python
context_drugs = []
context_texts = []

for match in results.matches:
    if match.score > 0.6:  # Higher threshold for relevance
        drug_name = match.metadata.get('drug', '')
        drug_text = match.metadata.get('text', '')

        # Check if side effect mentioned in drug description
        if side_effect.lower() in drug_text.lower():
            context_drugs.append(drug_name)
            context_texts.append(f"Drug: {drug_name}\n{drug_text[:300]}")
```

**Prompt Template:**
```python
context = "\n\n".join(context_texts[:50])  # Limit context for token budget

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
```

**Parameters:**
- **Retrieval Top-K**: `100` documents
- **Similarity Threshold**: `0.6`
- **Temperature**: `0.1` (deterministic extraction)
- **Max Tokens**: `400`

**Expected Behavior:**
- Retrieves drug documents mentioning the side effect
- LLM extracts relevant drug names from context
- Better recall than pure LLM for drugs in the database
- Context window limits number of drugs processed

**Advantages:**
- Grounded in retrieved documents
- Better recall than pure LLM
- Source attribution possible

**Disadvantages:**
- Context window limits coverage
- Requires text search within documents
- May miss drugs with implicit associations

---

### 3. RAG Format B (Drug-Side Effect Pairs)

**Description:**
Retrieve explicit drug-side effect pairs from Pinecone and filter for matches.

**Retrieval Strategy:**

**Step 1: Generate Query Embedding**
```python
query_embedding = embedding_client.get_embedding(side_effect)
```

**Step 2: Retrieve Drug-Side Effect Pairs**
```python
results = index.query(
    vector=query_embedding,
    top_k=200,  # Retrieve many pairs for comprehensive coverage
    namespace="drug-side-effects-formatB",
    include_metadata=True
)
```

**Step 3: Filter Exact Matches**
```python
matching_drugs = set()
context_pairs = []

for match in results.matches:
    if match.score > 0.7:  # Higher threshold for exact matches
        pair_drug = match.metadata.get('drug', '')
        pair_effect = match.metadata.get('side_effect', '')

        # Exact or fuzzy match for side effect
        if (pair_effect.lower() == side_effect.lower() or
            side_effect.lower() in pair_effect.lower() or
            pair_effect.lower() in side_effect.lower()):
            matching_drugs.add(pair_drug)
            context_pairs.append(f"• {pair_drug} → {pair_effect}")
```

**Prompt Template:**
```python
context = "\n".join(context_pairs[:100])  # Show evidence pairs

prompt = f"""The RAG Results below show drug-side effect pairs in the format "Drug → Side Effect".

### RAG Results:

{context}

### Question:
Based on these pairs, which drugs cause {side_effect}?

### Instructions:
- Extract all unique drug names that are paired with {side_effect}
- List only the drug names, separated by commas
- Do not include duplicates
- Base your answer strictly on the pairs shown above

Answer:"""
```

**Parameters:**
- **Retrieval Top-K**: `200` pairs
- **Similarity Threshold**: `0.7`
- **Temperature**: `0.1` (deterministic)
- **Max Tokens**: `500`

**Expected Behavior:**
- Direct retrieval of drug-side effect associations
- High precision (explicit pairs)
- High recall (comprehensive pair coverage)
- Scalable to large result sets

**Advantages:**
- **Highest precision** among retrieval methods
- Direct evidence pairs
- Scalable retrieval
- Easy to verify results

**Disadvantages:**
- Depends on pair granularity in database
- Exact matching may miss semantic variants
- Context window may limit very large result sets

---

### 4. GraphRAG (Cypher Query)

**Description:**
Query Neo4j graph database directly using Cypher to traverse drug-side effect relationships.

**Cypher Query:**
```cypher
MATCH (drug)-[r:HAS_SIDE_EFFECT]->(effect)
WHERE effect.name = '{side_effect_normalized}'
RETURN drug.name AS drug_name
ORDER BY drug.name
```

**Implementation:**

**Step 1: Execute Cypher Query**
```python
def reverse_lookup_cypher(side_effect: str) -> List[str]:
    """
    Direct graph query for reverse lookup
    """
    # Normalize side effect name (lowercase, escaped)
    side_effect_escaped = escape_special_characters(side_effect.lower())

    # Cypher query for reverse lookup
    cypher = f"""
    MATCH (drug)-[r:HAS_SIDE_EFFECT]->(effect)
    WHERE effect.name = '{side_effect_escaped}'
    RETURN drug.name AS drug_name
    ORDER BY drug.name
    """

    with driver.session() as session:
        result = session.run(cypher)
        drugs = [record['drug_name'] for record in result]

    return drugs
```

**Step 2: Optional vLLM Verification**
```python
# Get drugs directly from graph
graph_drugs = reverse_lookup_cypher(side_effect)

# Optional: Use vLLM for ranking or verification
prompt = f"""The graph database query found the following drugs that cause {side_effect}:

Drugs: {', '.join(graph_drugs[:100])}

Question: Verify this list and provide confidence assessment.

Instructions:
- Review the drug list for obvious errors
- Indicate confidence level (HIGH/MEDIUM/LOW)
- Suggest any missing drugs you're confident about

Answer:"""
```

**Parameters:**
- **Graph Query**: Direct Cypher execution
- **Temperature** (if using LLM): `0.1`
- **Max Tokens**: `500`

**Alternative Queries:**

**Fuzzy Matching (for spelling variations):**
```cypher
MATCH (drug)-[r:HAS_SIDE_EFFECT]->(effect)
WHERE toLower(effect.name) CONTAINS toLower('{side_effect}')
RETURN drug.name AS drug_name, effect.name AS matched_effect
ORDER BY drug.name
```

**With Relationship Count:**
```cypher
MATCH (drug)-[r:HAS_SIDE_EFFECT]->(effect)
WHERE effect.name = '{side_effect_normalized}'
RETURN drug.name AS drug_name, count(r) AS relationship_count
ORDER BY relationship_count DESC, drug.name
```

**Expected Behavior:**
- **Exact traversal** of graph relationships
- **Complete coverage** of all drugs in database
- **Deterministic** results (no embedding similarity)
- **Fastest** query execution

**Advantages:**
- **Highest precision and recall** (ground truth from database)
- Deterministic results
- Fast query execution
- No token limits
- Direct relationship traversal
- Supports complex multi-hop queries

**Disadvantages:**
- Requires Neo4j infrastructure
- Limited to drugs in database
- No semantic similarity (exact matches only)
- Graph schema dependencies

---

## Evaluation Metrics

### Primary Metrics

#### 1. Precision
Measures accuracy of predicted drugs.

```python
precision = len(predicted ∩ expected) / len(predicted)
```

**Interpretation:**
- High precision = Few false positives (hallucinations)
- Low precision = Many irrelevant drugs predicted

#### 2. Recall
Measures coverage of expected drugs.

```python
recall = len(predicted ∩ expected) / len(expected)
```

**Interpretation:**
- High recall = Most expected drugs found
- Low recall = Missing many relevant drugs

#### 3. F1 Score
Harmonic mean of precision and recall.

```python
f1_score = 2 * (precision * recall) / (precision + recall)
```

**Interpretation:**
- Balances precision and recall
- Best metric for overall performance

#### 4. Exact Match
Strict evaluation: all drugs must match.

```python
exact_match = 1 if set(predicted) == set(expected) else 0
```

#### 5. Partial Match @K
Success if at least K drugs match.

```python
partial_match_k = 1 if len(predicted ∩ expected) >= K else 0
```

### Secondary Metrics

#### 6. Coverage
Percentage of expected drugs retrieved.

```python
coverage = len(predicted ∩ expected) / len(expected) * 100
```

#### 7. Hallucination Rate
Percentage of predicted drugs not in ground truth.

```python
hallucination_rate = len(predicted - expected) / len(predicted) * 100
```

#### 8. Mean Reciprocal Rank (MRR)
If predictions are ranked.

```python
# Find position of first correct drug
first_correct_position = next(
    (i+1 for i, drug in enumerate(predicted) if drug in expected),
    None
)
mrr = 1 / first_correct_position if first_correct_position else 0
```

### Performance Metrics

#### 9. Query Latency
Average time per query (seconds).

```python
avg_latency = total_time / num_queries
```

#### 10. Throughput
Queries processed per second.

```python
throughput = num_queries / total_time
```

---

## Comparison Table

| **Approach** | **Query Complexity** | **Expected Precision** | **Expected Recall** | **Speed** | **Hallucination Risk** | **Scalability** |
|--------------|---------------------|------------------------|---------------------|-----------|------------------------|-----------------|
| **Pure LLM** | Low | Medium-Low (50-70%) | Low-Medium (40-60%) | **Fastest** | **High** | Excellent |
| **RAG Format A** | Medium | Medium-High (65-80%) | Medium (50-70%) | Medium | Medium | Good |
| **RAG Format B** | Medium-High | High (75-90%) | High (70-85%) | Medium-Slow | **Low** | Good |
| **GraphRAG (Cypher)** | Low | **Highest (95-100%)** | **Highest (90-100%)** | **Fast** | **None** | **Excellent** |

### Key Insights:

1. **GraphRAG is optimal** for reverse queries due to direct relationship traversal
2. **RAG Format B** provides best retrieval-based approach with explicit pairs
3. **Pure LLM** suitable only for quick prototyping or when retrieval unavailable
4. **RAG Format A** middle ground between simplicity and accuracy

---

## Implementation Recommendations

### Batch Processing

For large-scale evaluation of the full dataset:

**Pure LLM:**
```python
# Use vLLM batch inference
prompts = [create_pure_llm_prompt(row['side_effect']) for _, row in df.iterrows()]
responses = vllm_model.generate_batch(prompts, max_tokens=500, temperature=0.3)
```

**RAG Format A:**
```python
# Batch embeddings + concurrent retrieval + batch LLM
embeddings = embedding_client.get_embeddings_batch(side_effects, batch_size=50)
# Concurrent Pinecone queries with ThreadPoolExecutor
contexts = retrieve_contexts_concurrent(embeddings, max_workers=10)
# Batch LLM inference
responses = vllm_model.generate_batch(prompts, max_tokens=400, temperature=0.1)
```

**RAG Format B:**
```python
# Batch embeddings + concurrent retrieval + batch LLM
embeddings = embedding_client.get_embeddings_batch(side_effects, batch_size=50)
# Concurrent pair retrieval
pairs = retrieve_pairs_concurrent(embeddings, max_workers=10)
# Batch LLM inference
responses = vllm_model.generate_batch(prompts, max_tokens=500, temperature=0.1)
```

**GraphRAG:**
```python
# Concurrent Cypher queries (fastest approach)
def process_query(side_effect):
    return reverse_lookup_cypher(side_effect)

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_query, side_effects))
```

### Response Parsing

All approaches require parsing LLM output to extract drug lists:

```python
def parse_drug_list(response: str) -> List[str]:
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
        if cleaned and len(cleaned) > 1:
            drugs.append(cleaned)

    return drugs
```

### Drug Name Normalization

Normalize drug names for fair comparison:

```python
def normalize_drug_name(drug: str) -> str:
    """
    Normalize drug names for comparison
    """
    # Lowercase
    drug = drug.lower().strip()

    # Remove common suffixes
    drug = re.sub(r'\s+(hydrochloride|sulfate|sodium|injection|tablet)$', '', drug)

    # Remove parentheses content
    drug = re.sub(r'\s*\([^)]*\)', '', drug)

    # Remove extra spaces
    drug = re.sub(r'\s+', ' ', drug).strip()

    return drug
```

---

## Sample Evaluation Script Structure

```python
#!/usr/bin/env python3
"""
Reverse Query Evaluation Script
"""

class ReverseQueryEvaluator:
    def __init__(self, config_path: str):
        self.config_path = config_path

    def load_dataset(self):
        """Load reverse_queries.csv"""
        df = pd.read_csv("data/processed/reverse_queries.csv")
        return df

    def evaluate_pure_llm(self, queries: List[Dict]) -> List[Dict]:
        """Evaluate Pure LLM approach"""
        pass

    def evaluate_rag_format_a(self, queries: List[Dict]) -> List[Dict]:
        """Evaluate RAG Format A approach"""
        pass

    def evaluate_rag_format_b(self, queries: List[Dict]) -> List[Dict]:
        """Evaluate RAG Format B approach"""
        pass

    def evaluate_graphrag(self, queries: List[Dict]) -> List[Dict]:
        """Evaluate GraphRAG approach"""
        pass

    def calculate_metrics(self, predicted: List[str], expected: List[str]) -> Dict:
        """Calculate all evaluation metrics"""
        predicted_set = set(normalize_drug_name(d) for d in predicted)
        expected_set = set(normalize_drug_name(d) for d in expected)

        intersection = predicted_set & expected_set

        precision = len(intersection) / len(predicted_set) if predicted_set else 0
        recall = len(intersection) / len(expected_set) if expected_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'coverage': recall * 100,
            'hallucination_rate': len(predicted_set - expected_set) / len(predicted_set) * 100 if predicted_set else 0,
            'exact_match': 1 if predicted_set == expected_set else 0
        }
```

---

## Expected Results

### Hypothesis

**Ranking by F1 Score:**
1. GraphRAG (Cypher): **F1 ≈ 0.92-0.98** (near-perfect for drugs in graph)
2. RAG Format B: **F1 ≈ 0.75-0.85** (high precision/recall from pairs)
3. RAG Format A: **F1 ≈ 0.60-0.75** (good but context-limited)
4. Pure LLM: **F1 ≈ 0.45-0.65** (hallucinations + knowledge gaps)

**Ranking by Speed:**
1. GraphRAG (Cypher): **~20-50 queries/sec** (direct DB query)
2. Pure LLM: **~10-30 queries/sec** (no retrieval overhead)
3. RAG Format A: **~5-15 queries/sec** (retrieval + LLM)
4. RAG Format B: **~5-15 queries/sec** (retrieval + LLM)

---

## Next Steps

1. **Implement reverse lookup methods** in each architecture class:
   - `FormatARAG.reverse_query(side_effect)`
   - `FormatBRAG.reverse_query(side_effect)`
   - `GraphRAG.reverse_query(side_effect)`
   - `VLLMModel.reverse_query(side_effect)`

2. **Create evaluation script**: `experiments/evaluate_reverse_queries.py`

3. **Run experiments** on subset first (e.g., 50 side effects)

4. **Analyze results** and tune parameters

5. **Scale to full dataset** with batch processing

6. **Generate comparison report** with visualizations

---

## References

- Dataset: `data/processed/reverse_queries.csv`
- Binary evaluation: `experiments/evaluate_reverse_binary.py`
- Architectures: `src/architectures/`
- Metrics: `src/evaluation/metrics.py`

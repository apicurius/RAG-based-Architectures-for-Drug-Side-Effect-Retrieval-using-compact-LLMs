# Binary Query Execution Examples

**Supplementary Figure**: Detailed execution examples for binary queries across all architectures

**Example Query**: "Does octreotide cause dizziness?"
**Drug**: octreotide
**Side Effect**: dizziness
**Ground Truth**: YES (octreotide is associated with dizziness)

---

## 1. Pure LLM Approach

### Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Query Input                                          │
│ "Does octreotide cause dizziness?"                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Direct LLM Inference                                 │
│ - No database access                                    │
│ - No retrieved context                                  │
│ - Relies solely on model's training knowledge           │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 3. LLM Response                                         │
│ "Based on my knowledge, octreotide may cause           │
│  dizziness as a side effect..."                         │
│                                                          │
│ Answer: YES                                             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Final Output                                         │
│ {                                                        │
│   "prediction": "YES",                                  │
│   "ground_truth": "YES",                                │
│   "correct": true,                                      │
│   "method": "pure_llm"                                  │
│ }                                                        │
│                                                          │
│ Metrics:                                                │
│ - Accuracy: 62.90%                                      │
│ - Precision: 77.6%                                      │
│ - Recall: 36.3%                                         │
│ - F1 Score: 0.494                                       │
└─────────────────────────────────────────────────────────┘
```

### Characteristics
- **Strengths**: Fast, no database required
- **Weaknesses**: Low recall (misses 63.7% of true associations), hallucination risk
- **Use Case**: Quick preliminary checks, not production-ready

---

## 2. Format A RAG (Drug → Effects)

### Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Query Input                                          │
│ "Does octreotide cause dizziness?"                      │
│                                                          │
│ Extracted:                                              │
│ - Drug: "octreotide"                                    │
│ - Side Effect: "dizziness"                              │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Pinecone Vector Database Retrieval                   │
│ Query: "octreotide side effects"                        │
│ Index: drug-effects-format-a                            │
│                                                          │
│ Retrieved Document (Top-1):                             │
│ {                                                        │
│   "drug": "octreotide",                                 │
│   "side_effects": [                                     │
│     "nausea", "diarrhea", "abdominal pain",            │
│     "dizziness", "headache", "fatigue",                │
│     "hyperglycemia", "bradycardia"                      │
│   ],                                                    │
│   "similarity_score": 0.98                              │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Context-Augmented LLM Query                          │
│ Prompt:                                                 │
│ "Based on the following information:                    │
│  Drug: octreotide                                       │
│  Known side effects: nausea, diarrhea, abdominal       │
│  pain, dizziness, headache, fatigue...                 │
│                                                          │
│  Question: Does octreotide cause dizziness?            │
│  Answer with YES or NO."                                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 4. LLM Response with Context                            │
│ "YES, based on the provided information, dizziness     │
│  is listed as a known side effect of octreotide."      │
│                                                          │
│ Answer: YES                                             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Final Output                                         │
│ {                                                        │
│   "prediction": "YES",                                  │
│   "ground_truth": "YES",                                │
│   "correct": true,                                      │
│   "method": "format_a_rag",                             │
│   "retrieved_effects": 8,                               │
│   "effect_found": true                                  │
│ }                                                        │
│                                                          │
│ Metrics:                                                │
│ - Accuracy: 86.67%                                      │
│ - Precision: 91.9%                                      │
│ - Recall: 80.5%                                         │
│ - F1 Score: 0.858                                       │
└─────────────────────────────────────────────────────────┘
```

### Characteristics
- **Strengths**: Good accuracy, contextual grounding, scalable with Pinecone
- **Weaknesses**: May miss side effect if not in top-k retrieved documents
- **Use Case**: General drug safety queries

---

## 3. Format B RAG (Drug-Effect Pairs)

### Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Query Input                                          │
│ "Does octreotide cause dizziness?"                      │
│                                                          │
│ Extracted:                                              │
│ - Drug: "octreotide"                                    │
│ - Side Effect: "dizziness"                              │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Pinecone Targeted Vector Retrieval                   │
│ Query: "octreotide dizziness association"               │
│ Index: drug-effect-pairs-format-b                       │
│                                                          │
│ Retrieved Documents (Top-3):                            │
│                                                          │
│ Doc 1 (Score: 0.95):                                   │
│ {                                                        │
│   "drug": "octreotide",                                 │
│   "side_effect": "dizziness",                           │
│   "relationship": "causes",                             │
│   "evidence": "Clinical trials reported dizziness      │
│                in 12% of patients"                      │
│ }                                                        │
│                                                          │
│ Doc 2 (Score: 0.89):                                   │
│ {                                                        │
│   "drug": "octreotide",                                 │
│   "side_effect": "vertigo",                             │
│   "relationship": "causes"                              │
│ }                                                        │
│                                                          │
│ Doc 3 (Score: 0.85):                                   │
│ {                                                        │
│   "drug": "lanreotide",                                 │
│   "side_effect": "dizziness",                           │
│   "relationship": "causes"                              │
│ }                                                        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Context-Augmented LLM Query                          │
│ Prompt:                                                 │
│ "Based on the following drug-side effect pairs:         │
│  1. octreotide → dizziness (high confidence)           │
│  2. octreotide → vertigo (related effect)              │
│  3. lanreotide → dizziness (similar drug)              │
│                                                          │
│  Question: Does octreotide cause dizziness?            │
│  Answer with YES or NO."                                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 4. LLM Response with Strong Evidence                    │
│ "YES, there is direct evidence that octreotide         │
│  causes dizziness. Clinical data confirms this         │
│  association."                                          │
│                                                          │
│ Answer: YES                                             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 5. Final Output                                         │
│ {                                                        │
│   "prediction": "YES",                                  │
│   "ground_truth": "YES",                                │
│   "correct": true,                                      │
│   "method": "format_b_rag",                             │
│   "direct_match": true,                                 │
│   "match_score": 0.95                                   │
│ }                                                        │
│                                                          │
│ Metrics:                                                │
│ - Accuracy: 96.50%                                      │
│ - Precision: 93.6%                                      │
│ - Recall: 99.9%                                         │
│ - F1 Score: 0.967                                       │
└─────────────────────────────────────────────────────────┘
```

### Characteristics
- **Strengths**: Excellent accuracy, direct pair matching, near-perfect recall, efficient Pinecone queries
- **Weaknesses**: Requires more granular data storage in Pinecone
- **Use Case**: Production drug safety systems

---

## 4. GraphRAG (Neo4j)

### Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Query Input                                          │
│ "Does octreotide cause dizziness?"                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Cypher Query Generation                              │
│ MATCH (d:Drug)-[r:CAUSES]->(s:SideEffect)              │
│ WHERE toLower(d.name) = $drug                           │
│   AND toLower(s.name) = $sideEffect                     │
│ RETURN DISTINCT d.name AS drug                          │
│                                                          │
│ Parameters:                                             │
│ - drug: "octreotide"                                    │
│ - sideEffect: "dizziness"                               │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Neo4j Execution                                      │
│ - Direct graph traversal                                │
│ - No embeddings needed                                  │
│ - No LLM needed                                         │
│                                                          │
│ Result: 1 unique drug found                             │
│                                                          │
│ Graph Visualization:                                    │
│                                                          │
│     (Drug)                                              │
│   octreotide                                            │
│        │                                                 │
│        │ [CAUSES]                                        │
│        │                                                 │
│        ↓                                                 │
│  (SideEffect)                                           │
│    dizziness                                            │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Final Output                                         │
│ {                                                        │
│   "side_effect": "dizziness",                           │
│   "drugs": ["octreotide"],                              │
│   "drug_count": 1                                       │
│ }                                                        │
│                                                          │
│ Binary Answer: YES (drug_count > 0)                     │
│                                                          │
│ Metrics:                                                │
│ - Accuracy: 100.00%                                     │
│ - Precision: 100.0%                                     │
│ - Recall: 100.0%                                        │
│ - F1 Score: 1.000                                       │
│ - Query Time: <10ms                                     │
└─────────────────────────────────────────────────────────┘
```

### Characteristics
- **Strengths**: Perfect accuracy, deterministic, fast, no hallucination, no vector embeddings needed
- **Weaknesses**: Requires Neo4j infrastructure, graph data preparation
- **Use Case**: Critical applications requiring 100% accuracy

---

## Performance Comparison Summary

### Qwen2.5-7B-Instruct Results

| Method | Accuracy | Precision | Recall | F1 Score | Speed | Infrastructure |
|--------|----------|-----------|--------|----------|-------|----------------|
| **Pure LLM** | 62.90% | 77.6% | 36.3% | 0.494 | Very Fast | Minimal |
| **Format A** | 86.67% | 91.9% | 80.5% | 0.858 | Fast | Pinecone |
| **Format B** | 96.50% | 93.6% | 99.9% | 0.967 | Fast | Pinecone |
| **GraphRAG** | 100.00% | 100.0% | 100.0% | 1.000 | Very Fast | Neo4j |

### Llama-3.1-8B-Instruct Results

| Method | Accuracy | Precision | Recall | F1 Score | Speed | Infrastructure |
|--------|----------|-----------|--------|----------|-------|----------------|
| **Pure LLM** | 63.21% | 72.8% | 42.2% | 0.534 | Very Fast | Minimal |
| **Format A** | 84.54% | 98.7% | 70.0% | 0.819 | Fast | Pinecone |
| **Format B** | 95.86% | 92.4% | 99.9% | 0.960 | Fast | Pinecone |
| **GraphRAG** | 99.96% | 100.0% | 100.0% | 1.000 | Very Fast | Neo4j |

### Model Comparison

| Architecture | Qwen2.5-7B | Llama-3.1-8B | Difference |
|--------------|------------|--------------|------------|
| **Pure LLM** | 62.90% | 63.21% | +0.31% Llama |
| **Format A** | 86.67% | 84.54% | +2.13% Qwen |
| **Format B** | 96.50% | 95.86% | +0.64% Qwen |
| **GraphRAG** | 100.00% | 99.96% | +0.04% Qwen |

**Key Finding**: Both models perform similarly, with differences under 2.5% across all architectures

## Key Insights

### Recall Analysis (Sensitivity)
- **Pure LLM**: Only catches 36.3% of true associations (high false negative rate)
- **Format A**: Catches 80.5% of true associations (good improvement)
- **Format B**: Catches 99.9% of true associations (excellent)
- **GraphRAG**: Catches 100% of true associations (perfect)

### Precision Analysis (Specificity)
- **Pure LLM**: 77.6% precision (some false positives from hallucination)
- **Format A**: 91.9% precision (well-grounded in retrieved data)
- **Format B**: 93.6% precision (very accurate with direct pairs)
- **GraphRAG**: 100% precision (deterministic graph queries)

### Clinical Implications
For drug safety applications:
- **Pure LLM**: Too many missed associations (dangerous)
- **Format A**: Good but may still miss ~20% of side effects
- **Format B**: Excellent balance (99.9% recall, 93.6% precision)
- **GraphRAG**: Gold standard for critical applications

---

**Test Dataset**: 19,520 binary queries (9,760 YES, 9,760 NO)
**Source**: `data/processed/evaluation_dataset.csv`
**Models**: Qwen2.5-7B-Instruct & Llama-3.1-8B-Instruct
**Date**: September 2025
**Note**: Metrics shown are from Qwen2.5-7B-Instruct (both models achieved similar results)

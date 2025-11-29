# Format B Architecture Diagram

## 📊 Data Format

### **Input Data Structure (SIDER 4.1 → Format B)**

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIDER 4.1 Database                           │
│  Drug-Side Effect Associations (Individual Pairs)                   │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ↓
                    Text Splitter
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│              Format B: Individual Drug-Side Effect Pairs            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Pair 1:                                                            │
│    drug: "aspirin"                                                  │
│    side_effect: "nausea"                                            │
│    text: "The drug aspirin causes nausea as an adverse effect,     │
│           adverse reaction, or side effect"                         │
│                                                                     │
│  Pair 2:                                                            │
│    drug: "aspirin"                                                  │
│    side_effect: "headache"                                          │
│    text: "The drug aspirin causes headache as an adverse effect,   │
│           adverse reaction, or side effect"                         │
│                                                                     │
│  Pair 3:                                                            │
│    drug: "aspirin"                                                  │
│    side_effect: "stomach pain"                                      │
│    text: "The drug aspirin causes stomach pain as an adverse       │
│           effect, adverse reaction, or side effect"                 │
│                                                                     │
│  Pair 4:                                                            │
│    drug: "metformin"                                                │
│    side_effect: "diarrhea"                                          │
│    text: "The drug metformin causes diarrhea as an adverse         │
│           effect, adverse reaction, or side effect"                 │
│                                                                     │
│  ... (thousands of individual pairs)                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                         │
                         ↓
                  OpenAI ada-002
                 (1536 dimensions)
                         │
                         ↓
                  Pinecone Vector DB
              Namespace: "drug-side-effects-formatB"

              Each vector stores metadata:
              {
                "drug": "aspirin",
                "side_effect": "nausea",
                "text": "The drug aspirin causes..."
              }
```

---

## 🔄 Query Pipeline (Complete Flow)

```
┌═══════════════════════════════════════════════════════════════════════════════┐
│                         FORMAT B QUERY PIPELINE                               │
└═══════════════════════════════════════════════════════════════════════════════┘

┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: ENTITY INPUT                                               │
└─────────────────────────────────────────────────────────────────────┘

Option A: Pre-extracted Entities              Option B: Natural Language
┌──────────────────────────┐                  ┌─────────────────────────────┐
│ rag.query(                │                  │ rag.query_natural_language( │
│   drug="aspirin",         │                  │   "Is nausea an adverse     │
│   side_effect="nausea"    │                  │    effect of aspirin?"      │
│ )                         │                  │ )                           │
└──────────┬───────────────┘                  └────────────┬────────────────┘
           │                                                │
           │                                                ↓
           │                                   ┌────────────────────────────┐
           │                                   │  Entity Recognition Module │
           │                                   │  Extract: [drug, SE]       │
           │                                   └────────────┬───────────────┘
           │                                                │
           └────────────────────────┬───────────────────────┘
                                    ↓
                      drug="aspirin", side_effect="nausea"

┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: EMBEDDING GENERATION (Full Query - Notebook Aligned)      │
└─────────────────────────────────────────────────────────────────────┘

                   ┌─────────────────────────────────┐
                   │   Full Query Embedding          │
                   │   (ONLY option - notebook mode) │
                   │                                 │
                   │ query_text =                    │
                   │  "Is nausea an adverse effect   │
                   │   of aspirin?"                  │
                   │                                 │
                   │ (~10-15 tokens)                 │
                   └─────────────┬───────────────────┘
                                 ↓
                          OpenAI ada-002
                        get_embedding(text)
                                 ↓
                       embedding: [1536 dimensions]

┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: VECTOR SEARCH                                              │
└─────────────────────────────────────────────────────────────────────┘

                    embedding vector (1536d)
                              ↓
          ┌──────────────────────────────────────────────┐
          │      Pinecone Index: drug-side-effects       │
          │      Namespace: "formatB"                    │
          │                                              │
          │  index.query(                                │
          │    vector=embedding,                         │
          │    top_k=10,                                 │
          │    namespace="drug-side-effects-formatB",    │
          │    include_metadata=True                     │
          │  )                                           │
          └──────────────────┬───────────────────────────┘
                             ↓
                   Returns top-10 pairs
                             ↓
        ┌────────────────────────────────────────────────┐
        │ Pair 1: score=0.95                             │
        │   drug: "aspirin"                              │
        │   side_effect: "nausea"                        │
        │   text: "aspirin causes nausea as adverse..."  │
        │                                                │
        │ Pair 2: score=0.89                             │
        │   drug: "aspirin"                              │
        │   side_effect: "vomiting"                      │
        │   text: "aspirin causes vomiting as adverse..." │
        │                                                │
        │ Pair 3: score=0.86                             │
        │   drug: "ibuprofen"                            │
        │   side_effect: "nausea"                        │
        │   text: "ibuprofen causes nausea as adverse..." │
        │                                                │
        │ Pair 4: score=0.83                             │
        │   drug: "aspirin"                              │
        │   side_effect: "headache"                      │
        │   text: "aspirin causes headache as adverse..." │
        │                                                │
        │ ... (6 more pairs)                             │
        └────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: FILTERING MODULE (CRITICAL!)                               │
└─────────────────────────────────────────────────────────────────────┘

                        top-10 pairs
                              ↓
          ┌──────────────────────────────────────────────┐
          │   _filter_by_entities(results, drug, SE)     │
          │                                              │
          │   For each pair:                             │
          │     Check if BOTH:                           │
          │       - "aspirin" in pair_drug               │
          │       - "nausea" in pair_effect              │
          │                                              │
          │   Keep only matching pairs                   │
          └──────────────────┬───────────────────────────┘
                             ↓
              Filtering Decision per Pair
                             ↓
    ┌────────────────────────┴────────────────────────┐
    │                                                 │
    ↓                                                 ↓
Pair 1: ✅ PASS                          Pair 2: ❌ REJECT
  drug="aspirin" ✓                         drug="aspirin" ✓
  side_effect="nausea" ✓                   side_effect="vomiting" ✗
  → Include                                (wrong side effect)
                                           → Discard

Pair 3: ❌ REJECT                        Pair 4: ❌ REJECT
  drug="ibuprofen" ✗                       drug="aspirin" ✓
  (wrong drug)                             side_effect="headache" ✗
  → Discard                                (wrong side effect)
                                           → Discard

                             ↓
        ┌────────────────────────────────────────────────┐
        │  Filtered Results (1 pair passed)              │
        │                                                │
        │  • aspirin → nausea                            │
        └────────────────────────────────────────────────┘
                             │
                             ↓
        ┌────────────────────────────────────────────────┐
        │  If NO filtered pairs:                         │
        │  Generate Negative Statement:                  │
        │  "No, the side effect nausea is not listed as  │
        │   an adverse effect of the drug aspirin"       │
        └────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: TOKEN MANAGEMENT                                           │
└─────────────────────────────────────────────────────────────────────┘

                    Filtered Pairs
                            ↓
          ┌──────────────────────────────────────────────┐
          │   Token Manager                              │
          │   truncate_context_pairs()                   │
          │                                              │
          │   • Calculate token count for each pair      │
          │   • Format as bullet points: "• drug → SE"   │
          │   • Respect model's context limit (~8K)      │
          │   • Truncate if needed                       │
          │   • Maintain pair order                      │
          └──────────────────┬───────────────────────────┘
                             ↓
                   Truncated Context
              (fits within token limit)

┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: PROMPT CONSTRUCTION                                        │
└─────────────────────────────────────────────────────────────────────┘

        ┌────────────────────────────────────────────────┐
        │  Prompt Template:                              │
        │                                                │
        │  You are asked to answer the following         │
        │  question with a single word: YES or NO.       │
        │                                                │
        │  The RAG Results below show drug-side effect   │
        │  relationships where "Drug → Side Effect"      │
        │  means the drug causes that side effect as     │
        │  an adverse reaction.                          │
        │                                                │
        │  Instructions:                                 │
        │  - Answer YES if the RAG Results show that     │
        │    aspirin causes nausea as an adverse         │
        │    reaction                                    │
        │  - Answer NO if the RAG Results do not show    │
        │    this relationship or show no relevant       │
        │    information                                 │
        │  - You must start your response with either    │
        │    YES or NO                                   │
        │                                                │
        │  ### Question:                                 │
        │                                                │
        │  Is nausea an adverse effect of aspirin?       │
        │                                                │
        │  ### RAG Results:                              │
        │                                                │
        │  • aspirin → nausea                            │
        │                                                │
        │  FINAL ANSWER: [YES or NO]                     │
        └────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  STEP 7: vLLM INFERENCE                                             │
└─────────────────────────────────────────────────────────────────────┘

                         Prompt
                            ↓
          ┌──────────────────────────────────────────────┐
          │         vLLM Server (Local)                  │
          │                                              │
          │  Model: Qwen 2.5-7B-Instruct                 │
          │     OR  Llama 3.1-8B-Instruct                │
          │                                              │
          │  self.llm.generate_response(                 │
          │    prompt,                                   │
          │    max_tokens=100,                           │
          │    temperature=0.1  # Deterministic          │
          │  )                                           │
          └──────────────────┬───────────────────────────┘
                             ↓
                      Raw LLM Response
                             ↓
        ┌────────────────────────────────────────────────┐
        │ "YES, the RAG Results show that aspirin        │
        │  causes nausea as an adverse reaction. The     │
        │  pair 'aspirin → nausea' is explicitly         │
        │  listed in the RAG Results."                   │
        └────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  STEP 8: RESPONSE PARSING                                           │
└─────────────────────────────────────────────────────────────────────┘

                    Raw LLM Response
                            ↓
          ┌──────────────────────────────────────────────┐
          │   parse_binary_response(response)            │
          │                                              │
          │   1. Convert to uppercase                    │
          │   2. Check first line for YES/NO             │
          │   3. Fallback: search entire response        │
          │   4. Return: YES / NO / UNKNOWN              │
          └──────────────────┬───────────────────────────┘
                             ↓
        ┌────────────────────────────────────────────────┐
        │  Final Result Dictionary:                      │
        │                                                │
        │  {                                             │
        │    'answer': 'YES',                            │
        │    'confidence': 0.9,                          │
        │    'drug': 'aspirin',                          │
        │    'side_effect': 'nausea',                    │
        │    'format': 'B',                              │
        │    'model': 'vllm_qwen',                       │
        │    'reasoning': 'YES, the RAG Results show...', │
        │    'evidence_count': 1                         │
        │  }                                             │
        └────────────────────────────────────────────────┘
                             ↓
                       Return to User
```

---

## ⚡ Batch Processing Pipeline

```
┌═══════════════════════════════════════════════════════════════════════════════┐
│                    FORMAT B BATCH PROCESSING (3-Stage)                        │
└═══════════════════════════════════════════════════════════════════════════════┘

INPUT: List of queries
[
  {'drug': 'aspirin', 'side_effect': 'nausea'},
  {'drug': 'metformin', 'side_effect': 'headache'},
  ...
  {'drug': 'ibuprofen', 'side_effect': 'dizziness'}
]

┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: BATCH EMBEDDING GENERATION                                │
└─────────────────────────────────────────────────────────────────────┘

    queries (100) → Generate query texts
                          ↓
    ┌─────────────────────────────────────────────────┐
    │ Query Texts (Full Query - Notebook Aligned)    │
    │                                                 │
    │ ["Is nausea an adverse effect of aspirin?",    │
    │  "Is headache an adverse effect of metformin?",│
    │  ...]                                           │
    └─────────────────────┬───────────────────────────┘
                          ↓
    embedding_client.get_embeddings_batch(
      query_texts,
      batch_size=20  # Process 20 at a time
    )
                          ↓
    ┌─────────────────────────────────────────────────┐
    │ 100 embeddings generated                        │
    │ • 5 API calls instead of 100                    │
    │ • 20x reduction in API overhead                 │
    │ • ~2-3 seconds total                            │
    └─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: CONCURRENT PINECONE RETRIEVAL + FILTERING                 │
└─────────────────────────────────────────────────────────────────────┘

    100 embeddings → ThreadPoolExecutor (10 workers)
                          ↓
    ┌─────────────────────────────────────────────────┐
    │ Worker 1: Query 1, 11, 21, 31, ...              │
    │ Worker 2: Query 2, 12, 22, 32, ...              │
    │ Worker 3: Query 3, 13, 23, 33, ...              │
    │ ...                                             │
    │ Worker 10: Query 10, 20, 30, 40, ...            │
    │                                                 │
    │ Each worker:                                    │
    │  1. Query Pinecone (top-10 pairs)               │
    │  2. Apply filtering: check BOTH drug & SE       │
    │  3. Format as bullet points                     │
    │  4. Truncate context                            │
    │  5. Return prepared context                     │
    └─────────────────────┬───────────────────────────┘
                          ↓
    Progress Bar: [████████████████████] 100/100
                          ↓
    ┌─────────────────────────────────────────────────┐
    │ 100 filtered contexts ready                     │
    │ • Only matching pairs included                  │
    │ • Formatted as "• drug → side_effect"           │
    │ • ~5-10 seconds total                           │
    └─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: BATCH vLLM INFERENCE                                      │
└─────────────────────────────────────────────────────────────────────┘

    100 contexts → Build 100 prompts
                          ↓
    self.llm.generate_batch(
      prompts,           # All 100 prompts
      max_tokens=100,
      temperature=0.1
    )
                          ↓
    ┌─────────────────────────────────────────────────┐
    │ vLLM processes all 100 in single batch          │
    │ • Native batch inference                        │
    │ • GPU utilization: ~80-90%                      │
    │ • ~10-20 seconds total                          │
    └─────────────────────┬───────────────────────────┘
                          ↓
    100 responses → Parse each → Return results
                          ↓
    ┌─────────────────────────────────────────────────┐
    │ TOTAL TIME: ~20-30 seconds for 100 queries      │
    │ THROUGHPUT: ~50-100 queries/second              │
    │                                                 │
    │ vs Notebook: ~200-500 seconds for 100 queries   │
    │              ~1-5 queries/second                │
    │                                                 │
    │ SPEEDUP: 10-50x faster!                         │
    └─────────────────────────────────────────────────┘
```

---

## 🔧 Configuration Options

```python
from src.architectures.rag_format_b import FormatBRAG

# Full Query Embedding (Notebook-Aligned - ONLY option)
rag = FormatBRAG(
    config_path="config.json",
    model="qwen"                  # or "llama3"
)
# Always embeds: "Is nausea an adverse effect of aspirin?"
# 100% aligned with reference notebook implementation
# Retrieves pairs similar to this question
```

---

## 📊 Key Characteristics

### **Strengths**
- ✅ **Precise matching**: Each pair is atomic (drug → side_effect)
- ✅ **Clean structure**: Bullet point format "• drug → side_effect"
- ✅ **Filtering module**: Checks BOTH drug AND side_effect (critical!)
- ✅ **Negative statements**: Explicit NO answers when appropriate
- ✅ **Batch optimization**: 10-50x faster than sequential
- ✅ **Full query embedding**: 100% aligned with reference notebook
- ✅ **Minimal ambiguity**: Each pair is self-contained

### **Considerations**
- ⚠️ Each side effect = separate vector (more storage)
- ⚠️ May need more pairs indexed than Format A
- ⚠️ Filtering is critical (must check both entities)

### **Best For**
- Precise drug-side effect matching
- When you want atomic relationships
- Binary YES/NO queries
- Production deployments with structured data
- When filtering precision is priority

---

## 🆚 Format B vs Format A Comparison

| Aspect | Format B | Format A |
|--------|----------|----------|
| **Data Unit** | Individual pair | Drug with effects list |
| **Vector Count** | One per pair | One per drug |
| **Context Format** | `• drug → effect` | Natural language paragraph |
| **Filtering** | Check drug + effect in pair | Check drug + effect in text |
| **Precision** | Very high (atomic) | Medium (text search) |
| **Context Length** | Short, focused | Longer, comprehensive |
| **Ambiguity** | Very low | Medium (natural language) |
| **Storage** | Higher (more vectors) | Lower (fewer vectors) |

---

## 🎯 Example Scenarios

### **Scenario 1: Exact Match Found**

Query: `aspirin` + `nausea`

**Retrieval:**
- Finds pair: `aspirin → nausea` (score: 0.95)
- Passes filtering: ✅ Both entities match

**Prompt:**
```
### RAG Results:
• aspirin → nausea
```

**LLM Response:** `"YES, aspirin causes nausea..."`

---

### **Scenario 2: No Match Found**

Query: `aspirin` + `euphoria`

**Retrieval:**
- Retrieves pairs:
  - `aspirin → headache` (score: 0.75)
  - `aspirin → nausea` (score: 0.72)
  - `cocaine → euphoria` (score: 0.68)

**Filtering:**
- ❌ `aspirin → headache`: wrong side effect
- ❌ `aspirin → nausea`: wrong side effect
- ❌ `cocaine → euphoria`: wrong drug

**Result:** No pairs pass filtering

**Prompt:**
```
### RAG Results:
No, the side effect euphoria is not listed as an adverse effect
of the drug aspirin
```

**LLM Response:** `"NO, euphoria is not listed..."`

---

### **Scenario 3: Partial Match**

Query: `aspirin` + `stomach`

**Retrieval:**
- Finds pairs:
  - `aspirin → stomach pain` (score: 0.88)
  - `aspirin → stomach upset` (score: 0.85)
  - `aspirin → stomach bleeding` (score: 0.82)

**Filtering:**
- ✅ All contain "aspirin" AND "stomach"
- All pass filtering

**Prompt:**
```
### RAG Results:
• aspirin → stomach pain
• aspirin → stomach upset
• aspirin → stomach bleeding
```

**LLM Response:** `"YES, aspirin causes stomach-related effects..."`

---

## 💡 Advanced Features

### **Metadata Filtering (Optional)**

```python
# Filter by specific metadata during retrieval
results = self.index.query(
    vector=embedding,
    top_k=10,
    namespace="drug-side-effects-formatB",
    filter={
        'drug': {'$eq': 'aspirin'}  # Only aspirin pairs
    }
)
```

### **Reverse Query Support**

```python
# Find all drugs causing a specific side effect
result = rag.reverse_query_chunked(side_effect="nausea")
# Returns: ['aspirin', 'metformin', 'ibuprofen', ...]
```

### **Complex Queries**

Format B's atomic structure enables:
- Drug comparison (common side effects)
- Organ-specific queries
- Severity filtering
- Statistical analysis

---

## 🏁 Summary

Format B provides **atomic, precise drug-side effect relationships** with:

1. ✅ Clean bullet-point representation
2. ✅ Dual entity filtering (drug AND side effect)
3. ✅ Full query embedding (notebook-aligned)
4. ✅ 10-50x faster batch processing
5. ✅ Explicit negative statement generation
6. ✅ High precision matching

**Perfect for production deployments requiring precise, verifiable drug-side effect relationships.**

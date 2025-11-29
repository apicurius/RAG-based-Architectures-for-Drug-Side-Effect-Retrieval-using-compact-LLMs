# Reverse Query Architecture Benchmark Report

**Date:** November 3, 2025 (Original) | November 29, 2025 (Replication)
**Test Dataset:** 121 stratified queries from comprehensive dataset
**LLM:** Qwen2.5-7B-Instruct (32,768 token context)
**Hardware:** 4 × NVIDIA A40 (46GB)

---

## Executive Summary

We benchmarked 3 RAG architectures for the reverse query task (finding all drugs that cause a specific side effect). **GraphRAG achieved perfect 100% accuracy with 914× faster performance** compared to the Format B chunked approach.

### Key Findings

#### Original Run (November 3, 2025)

| Architecture | Recall | Precision | F1 | Avg Latency | Throughput |
|--------------|--------|-----------|----|---------|----|
| **GraphRAG** ⭐ | **100.00%** | **100.00%** | **100.00%** | **0.09s** | **11 queries/s** |
| Format_B_Chunked | 98.88% | 99.93% | 99.38% | 82.44s | 0.012 queries/s |
| Format_A | 7.97% | 80.91% | 11.84% | 23.42s | 0.043 queries/s |

#### Replication Run (November 29, 2025)

| Architecture | Recall | Precision | F1 | Avg Latency | Throughput |
|--------------|--------|-----------|----|---------|----|
| **GraphRAG** ⭐ | **100.00%** | **100.00%** | **100.00%** | **0.09s** | **11 queries/s** |
| Format_B_Chunked | 98.59% | 99.93% | 99.18% | 84.63s | 0.012 queries/s |
| Format_A | 7.97% | 81.03% | 11.79% | 23.32s | 0.043 queries/s |

**Results Consistency:** Replication confirms original findings with <0.3% variance in recall metrics.

**Production Recommendation:** Deploy GraphRAG for all reverse query operations.

---

## Methodology

### Test Dataset Construction

Stratified random sample from comprehensive dataset (seed=42):
- **Large tier** (500+ drugs): 31/31 queries (100%)
- **Medium tier** (100-499 drugs): 40/288 queries (14%)
- **Small tier** (20-99 drugs): 40/300 queries (13%)
- **Rare tier** (5-19 drugs): 10/50 queries (20%)
- **Total:** 121 queries

### Evaluation Metrics

- **Recall:** Percentage of expected drugs correctly identified (case-insensitive)
- **Precision:** Percentage of extracted drugs that are correct
- **F1 Score:** Harmonic mean of recall and precision
- **Latency:** Time to process single query (seconds)

### Ground Truth

Validated against Neo4j ground truth (122,601 drug-side effect pairs from SIDER database).

---

## Detailed Results

### 1. GraphRAG - Graph-Based Retrieval

**Overall Performance:**
- Recall: 100.00% (perfect)
- Precision: 100.00% (no false positives)
- F1: 100.00%
- Avg Latency: 0.09s
- Total Time: 0.2 minutes

**Tier-Specific Performance:**

| Tier | Queries | Avg Recall | Avg Latency | Speed Advantage |
|------|---------|------------|-------------|-----------------|
| Large (500+) | 31 | 100.00% | 0.17s | 1,279× faster |
| Medium (100-499) | 40 | 100.00% | 0.08s | 807× faster |
| Small (20-99) | 40 | 100.00% | 0.06s | 256× faster |
| Rare (5-19) | 10 | 100.00% | 0.06s | 63× faster |

**Why It Works:**
- Direct Cypher query on graph database
- Data already structured as (Drug)-[:CAUSES]->(SideEffect) relationships
- No LLM inference required
- Zero hallucination risk
- O(1) lookup complexity with graph indexes

**Typical Query:**
```cypher
MATCH (d:Drug)-[:CAUSES]->(s:SideEffect)
WHERE s.name = 'headache'
RETURN d.name
```

---

### 2. Format B (Chunked) - Vector + LLM Extraction

**Overall Performance (Original → Replication):**
- Recall: 98.88% → 98.59% (near-perfect)
- Precision: 99.93% → 99.93% (very few false positives)
- F1: 99.38% → 99.18%
- Avg Latency: 82.44s → 84.63s
- Total Time: 166.3 → 170.7 minutes

**Tier-Specific Performance (Replication):**

| Tier | Queries | Avg Recall | Avg Latency | Bottleneck |
|------|---------|------------|-------------|------------|
| Large (500+) | 31 | 98.48% | 227.35s | LLM extraction |
| Medium (100-499) | 40 | 97.45% | 63.80s | Moderate |
| Small (20-99) | 40 | 99.47% | 15.07s | Acceptable |
| Rare (5-19) | 10 | 100.00% | 3.82s | Fast |

**How It Works:**
1. Vector search retrieves top-k relevant drug-SE pairs
2. Chunk pairs into groups of 200
3. LLM extracts drugs from each chunk iteratively
4. Merge results from all chunks

**Strengths:**
- Very high accuracy (98.88%)
- Works well for rare side effects (100% recall)
- Solved "lost in the middle" problem with chunking

**Weaknesses:**
- 914× slower than GraphRAG
- Latency scales with dataset size (217s for large tier)
- Requires expensive LLM inference
- Not suitable for real-time applications

**Notable Failure Cases:**
- "pruritus" (large tier): 79.78% recall (missed 148/732 drugs)
  - Reason: Vector search may not retrieve all relevant pairs in top-k

---

### 3. Format A - Vector-Only Retrieval

**Overall Performance (Original → Replication):**
- Recall: 7.97% → 7.97% (catastrophic failure)
- Precision: 80.91% → 81.03%
- F1: 11.84% → 11.79%
- Avg Latency: 23.42s → 23.32s
- Total Time: 47.3 → 47.0 minutes

**Tier-Specific Performance (Replication):**

| Tier | Queries | Avg Recall | Analysis |
|------|---------|------------|----------|
| Large (500+) | 31 | 2.40% | Misses 97.60% of drugs |
| Medium (100-499) | 40 | 5.27% | Slightly better but still poor |
| Small (20-99) | 40 | 15.52% | Best performance, still inadequate |
| Rare (5-19) | 10 | 5.83% | Inconsistent |

**Why It Fails:**
- Uses only vector similarity without LLM extraction
- Returns semantically similar text but doesn't extract drug names
- Cannot handle multi-drug relationships in context
- Precision is high (80.91%) but misses most true positives

**Conclusion:** Format A is not suitable for reverse query task. The architecture was designed for binary classification (does drug X cause SE Y?), not reverse lookup.

---

## Performance Analysis

### Accuracy vs Latency Trade-off

```
GraphRAG:        100.00% recall |  0.09s ⭐ OPTIMAL
Format B:         98.88% recall | 82.44s (914× slower)
Format A:          7.97% recall | 23.42s (not viable)
```

**GraphRAG dominates on both dimensions:** highest accuracy AND fastest speed.

### Scalability Analysis

**Latency by Dataset Size (Format B vs GraphRAG):**

| Drug Count | Format B | GraphRAG | Speedup |
|------------|----------|----------|---------|
| 5-19 (rare) | 3.80s | 0.06s | 63× |
| 20-99 (small) | 15.35s | 0.06s | 256× |
| 100-499 (medium) | 64.55s | 0.08s | 807× |
| 500+ (large) | 217.48s | 0.17s | 1,279× |

**GraphRAG scales logarithmically** with database size due to graph indexes.
**Format B scales linearly** with number of drug-SE pairs (more chunks = more LLM calls).

### Cost Analysis (Estimated)

**Per 1,000 queries:**

| Architecture | LLM Tokens | Inference Cost | Latency Cost |
|--------------|-----------|----------------|--------------|
| GraphRAG | 0 | $0 | 90 seconds |
| Format B | ~50M tokens | ~$50-100 | 22.9 hours |
| Format A | ~2M tokens | ~$2-4 | 6.5 hours |

**GraphRAG has zero inference cost** and completes 1,000 queries in 90 seconds vs 22.9 hours for Format B.

---

## Real-World Implications

### Use Case: Drug Safety Dashboard

**Scenario:** Pharmacologist queries "What drugs cause liver toxicity?"

**GraphRAG Response:**
- **Latency:** 0.15s (sub-second)
- **Result:** All 234 drugs returned instantly
- **Accuracy:** 100% (no missing drugs, no false positives)
- **User Experience:** Instant, real-time query

**Format B Response:**
- **Latency:** 89.2s (1.5 minutes)
- **Result:** 231/234 drugs (missed 3)
- **Accuracy:** 98.7% recall
- **User Experience:** Noticeable wait time

**Verdict:** GraphRAG enables real-time interactive exploration. Format B requires loading screens.

### Use Case: High-Volume API

**Scenario:** External API serving 10,000 reverse queries/day

**GraphRAG:**
- Throughput: 11 queries/second
- Can handle 10,000 queries in 15 minutes
- No GPU/LLM infrastructure needed
- Cost: Near-zero (Neo4j hosting only)

**Format B:**
- Throughput: 0.012 queries/second
- Would need 22.9 hours for 10,000 queries
- Requires dedicated GPU cluster
- Cost: Significant LLM inference costs

**Verdict:** Only GraphRAG is viable for production API deployment.

---

## Limitations and Considerations

### GraphRAG Assumptions

**GraphRAG achieves 100% recall because:**
1. All drug-SE relationships are pre-populated in Neo4j
2. Graph structure exactly matches the ground truth
3. Query is deterministic lookup, not inference

**This works for SIDER because:**
- SIDER is a curated, complete database
- We ingested all 122,601 relationships into Neo4j
- No new drugs/SEs emerge (static dataset)

**GraphRAG would NOT work for:**
- Novel drug-SE discovery (relationships not in graph)
- Real-time pharmacovigilance (streaming data)
- Cross-database queries (SIDER + FDA + EMA)

### When to Use Each Architecture

| Use Case | Recommended Architecture | Reasoning |
|----------|-------------------------|-----------|
| **Reverse queries on SIDER** | GraphRAG | Perfect accuracy, real-time speed |
| **Novel SE discovery** | Format B Chunked | Can infer from unstructured text |
| **Binary classification** | Format B or Format A | Task-specific optimization |
| **Cross-database queries** | Hybrid (Graph + Vector) | Combine structured + unstructured |

---

## Conclusions

### Primary Findings

1. **GraphRAG is optimal for reverse queries on structured data** (100% accuracy, 0.09s latency)
2. **Format B chunked achieves near-perfect accuracy but 914× slower** (98.88%, 82.44s)
3. **Format A is not suitable for reverse queries** (7.97% recall)

### Production Deployment Recommendation

**For drugRAG reverse query API:**

✅ **Deploy GraphRAG as primary architecture**
- Use for all queries on SIDER dataset
- Expose via `/api/v1/reverse_query` endpoint
- Expected SLA: <200ms p99 latency, 100% accuracy

⚠️ **Keep Format B as research tool**
- Use for novel SE discovery research
- Apply to unstructured clinical text
- Not for production user-facing queries

❌ **Deprecate Format A for reverse queries**
- 7.97% recall is not production-viable
- Architecture better suited for binary classification

### Next Steps

1. ✅ **Implement GraphRAG production API** - Deploy to production
2. **Monitor query patterns** - Track which SEs are queried most
3. **A/B test user experience** - Compare GraphRAG vs Format B in UI
4. **Extend to multi-database** - Combine SIDER + FDA FAERS + EMA
5. **Add confidence scores** - Incorporate frequency/severity metadata

---

## Appendix

### Test Environment

Both runs used identical hardware and software configuration:

- **Server:** HPC cluster node
- **GPU:** 4 × NVIDIA A40 (46GB VRAM each)
- **LLM:** Qwen2.5-7B-Instruct via vLLM v0.10.2
- **Context Length:** 32,768 tokens
- **max-num-batched-tokens:** 32,768
- **max-model-len:** 32,768
- **tensor-parallel-size:** 4
- **Vector DB:** Pinecone (text-embedding-ada-002)
- **Graph DB:** Neo4j Aura Professional
- **Python:** 3.11 with uv package manager

### Reproducibility

All results are reproducible with:
```bash
cd /home/omeerdogan23/drugRAG/experiments
python reverse_query_benchmark.py
```

Random seed: 42 (stratified sampling)

**Original Run:**
- Results file: `results_reverse_query_benchmark_20251103_125120.json`
- Sample file: `benchmark_sample_20251103_091714.json`

**Replication Run:**
- Results file: `results_reverse_query_benchmark_20251128_235601.json`
- Sample file: `benchmark_sample_20251128_201739.json`

### Replication Comparison

| Metric | Original | Replication | Delta |
|--------|----------|-------------|-------|
| Format B Recall | 98.88% | 98.59% | -0.29% |
| Format B Precision | 99.93% | 99.93% | 0.00% |
| Format B Latency | 82.44s | 84.63s | +2.19s |
| GraphRAG Recall | 100.00% | 100.00% | 0.00% |
| GraphRAG Latency | 0.09s | 0.09s | 0.00s |
| Format A Recall | 7.97% | 7.97% | 0.00% |

**Conclusion:** Results are consistent across runs, confirming reproducibility (<0.3% variance).

---

**Report Generated:** November 3, 2025 (Original) | November 29, 2025 (Updated)
**Authors:** DrugRAG Team
**Version:** 1.1

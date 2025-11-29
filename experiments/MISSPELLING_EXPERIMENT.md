# Misspelling Robustness Experiment - Quick Reference

## Purpose

Test semantic understanding of LLM/RAG approaches vs naive string matching by evaluating how well different architectures handle the 10 misspelled drug names from `misspellings.csv`.

## Quick Start

```bash
cd /home/omeerdogan23/drugRAG/experiments

# Run full experiment (4 architectures × 2 models)
python evaluate_misspelling.py --architectures all --models both

# Or test specific configurations
python evaluate_misspelling.py --architectures pure_llm format_b --models qwen
```

## What It Does

1. **Generates Datasets**
   - Extracts 10 drugs from `misspellings.csv`
   - Creates queries for each drug from `evaluation_dataset.csv`
   - Produces two datasets: correct spellings vs misspelled

2. **Tests Each Architecture**
   - Pure LLM (baseline semantic understanding)
   - RAG Format A (aggregated documents)
   - RAG Format B (granular pairs)
   - GraphRAG (exact string matching)

3. **Calculates Degradation**
   - Compares correct vs misspelled performance
   - Measures robustness to spelling errors
   - Identifies which approaches handle typos best

## Expected Results

| Architecture | Expected Degradation | Reason |
|--------------|---------------------|--------|
| Pure LLM | ~5-10% | Semantic understanding from training |
| RAG Format A | ~20-30% | Embedding similarity robust to typos |
| RAG Format B | ~30-50% | Embeddings + exact name filtering |
| GraphRAG | ~80-95% | Exact string matching fails |

**Key Insight**: LLM/RAG approaches vastly superior to naive lookup for real-world scenarios with spelling errors.

## Output Location

Results saved to: `results/misspelling_experiment/`

Files:
- `detailed_results_TIMESTAMP.json` - Complete metrics
- `comparison_TIMESTAMP.csv` - Comparison table
- `summary_report_TIMESTAMP.txt` - Human-readable summary

## Runtime

~30-45 minutes for all 16 configurations with batch processing.

## Prerequisites

- vLLM server running
- Pinecone vector database configured
- Neo4j graph database configured (for GraphRAG)

## Implementation Details

### Dataset Generator
`src/utils/misspelling_dataset_generator.py`
- Loads misspelling pairs from CSV
- Filters evaluation dataset to test drugs
- Creates matched correct/misspelled datasets

### Evaluation Script
`experiments/evaluate_misspelling.py`
- Runs batch evaluation for efficiency
- Calculates comprehensive metrics
- Computes degradation and robustness scores
- Saves detailed results and comparisons

### Misspelling Types Tested

1. Letter addition (r): `lormetazepam` → `lormetazerpam`
2. Letter omission (o): `griseofulvin` → `grisefulvin`
3. Letter switch (p↔d): `lercanidipine` → `lercanipidine`
4. Multi-letter addition (io): `miglitol` → `miglitilol`
5. Letter omission (u): `fluoxetine` → `floxetine`
6. Letter switch (n↔r): `ropinirole` → `ropirinole`
7. Letter substitution (o→a): `latanoprost` → `latanaprost`
8. Letter switch (a↔e): `nateglinide` → `netaglinide`
9. Letter addition (l): `adefovir` → `adeflovir`
10. Letter substitution (o→a): `levobunolol` → `levabnolol`

These represent realistic medical term misspellings that could occur in real-world queries.

# Misspelling Robustness Experiment

## Overview

This experiment tests how different DrugRAG architectures handle misspelled drug names, demonstrating that **semantic understanding** in LLM and RAG approaches is superior to naive string matching.

## Experiment Design

### Test Drugs (10 total)
The experiment uses 10 carefully curated drug name misspellings from `experiments/misspellings.csv`:

1. `lormetazepam` → `lormetazerpam` (one letter addition "r")
2. `griseofulvin` → `grisefulvin` (one letter omission "o")
3. `lercanidipine` → `lercanipidine` (two letters switched "p" and "d")
4. `miglitol` → `miglitilol` (two letter addition "io")
5. `fluoxetine` → `floxetine` (one letter omission "u")
6. `ropinirole` → `ropirinole` (two letter switched "n" and "r")
7. `latanoprost` → `latanaprost` (one letter switched "o" to "a")
8. `nateglinide` → `netaglinide` (two letter switched "a" and "e")
9. `adefovir` → `adeflovir` (addition of "l")
10. `levobunolol` → `levabnolol` (one letter switched "o" to "a")

### Architectures Tested

1. **Pure LLM** (Baseline)
   - Direct question → LLM inference
   - No retrieval step
   - Expected: **Minimal degradation** (~5-10%)
   - Reason: Semantic understanding from training data

2. **RAG Format A** (Aggregated Documents)
   - Drug → [list of side effects]
   - Vector similarity retrieval
   - Expected: **Moderate degradation** (~20-30%)
   - Reason: Embeddings partially robust to typos

3. **RAG Format B** (Granular Pairs)
   - Individual drug-effect pairs
   - Includes exact name filtering: `drug.lower() in pair_drug.lower()`
   - Expected: **Moderate-High degradation** (~30-50%)
   - Reason: Embedding similarity + exact name filtering vulnerability

4. **GraphRAG** (Neo4j)
   - Exact string matching in Cypher: `WHERE s.name = '{drug}'`
   - Expected: **Severe degradation** (~80-95%)
   - Reason: Exact string matching fails completely with typos

### Models Tested

- **Qwen2.5-32B-Instruct** - Primary model
- **LLAMA3-70B-Instruct** - Comparison model

### Metrics

For each architecture × model × condition (correct/misspelled):

- **Accuracy**: (TP + TN) / Total
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **Precision**: TP / (TP + FP)
- **Sensitivity** (Recall): TP / (TP + FN)
- **Specificity**: TN / (TN + FP)

Plus degradation metrics:

- **Absolute Degradation**: correct_metric - misspelled_metric
- **Percentage Degradation**: (absolute_degradation / correct_metric) × 100
- **Robustness Score**: misspelled_metric / correct_metric (higher is better)

## Running the Experiment

### Quick Start (All Architectures, Both Models)

```bash
cd /home/omeerdogan23/drugRAG/experiments
python evaluate_misspelling.py --architectures all --models both
```

### Custom Configuration

```bash
# Test only specific architectures
python evaluate_misspelling.py --architectures pure_llm format_b --models qwen

# Test only one model
python evaluate_misspelling.py --architectures all --models llama3

# Specify custom config
python evaluate_misspelling.py --config ../config.json --results-dir ../results/custom_misspelling
```

### Prerequisites

1. **vLLM Server Running**:
   ```bash
   # Start vLLM server (if not already running)
   ./start_vllm_server.sh
   ```

2. **Datasets Generated**:
   - Script will auto-generate from `experiments/misspellings.csv` and `data/processed/evaluation_dataset.csv`
   - Output: `data/processed/misspelling_experiment_correct.csv` and `misspelling_experiment_misspelled.csv`

## Expected Runtime

- ~30-45 minutes for all 16 configurations (4 architectures × 2 models × 2 conditions)
- Batch processing optimization significantly reduces time vs individual queries

## Output Files

Results are saved to `results/misspelling_experiment/`:

1. **`detailed_results_YYYYMMDD_HHMMSS.json`**
   - Complete metrics for each architecture × model × condition
   - Degradation calculations
   - Timing information

2. **`comparison_YYYYMMDD_HHMMSS.csv`**
   - Comparison table with:
     - Architecture, Model, Metric
     - Correct vs Misspelled values
     - Degradation percentages
     - Robustness scores

3. **`summary_report_YYYYMMDD_HHMMSS.txt`**
   - Human-readable summary
   - Key insights and findings

## Expected Results

### Robustness Ranking (Best to Worst)

1. **Pure LLM** - Highest robustness (~90-95% retention)
   - Semantic understanding from training
   - Minimal impact from spelling errors

2. **RAG Format A** - Good robustness (~70-80% retention)
   - Embedding similarity captures semantic meaning
   - Aggregated documents provide context

3. **RAG Format B** - Moderate robustness (~50-70% retention)
   - Embedding similarity helps
   - Exact name filtering causes some failures

4. **GraphRAG** - Low robustness (~5-20% retention)
   - Exact string matching in Cypher queries
   - Complete failure on misspelled names

### Key Insight

This experiment demonstrates that **LLM and RAG approaches using embeddings are vastly superior to naive string lookup** for real-world scenarios where users may make spelling errors.

## Analysis

### Why Pure LLM is Most Robust

- Trained on massive text corpora with natural spelling variations
- Internal representations capture semantic meaning beyond exact strings
- Can "autocorrect" or understand intent despite typos

### Why RAG with Embeddings is Moderately Robust

- Embedding models (e.g., text-embedding-ada-002) learn semantic similarity
- Similar spellings → similar embeddings
- Retrieval still works even if not exact match

### Why GraphRAG Fails

- Cypher query: `WHERE s.name = '{drug}'` requires exact match
- `'asprin' != 'aspirin'` → no relationship returned
- No semantic understanding in database lookup

## Visualization Ideas

Potential plots to generate from results:

1. **Degradation Bar Chart**: Percentage degradation by architecture
2. **Robustness Heatmap**: Architecture × Model robustness scores
3. **F1 Comparison**: Side-by-side correct vs misspelled F1 scores
4. **Per-Drug Analysis**: Which misspellings cause most issues

## Contact

For questions or issues with this experiment, please check:
- Main evaluation script: `experiments/evaluate_misspelling.py`
- Dataset generator: `src/utils/misspelling_dataset_generator.py`

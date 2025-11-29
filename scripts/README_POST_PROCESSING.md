# Post-Processing Instructions for Comprehensive Dataset

## When to Run

After the parallel dataset generation completes (`generate_comprehensive_dataset_parallel.py`), you need to fix the case sensitivity bug in the evaluation metrics.

## Quick Start

```bash
cd /home/omeerdogan23/drugRAG/scripts

# Wait for dataset generation to complete, then run:
python3 fix_dataset_case_sensitivity.py ../data/processed/comprehensive_reverse_queries_YYYYMMDD_HHMMSS.json
```

**Replace `YYYYMMDD_HHMMSS` with the actual timestamp from the generated file.**

## What This Does

The script fixes a case sensitivity bug where:
- Ground truth has: `"acamprosate"` (lowercase)
- LLM extracted: `"Acamprosate"` (capitalized)
- Original eval marked it as WRONG (false negative)
- Fixed eval marks it as CORRECT ✅

## Expected Output

1. **Corrected dataset:** `comprehensive_reverse_queries_YYYYMMDD_HHMMSS_case_corrected.json`
2. **Correction report:** `comprehensive_reverse_queries_YYYYMMDD_HHMMSS_case_correction_report.md`

## Expected Results

Based on checkpoint analysis (first 50 queries):

| Metric | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| Average Recall | 97.11% | **98.01%** | +0.90% |
| Queries <95% | 7/50 (14%) | 4/50 (8%) | -43% failures |

**Major improvements expected:**
- `thrombocytopenia`: 68.86% → 95.74% (+26.89%)
- `vomiting`: 87.43% → 98.81% (+11.39%)
- `body temperature increased`: 93.24% → 99.21% (+5.97%)

## Usage

### Basic Usage (Recommended)

```bash
python3 fix_dataset_case_sensitivity.py <dataset_path>
```

The script will automatically find the ground truth at `../data/processed/neo4j_ground_truth.json`.

### Advanced Usage

If ground truth is in a different location:

```bash
python3 fix_dataset_case_sensitivity.py <dataset_path> <ground_truth_path>
```

### Example

```bash
python3 fix_dataset_case_sensitivity.py \
  ../data/processed/comprehensive_reverse_queries_20251102_180545.json \
  ../data/processed/neo4j_ground_truth.json
```

## Output Files

### 1. Corrected Dataset (`*_case_corrected.json`)

Same structure as original, but with recalculated metrics:
- `recall`: Case-insensitive matching
- `precision`: Case-insensitive matching
- `f1`: Recalculated from corrected precision/recall

**Use this file for all future analysis!**

### 2. Correction Report (`*_case_correction_report.md`)

Detailed markdown report showing:
- Overall impact on performance
- Queries with significant changes (>5% improvement)
- Before/after comparison
- Technical details of the fix

## What Gets Fixed

**Metrics recalculated:**
- ✅ Recall (true positive rate)
- ✅ Precision (positive predictive value)
- ✅ F1 score
- ✅ Average recall (metadata)
- ✅ Average precision (metadata)

**Data preserved (unchanged):**
- ✅ Extracted drug lists
- ✅ Expected drug counts
- ✅ Side effect names
- ✅ Tier classifications
- ✅ Timestamps
- ✅ YES/NO examples

## Validation

After running, check the report to confirm:

1. **Average recall improved** (expected +0.5% to +1.5%)
2. **Queries <95% reduced** (expected 40-50% fewer failures)
3. **Major improvements** on thrombocytopenia, vomiting, body temperature queries

## Next Steps

After post-processing:

1. **Review the correction report** to understand the impact
2. **Use the corrected dataset** (`*_case_corrected.json`) for all analysis
3. **Update any existing evaluations** that used the original dataset
4. **Document the correction** in your final evaluation report

## Troubleshooting

**Error: Dataset not found**
- Check the file path is correct
- Ensure you're in the `scripts/` directory
- Use absolute paths if needed

**Error: Ground truth not found**
- Verify `neo4j_ground_truth.json` exists at `../data/processed/`
- Or specify the correct path manually

**Wrong improvement numbers**
- Expected improvement: +0.5% to +1.5% average recall
- If much higher/lower, check you're using the right dataset

## Technical Details

**What changed:**
```python
# Before (case-sensitive - WRONG)
matches = set(extracted_drugs) & set(expected_drugs)

# After (case-insensitive - CORRECT)
matches = set([d.lower() for d in extracted_drugs]) & set([d.lower() for d in expected_drugs])
```

**Why this matters:**
- LLMs may capitalize drug names differently than ground truth
- Case sensitivity causes false negatives
- Real performance is ~1% higher than originally calculated

## Files

- **Script:** `scripts/fix_dataset_case_sensitivity.py`
- **Test run:** Already validated on checkpoint_50 (50 queries)
- **This guide:** `scripts/README_POST_PROCESSING.md`

---

**Last updated:** 2025-11-02
**Tested on:** comprehensive_dataset_checkpoint_50.json (50 queries)
**Status:** ✅ Ready for production use

# Case Sensitivity Correction Report

**Date:** 2025-11-03 08:21:30
**Original Dataset:** comprehensive_reverse_queries_20251102_225909.json
**Corrected Dataset:** comprehensive_reverse_queries_20251102_225909_case_corrected.json

---

## Summary

**Root Cause:** The evaluation used case-sensitive string matching, causing false negatives when the LLM extracted drugs with different capitalization than the ground truth.

**Example:**
- Ground truth: `"acamprosate"` (lowercase)
- LLM extracted: `"Acamprosate"` (capitalized)
- Original evaluation: ❌ Marked as wrong (false negative)
- Corrected evaluation: ✅ Correctly matched

---

## Overall Impact

| Metric | Original | Corrected | Improvement |
|--------|----------|-----------|-------------|
| **Average Recall** | 98.76% | **98.84%** | **+0.08%** |

**Queries affected:** 2 queries with >5% improvement

**Comparison to Priority 1:**
- Priority 1 (validated): 98.37%
- Original comprehensive: 98.76% (+0.39%)
- **Corrected comprehensive: 98.84%** (+0.47%)

---

## Significant Changes (>5% improvement)

Total: 2 queries

| Side Effect | Original | Corrected | Improvement | Tier | Drugs |
|------------|----------|-----------|-------------|------|-------|
| thrombocytopenia | 68.86% | **95.74%** | +26.89% | large | 496/517 |
| hypertension | 82.54% | **98.28%** | +15.73% | medium | 459/464 |


---

## Performance Distribution

**Queries below 95% recall:**
- Original: 30/669 (4.5%)
- Corrected: 27/669 (4.0%)
- **Improvement: 3 fewer failures** (10.0% reduction)

---

## Conclusion

The case sensitivity bug was masking the true performance of the chunked strategy. With case-insensitive matching:

✅ Average recall improves from 98.76% to **98.84%**
✅ Performance now matches Priority 1 validation (98.84% vs 98.37%)
✅ 3 fewer apparent "failures"

**Recommendation:** Use the case-corrected dataset (`*_case_corrected.json`) for all future analysis and evaluation. The corrected metrics accurately reflect the true performance of the chunked extraction strategy.

---

## Technical Details

**Fix applied:**
```python
# Before (case-sensitive)
extracted_set = set(extracted_drugs)
expected_set = set(expected_drugs)
matches = extracted_set & expected_set

# After (case-insensitive)
extracted_set = set([d.lower() for d in extracted_drugs])
expected_set = set([d.lower() for d in expected_drugs])
matches = extracted_set & expected_set
```

**Files:**
- Original dataset: `comprehensive_reverse_queries_20251102_225909.json`
- Corrected dataset: `comprehensive_reverse_queries_20251102_225909_case_corrected.json`
- This report: `comprehensive_reverse_queries_20251102_225909_case_correction_report.md`

---

**Generated:** 2025-11-03 08:21:30
**Script:** `scripts/fix_dataset_case_sensitivity.py`

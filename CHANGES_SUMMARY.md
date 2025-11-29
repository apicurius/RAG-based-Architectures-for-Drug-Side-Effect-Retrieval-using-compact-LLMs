# DrugRAG Configuration Improvements - Summary

## Date: 2025-11-24

## Changes Made

### 1. Aligned Llama Parameters with Qwen
**File:** `llama.sh`

**Problem:** Llama had `--max-model-len 4096` while Qwen had `--max-model-len 16384`, creating a 4x difference in context window.

**Fix:** Updated Llama configuration to match Qwen:
- Changed `--max-model-len` from 4096 to **16384**
- Updated display message from "4K tokens" to "16K tokens"
- Updated comment to match Qwen's configuration style

**Impact:** Both models now have identical starting parameters for fair comparison.

---

### 2. Added Detailed Prompt-Answer Pair Logging

#### Architecture Files Updated:
All architecture classes now return `prompt` and `full_response` fields:

1. **`src/architectures/rag_format_a.py`**
   - Added `prompt` field to return dictionary
   - Added `full_response` field to return dictionary

2. **`src/architectures/rag_format_b.py`**
   - Added `prompt` field to return dictionary
   - Added `full_response` field to return dictionary

3. **`src/architectures/graphrag.py`**
   - Added `prompt` field to return dictionary
   - Added `full_response` field to return dictionary

4. **`src/models/vllm_model.py`**
   - Added `prompt` field to VLLMModel.query() return dictionary
   - Both VLLMQwenModel and VLLMLLAMA3Model inherit this enhancement

#### Evaluation Script Updated:
**File:** `experiments/evaluate_vllm.py`

**Changes:**
1. Added `prompt` field to detailed_results CSV output (line 467)
2. Added prompt logging for misclassifications (line 455-456)
3. Added `prompt` and `full_response` to JSON output (lines 547-548)
4. Enhanced result summary with detailed logging information (lines 577-580)

**Output Files Now Include:**
- CSV: `evaluation_logs/{timestamp}_{architecture}_{test_size}/detailed_results.csv`
  - Contains: sample_id, drug, side_effect, ground_truth, predicted, is_correct, confidence, **prompt**, full_response, model, architecture, retrieval_context, num_retrieved_docs, retrieval_scores, elapsed_time_s
  
- JSON: `results_vllm_{architecture}_{test_size}_{timestamp}.json`
  - Contains: drug, side_effect, true_label, predicted, confidence, **prompt**, full_response

---

## Benefits

### 1. Fair Model Comparison
- Both Qwen and Llama now have identical context windows (16K tokens)
- Eliminates bias from different parameter settings
- Ensures apples-to-apples performance comparison

### 2. Enhanced Debugging Capability
- Full prompt visibility for error analysis
- Complete response tracking for quality assessment
- Enables detailed investigation of model behavior
- Facilitates prompt engineering improvements

### 3. Audit Trail
- Complete record of all model interactions
- Reproducible results with exact prompts
- Comprehensive data for academic publication
- Enables post-hoc analysis without re-running experiments

---

## Files Modified

1. `llama.sh` - Parameter alignment
2. `src/architectures/rag_format_a.py` - Logging enhancement
3. `src/architectures/rag_format_b.py` - Logging enhancement
4. `src/architectures/graphrag.py` - Logging enhancement
5. `src/models/vllm_model.py` - Logging enhancement
6. `experiments/evaluate_vllm.py` - Output enhancement

---

## Testing Recommendations

Run a small test to verify changes:
```bash
# Start llama server with new parameters
./llama.sh

# Run evaluation with detailed logging
cd experiments
python evaluate_vllm.py --architecture format_a_llama3 --test_size 10

# Check output files
ls -la evaluation_logs/
cat results_vllm_*.json | jq '.detailed_results[0].prompt'
```

---

## Notes

- No changes were made to enhanced architectures (enhanced_format_b, enhanced_graphrag) as they're not used in the standard evaluation flow
- The evaluate_reverse_binary.py script was also updated but is not currently in use
- All changes are backward compatible - existing code will continue to work


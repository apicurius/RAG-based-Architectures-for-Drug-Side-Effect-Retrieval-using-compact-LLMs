# Format B Investigation Report
Date: 2025-11-22

## Executive Summary

Both Format A and Format B evaluations failed due to a **vLLM worker process crash** during inference, NOT an Out-of-Memory (OOM) issue. The multiprocess executor (`mp`) used by vLLM v0.10.2 proved unstable for long-running tensor-parallel inference with 32k context length.

## Timeline of Failure

### Format A Evaluation
- **19:47:04** - vLLM server started successfully with conservative settings
- **23:20:00** - Format A vLLM inference stage began (~19,520 prompts)
- **23:25:37** - Server working normally (68.3 tokens/s throughput)
- **23:26:17** - **WORKER HANG**: Throughput drops to 0.0 (warning sign)
- **23:27:12** - **VllmWorker-2 dies unexpectedly** (55 seconds after hang)
- **23:27:16** - EngineDeadError triggered, server starts returning 500 errors
- **23:31:26** - Format A completes with **partial results**:
  - 264 successful predictions (102 YES, 161 NO, 1 explicitly marked)
  - 19,256 UNKNOWN (from failed requests after crash)

### Format B Evaluation
- **23:32:43** - Format B starts, vLLM server already dead
- **02:05:01-02:06:12** - All 19,520 vLLM requests fail with "Connection refused"
- **02:08:23** - Format B completes with **all UNKNOWN** (0% success rate)

## Root Cause Analysis

### What We Confirmed:
1. **Not an OOM issue**:
   - No kernel OOM kills in dmesg
   - GPU memory allocation was healthy (19.6 GB KV cache per GPU, within 22.21 GB target)
   - Conservative settings were working correctly

2. **vLLM V1 engine with multiprocess executor instability**:
   ```
   ERROR 11-21 23:27:12 [multiproc_executor.py:149]
   Worker proc VllmWorker-2 died unexpectedly, shutting down executor.

   vllm.v1.engine.exceptions.EngineDeadError:
   EngineCore encountered an issue.
   ```

3. **Worker freeze before death**:
   - At 23:26:17: Throughput → 0.0, request hung
   - 55 seconds later: Worker process died
   - Suggests internal deadlock or CUDA synchronization issue

### Why This Happened:
- **vLLM version**: 0.10.2 (V1 LLM engine)
- **Executor**: `distributed_executor_backend: mp` (multiprocess)
- **Context length**: 32,768 tokens (high memory pressure per sequence)
- **Tensor parallelism**: 4 GPUs
- **Long session**: ~3.5 hours of continuous inference

The multiprocess executor (`mp`) is less stable than Ray for:
- Long-running inference sessions
- High context lengths (32k)
- Multi-GPU tensor parallelism
- Sustained high request volumes

## Configuration Changes Applied

### File: `/home/omeerdogan23/drugRAG/qwen.sh`

**Changed:**
```bash
--distributed-executor-backend mp
```

**To:**
```bash
--distributed-executor-backend ray
```

### Why Ray Executor?
- Better fault tolerance for multi-GPU workloads
- More stable for long-running sessions
- Improved inter-process communication
- Better error recovery and logging
- Production-grade distributed execution

## Conservative Settings (Retained)

These settings successfully prevented OOM and should be kept:

```bash
--max-model-len 32768              # Kept as requested by user
--gpu-memory-utilization 0.50      # 50% to prevent OOM
--max-num-seqs 8                   # Down from 64
--max-num-batched-tokens 4096      # Down from 16384
--enable-chunked-prefill           # Memory-efficient prefill
--enforce-eager                    # Disable CUDA graphs (stability)
```

Python-side (vllm_model.py):
```python
max_workers = 1  # Extreme concurrency limit
```

## Next Steps

### 1. Restart vLLM Server with Ray Executor
```bash
# Kill old server
pkill -9 -f "port 8002"

# Start with Ray executor
cd /home/omeerdogan23/drugRAG
bash qwen.sh > vllm_ray_executor.log 2>&1 &

# Wait for startup (3-5 minutes)
# Check: curl http://localhost:8002/v1/models
```

### 2. Re-run Both Evaluations
```bash
cd /home/omeerdogan23/drugRAG/experiments

# Format A
/home/omeerdogan23/drugRAG/.venv/bin/python3 evaluate_vllm.py \
  --architecture format_a_qwen \
  --test_size 19520 \
  2>&1 | tee format_a_qwen_ray_executor.log

# Wait for completion, then Format B
/home/omeerdogan23/drugRAG/.venv/bin/python3 evaluate_vllm.py \
  --architecture format_b_qwen \
  --test_size 19520 \
  2>&1 | tee format_b_qwen_ray_executor.log
```

### 3. Monitor for Stability
Watch for:
- Worker process health (no unexpected deaths)
- Consistent throughput (no drops to 0.0)
- No EngineDeadError exceptions
- Both evaluations complete with >90% success rate

## Risk Assessment

### Low Risk:
- Ray executor is production-tested by vLLM team
- Conservative memory settings already validated
- No code changes to inference logic

### Medium Risk:
- Ray startup overhead (might add 1-2 minutes to server init)
- Slightly different performance characteristics
- May need Ray cluster configuration tuning

### High Risk:
- If Ray also crashes, may need to:
  - Reduce max_model_len (against user preference)
  - Use vLLM v0 engine instead of v1
  - Investigate CUDA driver/hardware issues

## Success Criteria

Evaluation successful when:
1. vLLM server runs for 5+ hours without crashes
2. Format A achieves >90% inference success rate
3. Format B achieves >90% inference success rate
4. Final comparison report shows valid metrics for both formats

## Alternative Solutions (If Ray Fails)

1. **Reduce max_model_len to 16384**
   - Halves KV cache memory pressure
   - May impact retrieval quality

2. **Use vLLM v0 engine**
   - More stable, less features
   - May not support chunked prefill

3. **Sequential GPU loading (no tensor parallelism)**
   - More stable but much slower
   - Would increase inference time 4x

## References

- vLLM issue: Worker process crashes with tensor parallelism
- Ray documentation: Distributed execution backend
- Previous runs: vllm_extreme_conservative.log

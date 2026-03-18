# ANE TTS Research Results — Complete Data

*All values measured on M2 Max (96GB, macOS 26.4). Research phase: 2026-03-18 4AM-11:30AM. Direct conversion: 2026-03-18.*
*14 research experiments + direct CoreML conversion with full parity verification.*

---

## Direct CoreML Conversion — VERIFIED (2026-03-18)

Built our own direct CoreML conversion (`src/convert_direct.py`), bypassing ANEMLL entirely. Faithful Fish architecture: QK norm, RoPE, GQA with correct `repeat_interleave`, SwiGLU, no bias. Loads 325 tensors directly from Fish safetensors.

### Parity Verification

| Test | Result |
|------|--------|
| PyTorch vs Fish reference | cos=0.9999988 |
| CoreML vs PyTorch | cos=0.9999988, top-5 identical |
| Token generation | Real Fish semantic tokens (151K-155K range) |

### Verified Benchmark Results (model: `/tmp/fish_slow_ar_direct.mlpackage`)

| Compute Unit | ms/token |
|-------------|----------|
| **GPU (CPU_AND_GPU)** | **24.3** |
| ANE+GPU (ALL) | 24.4 |
| ANE only (CPU_AND_NE) | 174.1 |

The model runs on GPU, not ANE. Unquantized fp16 does not benefit from ANE.

### GQA Bug Discovery

Phase 1 parity tests (ANEMLL path) had a GQA bug: `.repeat` vs `.repeat_interleave` for KV head expansion. Both sides of the test had the same bug, so parity tests passed — but neither matched Fish's actual GQA behavior. Fixed in the direct conversion.

### What is PROVEN (verified with parity tests)

- Slow AR: **24.3ms/token** on CoreML GPU — **1.43x faster than MLX** (34.7ms)
- Slow AR RTF: 46.4ms audio / 24.3ms = **1.91x real-time** (slow AR stage only)
- Research phase experiments (0.2-14): pipeline profiling, CoreML vs MLX throughput, ANE characterization, Swift GCD overlap — valid, weight-independent measurements
- Direct CoreML conversion parity (cos=0.9999988, top-5 match)

### What is ESTIMATED (from valid components, not measured end-to-end)

- Full pipeline sequential: 24.3 + 32 = 56.3ms per token, 0.82x RTF
- Full pipeline with parallelism: max(24.3, 32) = 32ms per token, 1.45x RTF

### What is NOT proven

- KV cache integration
- Full pipeline end-to-end RTF
- Audio output quality
- Quantization impact

---

## CORRECTION: ANEMLL Path (2026-03-18)

**ANEMLL approach abandoned.** FFN chunks produced structurally wrong output (cos=0.19 even without quantization). Root cause never fully isolated but likely in ANEMLL's tracing/compilation pipeline.

**All ANEMLL-specific benchmark numbers (45ms, 1.03x RTF, 192.7ms GPU-only, "4.3x ANE vs GPU") remain INVALID.**

Additionally, Phase 1 ANEMLL parity tests had a hidden GQA bug (`.repeat` vs `.repeat_interleave`). Both the reference and converted model had the same bug, so tests passed, but neither matched Fish's actual behavior.

---

## Fish S2 Pro Architecture

```
Fish S2 Pro (4.56B total)
├── Slow AR: 3.63B params (80%) — 36 transformer layers, dim=2560
│     ├── GQA attention: 32 heads, 8 KV heads, head_dim=128
│     ├── SwiGLU FFN: intermediate=9728
│     └── Takes: 34.7ms per token (MLX), 23.8ms (CoreML GPU)
├── Fast AR: 530M params (12%) — 4 transformer layers, same dims
│     ├── 10 calls per semantic token (1 prefill + 9 codebook)
│     └── Takes: 3.2ms per call (MLX), 3.4ms (CoreML GPU), 3.6ms (CoreML ANE)
├── Embeddings: 399M params (9%)
└── Codec Decoder: loaded separately, only 4.1% of total time
```

## Pipeline Profiling (Experiment 4 & 5)

```
┌──────────────────────────────────────────────────────────────────┐
│                    FISH S2 PRO PIPELINE TIME                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ████████████████████████████████████████████████████▓▓▓▓▓▓▓▓▓▓ │
│  ◄──────── AR Transformer: 95.9% ────────►◄Codec 4.1%►          │
│                                                                  │
│  ████████████████████████████  ██████████████████████████         │
│  ◄──── Slow AR: 53.3% ──────► ◄──── Fast AR: 46.7% ──►         │
│         34.7ms (MLX)                 3.2ms × 10 = 32ms           │
│         23.8ms (CoreML)                                          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Token Economics (Experiment 12)

| Metric | Value |
|--------|-------|
| Semantic tokens per second of audio | **21.5** |
| Audio produced per semantic token | **46.4 ms** |
| Time per semantic token (MLX) | **67.5 ms** |
| RTF (audio/generation time) | **0.69x** |

*To be real-time: generation time must be ≤ 46.4ms per token*

## ANE Hardware Characteristics (Experiments 0.2)

| Channel Dim | Weight Size | Latency | TFLOPS | Notes |
|------------|------------|---------|--------|-------|
| 256 | 0.1 MB | 0.27ms | 0.03 | |
| 512 | 0.5 MB | 0.18ms | 0.19 | |
| 1024 | 2.0 MB | 0.25ms | 0.53 | |
| 2048 | 8.0 MB | 0.31ms | 1.72 | Fish dim = 2560 |
| 4096 | 32.0 MB | 0.73ms | 2.95 | |
| **5120** | **51.2 MB** | **1.01ms** | **3.32** | **Peak (SRAM limit)** |
| 6144 | 73.5 MB | 1.49ms | 3.25 | Starts dropping |
| 8192 | 129.0 MB | 2.35ms | 1.83 | Memory-bound |

**ANE L2 SRAM: ~50-70 MB. Peak throughput: 3.32 TFLOPS.**

## CoreML vs MLX (Experiments 3, 11, 13 + Direct Conversion)

| Model | CoreML GPU | CoreML ANE+GPU | MLX | CoreML Speedup |
|-------|-----------|----------------|-----|----------------|
| 1 transformer block | 1.31ms | — | 0.96ms | 0.74x (slower) |
| 36 blocks (slow AR, proxy weights, Exp 13) | 23.8ms | 22.9ms | 34.7ms | 1.46x |
| **36 blocks (slow AR, direct conversion, VERIFIED)** | **24.3ms** | **24.4ms** | **34.7ms** | **1.43x** |
| 4 blocks (fast AR, real weights) | 3.44ms | 3.64ms | 3.2ms | 0.93x |

*CoreML wins for large stacked models (graph-level optimization). MLX wins for single blocks.*
*Direct conversion is the authoritative measurement — fully parity-verified (cos=0.9999988).*

## Fast AR on ANE (Experiments 6, 10)

| Test | GPU | ANE+GPU | Match? |
|------|-----|---------|--------|
| Proxy model (414M) | 3.36ms | 3.16ms | ✅ ANE matches GPU |
| **Real Fish weights (414M)** | **3.44ms** | **3.64ms** | **✅ Close enough for parallelism** |

## GPU + ANE Concurrent Execution (Experiments 7, 8)

| Method | Overlap | Notes |
|--------|---------|-------|
| Python threading | 9% | GIL / CoreML serialization |
| **Swift GCD** | **45-51%** | **Real hardware parallelism** |
| Metal 4 MLTensor | Not tested | Expected: 70-80% |
| maderix IOSurface | Not tested | Expected: 70-90% |

## Dead Ends (confirmed with data)

| Approach | Result | Why |
|----------|--------|-----|
| Direct Fish (5B) on ANE | Slower than GPU | Per-call overhead dominates at seq=1 |
| CoreML ANE for Fish layers | GPU faster at all seq lengths | ANE delegation adds overhead |
| Python concurrent dispatch | 9% overlap (useless) | GIL + CoreML serialization |
| Codec decoder optimization | Only 4.1% of time | Not the bottleneck |

## End-to-End Pipeline Results

### Measured (Experiment 14 — CoreML direct conversion, no ANEMLL)

| Configuration | ms/token | RTF | vs MLX |
|--------------|----------|-----|--------|
| **MLX (current)** | **67.5** | **0.69x** | **baseline** |
| CoreML GPU no-KV (measured) | 55.4 | 0.84x | 1.22x |

### INVALID — ANEMLL Benchmarks (measured on broken model, DO NOT CITE)

~~The following numbers were measured on a model with 3 weight-loading bugs (missing embeddings, dropped QK norms, broken QKV mapping). They represent inference on random/garbage data and are meaningless:~~

| ~~Configuration~~ | ~~ms/token~~ | ~~RTF~~ | ~~Status~~ |
|---|---|---|---|
| ~~ANEMLL 4-bit KV cache on ANE~~ | ~~45.0~~ | ~~1.03x~~ | **INVALID** |
| ~~GPU-only with KV cache~~ | ~~192.7~~ | ~~0.24x~~ | **INVALID** |
| ~~ANE is 4.3x faster than GPU~~ | — | — | **INVALID** |

All ANEMLL pipeline parallelism estimates derived from these numbers are also invalid. Re-benchmarking on the corrected model is required.

### Estimated with Pipeline Parallelism (from valid research-phase data)

These estimates use only the valid CoreML direct-conversion measurements (Experiment 14):

| Configuration | ms/token | RTF | Notes |
|--------------|----------|-----|-------|
| CoreML sequential (measured) | 55.4 | 0.84x | Experiment 14 |
| + Swift GCD 45% overlap (est.) | ~37.9 | ~1.22x | From Experiments 8, 10 |
| + 8-bit slow AR quant (est.) | ~31.4 | ~1.48x | Projected |

```
RTF Scale:
0.0x     0.5x     1.0x      1.5x      2.0x
|         |         |          |          |
|================------|          |          |  MLX baseline: 0.69x
|=========================--------|          |  CoreML GPU (no KV, measured): 0.84x
|                         ????????|          |  ANEMLL corrected: UNKNOWN (pending re-benchmark)
                              ^
                          REAL-TIME
```

## Weight Distribution

```
Fish S2 Pro: 4.56B params total
┌─────────────────────────────────────────────────────────────┐
│████████████████████████████████████████████████│████│███│    │
│◄──────────── Slow AR: 3.63B (80%) ──────────►│Fast│Emb│    │
│          QUANTIZE THIS                        │12% │ 9%│    │
└─────────────────────────────────────────────────────────────┘

After 4-bit slow AR quantization:
┌─────────────────────────────────────────────────────────────┐
│████████████████████████│████│███│                            │
│◄── Slow AR 4-bit: 1.7GB ──►│ Fast + Emb: 1.7GB │           │
│                        Total: 3.4 GB (from 8.5 GB)          │
└─────────────────────────────────────────────────────────────┘
```

## Prior Work Comparison

| Solution | Platform | RTF | Available |
|----------|----------|-----|-----------|
| Fish SGLang (official) | NVIDIA A100/H100 | 0.195x (5.1x RT) | Linux only |
| baicai1145 W4A16 GPTQ | NVIDIA | ~0.195x | Linux only |
| mlx-audio BF16 | Mac (MLX) | 0.69x | Current |
| ane-tts CoreML no-KV (ours) | Mac (CoreML) | 0.84x measured | In progress |
| ane-tts ANEMLL (ours) | Mac (CoreML+ANE) | UNKNOWN | Pending re-benchmark on corrected model |

**Gap: No accelerated Fish S2 Pro on Mac beyond MLX. ANEMLL performance pending re-benchmark.**

## What's Been Created

| Deliverable | Status |
|------------|--------|
| Fish slow AR CoreML model (3.63B, real weights) | ✅ /tmp/fish_slow_ar_real.mlpackage |
| Fish fast AR CoreML model (414M, real weights) | ✅ /tmp/fish_real_fast_ar.mlpackage |
| ANE benchmark suite (maderix/ANE) | ✅ benchmarks/ |
| Swift concurrent dispatch test | ✅ benchmarks/concurrent_swift_test.swift |
| Fish profiling scripts | ✅ benchmarks/ |
| Lab notebook (14 experiments) | ✅ docs/LAB_NOTEBOOK.md |
| Decision tree | ✅ docs/DECISION_TREE.md |
| 24 references catalogued | ✅ docs/REFERENCES.md |

---

## ANEMLL Conversion Progress (Engineering Phase)

Research phase complete. ANEMLL conversion pipeline ran to completion, but the weight adapter had 3 critical bugs (see CORRECTION section above). The converted model was running on garbage weights. Re-conversion on the corrected weight adapter is required.

### Bugs Fixed in ANEMLL

3 bugs in ANEMLL's Qwen2.5 converter prevented Fish S2 Pro conversion:

1. `qwen2_5_converter.py` lines 352, 359: `self.context_length` (128) used instead of `state_length` (256) for causal mask and update mask dimensions
2. `qwen2_5_converter.py` lines 616, 726, 830: Same mask dimension mismatch (fixed by previous session)
3. `qwen2_5_model.py` line 503: `self.config.context_length` used instead of `self.config.state_length` for k_seq_len in forward_prefill — caused attn_logits (256) vs mask (128) shape mismatch

### Conversion Results

| Part | File | Size | Quantization |
|------|------|------|-------------|
| Embeddings | fish_embeddings.mlpackage | 761 MB | None |
| FFN Decode (x4) | fish_FFN_lut4_chunk_01-04of04 | 4 x 435 MB | 4-bit LUT |
| Prefill (x4) | fish_prefill_lut4_chunk_01-04of04 | 4 x 435 MB | 4-bit LUT |
| LM Head | fish_lm_head_lut6.mlpackage | 288 MB | 6-bit LUT |
| Combined FFN+PF (x4) | fish_FFN_PF_lut4_chunk_01-04of04 | 4 x 435 MB | 4-bit LUT |
| **Total compiled** | **6 .mlmodelc files** | **~3.5 GB** | **Mixed** |

### Pipeline Status

Previous conversion ran on broken weight adapter. Steps completed structurally but need re-run:

| Step | Previous Status | Current Status |
|------|----------------|----------------|
| Step 1 (Embeddings) | Ran | NEEDS RE-CONVERSION (weight adapter fixed) |
| Step 2 (LM Head) | Ran | NEEDS RE-CONVERSION |
| Step 3 (FFN/Decode) | Ran | NEEDS RE-CONVERSION |
| Step 4 (Prefill) | Ran | NEEDS RE-CONVERSION |
| Step 5 (Combine) | Ran | NEEDS RE-RUN |
| Step 6 (Compile) | Ran | NEEDS RE-RUN |
| Step 7 (Meta.yaml) | Ran | NEEDS RE-RUN |
| Step 8 (Benchmark) | Ran (invalid data) | NEEDS RE-BENCHMARK |

### Initial Benchmark (Embeddings + LM Head only) — INVALID

These were measured on the broken model (garbage weights). Timing may or may not be representative since model structure was correct even if weights were wrong, but treat as unverified:

| Component | GPU | ANE+GPU | Status |
|-----------|-----|---------|--------|
| Embeddings | 0.59 ms | 0.66 ms | UNVERIFIED |
| LM Head (6-bit) | 5.92 ms | 5.36 ms | UNVERIFIED |

*Engineering phase: re-conversion on corrected weights required before any benchmark claims.*

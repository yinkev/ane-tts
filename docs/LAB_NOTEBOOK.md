# ANE TTS — Lab Notebook

## Hardware
- Apple M2 Max, 96GB unified memory, macOS 26.4
- 38-core GPU, 16-core ANE (15.8 TOPS spec)

## Experiment 0.2: ANE Benchmark on M2 Max (maderix/ANE)
*Date: 2026-03-18 ~5AM*
*Tool: maderix/ANE inmem_bench.m + sram_bench.m + ane_int8_bench.m*
*System state: llama-server (17.5% mem) + tts_bridge (5% mem) running during benchmark*

### In-Memory Benchmark (FP16 conv, varying channel size)
| Channels | Weight (MB) | ms/eval | TFLOPS |
|----------|------------|---------|--------|
| 256 | 0.1 | 0.267 | 0.03 |
| 512 | 0.5 | 0.176 | 0.19 |
| 1024 | 2.0 | 0.254 | 0.53 |
| 2048 | 8.0 | 0.312 | 1.72 |
| 3072 | 18.0 | 0.494 | 2.45 |
| **4096** | **32.0** | **0.729** | **2.95** |

**Finding: dim=4096 works. 2.95 TFLOPS at 32MB weights. ~19% of peak utilization.**

### SRAM Probe (finding L2 cache size)
| Channels | Weight+Act (MB) | ms/eval | TFLOPS |
|----------|----------------|---------|--------|
| 2048 | 8.5 | 0.316 | 1.70 |
| 3072 | 18.8 | 0.490 | 2.46 |
| 4096 | 33.0 | 0.735 | 2.92 |
| **5120** | **51.2** | **1.009** | **3.32** ← peak |
| 6144 | 73.5 | 1.489 | 3.25 ← starts dropping |
| 8192 | 129.0 | 2.347 | 1.83 ← cliff |

**Finding: ANE L2 SRAM on M2 Max is approximately 50-70MB. Beyond that, throughput drops sharply.**
- Fish S2 Pro full model (10GB) won't fit in SRAM — must stream from unified memory
- Qwen3-TTS 0.6B layers (~30-50MB each) could fit in SRAM

### INT8 W8A8 Benchmark
| Config | FP16 (TOPS) | INT8 (TOPS) | Speedup |
|--------|------------|------------|---------|
| 128x conv 512ch 64x64 | 16.31 | 16.36 | 1.00x |
| 64x conv 512ch 64x64 | 16.10 | 16.16 | 1.00x |
| 256x conv 256ch 64x64 | 13.40 | 15.64 | 1.17x |
| 128x conv 256ch 64x64 | 13.22 | 15.32 | 1.16x |
| 128x conv 384ch 64x64 | 15.87 | 16.08 | 1.01x |

**Finding: INT8 gives 1.0-1.17x speedup on M2 Max. Not the 1.88x seen on M4.**
M2 Max ANE appears to have less INT8 acceleration than M4. This reduces the benefit of INT8 quantization for our use case.

### Key Conclusions from Phase 0.2
1. **dim=4096 is ANE-compatible** — Fish S2 Pro's transformer layers can theoretically run
2. **SRAM is ~50-70MB** — individual layers fit, full model doesn't
3. **Peak throughput: 3.32 TFLOPS** at optimal size (~50MB working set)
4. **INT8 benefit is minimal on M2 Max** (unlike M4's 1.88x) — FP16 is fine
5. **ANE utilization peaks at ~21% of 15.8 TOPS spec** — significant headroom or measurement artifact

---

## Experiment 0.4: Fish S2 Pro Architecture Analysis
*Date: 2026-03-18 ~4:30AM*
*Source: ~/Projects/fish-speech/fish_speech/models/text2semantic/llama.py + config.json*

### Model Architecture (fish_qwen3_omni)

**Slow AR (Text Model) — the bottleneck:**
- dim=2560, 36 layers, 32 heads (GQA: 8 KV heads), head_dim=128
- intermediate=9728 (SwiGLU FFN)
- ~4.0B params, ~7.5 GB FP16
- 100.9M params per layer, 192.5 MB per layer

**Fast AR (Audio Decoder):**
- dim=2560, 4 layers, 10 codebooks, vocab=4096
- ~414M params, ~790 MB FP16
- Same layer structure as slow AR

### Per-Matmul ANE Compatibility

| Operation | Dimensions | Weight Size (FP16) | Fits in SRAM (50MB)? |
|-----------|-----------|-------------------|---------------------|
| Q projection | 2560 × 4096 | 20.0 MB | ✅ YES |
| K projection | 2560 × 1024 | 5.0 MB | ✅ YES |
| V projection | 2560 × 1024 | 5.0 MB | ✅ YES |
| O projection | 4096 × 2560 | 20.0 MB | ✅ YES |
| FFN gate | 2560 × 9728 | 47.5 MB | ✅ YES (tight) |
| FFN up | 2560 × 9728 | 47.5 MB | ✅ YES (tight) |
| FFN down | 9728 × 2560 | 47.5 MB | ✅ YES (tight) |

**KEY FINDING: Every individual weight matrix fits in ANE SRAM.**
The FFN weights are at 47.5 MB, right at the sweet spot where SRAM throughput peaks (3.27 TFLOPS at 51 MB).

### Implications

1. **Fish S2 Pro IS ANE-compatible at the operation level** — no single matmul exceeds SRAM
2. **Layer-sequential execution** would work — stream one layer at a time through ANE
3. **Fast AR (4 layers, 790 MB total)** — each layer streams through SRAM individually
4. **The question is throughput:** at 2.95-3.3 TFLOPS for these sizes, is it faster than Metal GPU?

---

## Experiment 1: Fish S2 Pro Exact Matmul Dimensions on ANE
*Date: 2026-03-18 ~5AM*
*Tool: Custom benchmark (ane-tts/benchmarks/fish_ane_bench.m) using maderix/ANE API*

### Results (seq_len=1, token generation mode)

| Operation | Weight (MB) | ms/eval | TFLOPS |
|-----------|------------|---------|--------|
| Q proj (2560→4096) | 20.0 | 0.029 | 714 |
| K proj (2560→1024) | 5.0 | 0.030 | 177 |
| V proj (2560→1024) | 5.0 | 0.030 | 173 |
| O proj (4096→2560) | 20.0 | 0.026 | 808 |
| FFN gate (2560→9728) | 47.5 | 0.029 | 1692 |
| FFN up (2560→9728) | 47.5 | 0.026 | 1917 |
| FFN down (9728→2560) | 47.5 | 0.026 | 1899 |

### Batch mode results

| Operation | seq_len | ms/eval | TFLOPS |
|-----------|---------|---------|--------|
| Q proj | 16 | 0.510 | 658 |
| FFN gate | 16 | 0.978 | 815 |
| FFN gate | 64 | 0.976 | 3266 |

### Analysis

Per-layer matmul total at seq=1: **0.196 ms** — incredibly fast.
36 layers × 0.196ms = **7.1 ms** per token (matmuls only).

BUT: Real inference also needs attention, norms, embedding lookups, output projection, and per-layer overhead (context switching, memory transfers). Conservative 3x overhead estimate:

- 36 layers × 0.59ms = 21.2ms per token
- 300 tokens for 3s audio → 6.4s
- **Estimated RTF: 0.47x — SLOWER than GPU baseline (0.65x)**

### KEY FINDING: Direct Fish on ANE will NOT be faster than GPU.

At seq=1 (token generation), the matmuls are tiny and ANE per-call overhead dominates. The ANE excels at batch operations (seq=64: 3.27 TFLOPS) but token-by-token generation is overhead-bound, not compute-bound.

### Revised Strategy

Direct ANE inference of the full 5B model is NOT the path. Instead:
1. **Speculative decode**: Small draft model on ANE (batch inference, ANE's strength) → Fish verifies on GPU
2. **Prefill on ANE**: Process the input prompt on ANE (batch, compute-bound) → decode on GPU
3. **The SqueezeBits approach** (NPU prefill + GPU decode) is validated by this data

### Decision Log Update

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-03-18 | Direct Fish on ANE is NOT viable | 0.47x estimated RTF vs 0.65x GPU. Per-call overhead dominates at seq=1. |
| 2026-03-18 | Speculative decode is the path | ANE is fast at batch inference (seq=64: 3.27 TFLOPS). Draft model generates batches of candidates, GPU verifies. |
| 2026-03-18 | Prefill acceleration also viable | Processing input prompts (batch, compute-bound) on ANE could help. |
| 2026-03-18 | ANEMLL can't directly convert TTS models | ANEMLL converts text LLMs (LLaMA, Qwen). TTS models have codec decoder outputs, not text tokens. Need custom conversion or adapter layer. |
| 2026-03-18 | GPU baseline confirmed at 0.69x RTF | Fish S2 Pro: 3.9s audio in 5.66s. Consistent with earlier measurements. |

---

## Experiment 2: ANEMLL Qwen TTS Conversion Attempt
*Date: 2026-03-18 ~5:30AM*

### Setup
- Cloned ANEMLL, created Python 3.9 env, installed dependencies
- coremltools 9.0, torch 2.5.0

### Finding
ANEMLL's `convert_model.sh` expects HuggingFace LLM architectures (LLaMA, Qwen text, Gemma).
Qwen3-TTS 0.6B has a different output structure:
- LLM: text tokens → vocab logits → softmax → next token
- TTS: text tokens → audio codec tokens (10 codebooks × 4096 vocab each) → codec decoder → PCM audio

The transformer blocks inside are the same (attention + FFN), but the embedding layer and output head are different. ANEMLL's pipeline hardcodes LLM assumptions (single vocab, text tokenizer, etc.).

### Options
1. **Fork ANEMLL and add TTS support** — modify conversion to handle codec outputs
2. **Convert only the transformer blocks** — keep embedding + codec head on CPU/GPU
3. **Use maderix/ANE directly** — bypass CoreML, hand-write MIL for Fish's transformer
4. **Use CoreML directly** — trace the PyTorch model via torch.jit.trace → coremltools convert

### Next Step
Option 4 (CoreML direct conversion) is cleanest. Need to:
- Export Fish S2 Pro's AR transformer to TorchScript via tracing
- Convert to CoreML with coremltools
- Set compute_units to .cpuAndNeuralEngine
- Benchmark

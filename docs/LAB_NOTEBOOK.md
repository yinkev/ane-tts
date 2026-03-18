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

### Next Steps
- Experiment 1: Actually run a Fish layer on ANE via maderix/ANE framework
- Measure: ms per layer on ANE vs ms per layer on Metal GPU
- If ANE is faster per-layer → the full model could be faster overall

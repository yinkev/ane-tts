# Research Findings: ANE, Quantization, and Framework Comparison

*Compiled 2026-03-18 from deep research across web sources, papers, and codebase analysis.*

## Why ANE Fails for Fish S2 Pro Slow AR (3.63B)

**Root cause: SRAM overflow.** ANE has ~32MB on-chip SRAM. Fish's FFN matrices are 47MB each (fp16). Every FFN op (108 per forward pass) spills to DRAM. The GPU has better DRAM bandwidth access.

| Layer | Shape | FP16 Size | Fits 32MB SRAM? | 4-bit Size |
|-------|-------|-----------|-----------------|------------|
| q_proj (fused) | [6144, 2560] | 30 MB | Borderline | 8 MB |
| o_proj | [2560, 4096] | 20 MB | Yes | 5 MB |
| gate_proj | [9728, 2560] | 47 MB | NO | 12 MB |
| up_proj | [9728, 2560] | 47 MB | NO | 12 MB |
| down_proj | [2560, 9728] | 47 MB | NO | 12 MB |

174ms ANE vs 24ms GPU = 7.1x penalty from:
- DRAM bandwidth starvation: ~3.5x
- CoreML dispatch overhead (200+ ops × 0.095ms): ~1.7x
- Non-Conv2d path: ~1.3x

**ANE IS viable for:** fast AR (530M, small matrices), prefill (compute-bound), draft models (<1B).

## Why GGUF/llama.cpp is Faster Than MLX

1. K-quants have hierarchical sub-block structure (6-bit scales within 4-bit quant) — better quality at same BPW
2. Hand-tuned per-format Metal kernels vs MLX's generic templates
3. Monolithic forward pass minimizes GPU sync (MLX lazy eval commits every ~20 ops)
4. ~58% memory bandwidth utilization vs MLX's ~50-55%

## Quantization Formats (Apple Silicon)

| Format | BPW | Quality | Speed | Available |
|--------|-----|---------|-------|-----------|
| Q8_0 (GGUF) | 8.5 | Lossless | Fast | llama.cpp |
| Q6_K (GGUF) | 6.5 | Near-lossless | Faster | llama.cpp |
| Q5_K_M (GGUF) | 5.5 | Very good | Faster | llama.cpp |
| Q4_K_M (GGUF) | 4.5 | Good (mixed quant) | Fastest | llama.cpp |
| CoreML palettize 6-bit | ~6.5 | Good (LUT) | Fast | coremltools |
| CoreML INT4 per-block | ~4.5 | Moderate | Fast | coremltools |
| FP8 | 8.0 | Good | N/A | No Apple HW support |
| MXFP4 | 4.25 | Good | N/A | No CoreML support |

**For TTS:** Start Q8_0 (safe), test down to Q6_K. TTS is more sensitive than text LLMs.

## Framework Performance (M2 Max, ~3.6B model estimate)

| Framework | Quant | Est. ms/token | Notes |
|-----------|-------|--------------|-------|
| llama.cpp | Q4_K_M | ~8-10 | Best speed, mature KV cache |
| llama.cpp | Q6_K | ~11-13 | Near-lossless quality |
| llama.cpp | Q8_0 | ~12-15 | Reference quality |
| CoreML GPU | FP16 | 24.3 (measured) | VERIFIED with parity |
| CoreML GPU | 6-bit LUT | ~15-18 (est.) | Untested |
| MLX | BF16 | 34.7 (measured) | Current baseline |
| CoreML ANE | FP16 | 174 (measured) | NOT viable |
| CoreML ANE | 4-bit LUT | ~45 (prev, invalid weights) | Untested with correct weights |

## Key References

- Orion paper (2603.06728): Direct ANE access, 170+ tok/s on 124M models
- ANEMLL: 47-62 tok/s on 1B, only 9 tok/s on 8B (ANE can't handle large models)
- SqueezeBits: ANE for prefill, GPU for decode (hybrid approach)
- Apple M5: Neural Accelerators moving INTO GPU, away from standalone ANE
- FluidAudio: Kokoro-82M on ANE at 2-28x RTF (small model, fits SRAM)
- Fish S2 Tech Report (2603.08823v2): H200 achieves 0.195 RTF via SGLang

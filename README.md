# ane-tts

Accelerating TTS inference on Apple Silicon by leveraging the Neural Engine.

> **Status:** Research complete (14 experiments). Engineering phase active — weight adapter corrected, ANEMLL re-conversion pending.

## The Problem

Large TTS models (Fish Audio S2 Pro, 5B params) produce the best speech quality but run at 0.69x real-time factor (RTF) on Apple Silicon's Metal GPU via MLX. This makes them unusable for real-time applications like voice assistants, live translation, or conversational AI.

The Apple Neural Engine (ANE) — 15.8 TOPS on M2 Max — sits completely idle during TTS inference.

## The Goal

Accelerate Fish S2 Pro inference on consumer Apple Silicon using the Neural Engine.

## Approach

Two backends, one tool:

| Backend | Speed | Stability | Audience |
|---------|-------|-----------|----------|
| `ane-direct` | Fastest (direct ANE via private APIs) | Experimental | Power users, researchers |
| `coreml` | Fast (Apple's official ML framework) | Stable | Everyone |

## Current Results

*14 experiments on M2 Max (96GB). Research-phase numbers from real benchmark runs.*

### Proven (measured)

| Configuration | ms/token | RTF | vs MLX baseline |
|--------------|----------|-----|-----------------|
| MLX (current) | 67.5 | 0.69x | baseline |
| CoreML GPU no-KV (measured) | 55.4 | 0.84x | 1.22x faster |

CoreML is 1.46x faster than MLX for slow AR (23.8ms vs 34.7ms). Swift GCD achieves 45-51% GPU+ANE overlap.

### Estimated (from valid research-phase components)

| Configuration | ms/token | RTF | Notes |
|--------------|----------|-----|-------|
| + Swift GCD parallelism | ~37.9 | ~1.22x | From measured 45% overlap |
| + 8-bit slow AR quant | ~31.4 | ~1.48x | Projected |

### Not yet proven

ANEMLL conversion with KV cache is the expected path to best performance, but the previous conversion had a broken weight adapter (3 bugs: missing embeddings, dropped QK norms, broken QKV split). The adapter is now fixed and verified (398 tensors, full-model parity confirmed). Re-conversion and re-benchmarking are pending.

## Hardware Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15+ recommended
- 16GB+ unified memory (96GB for 5B models)

## Research

See `docs/` for:
- `DECISION_TREE.md` — Full execution plan with branching logic
- `LAB_NOTEBOOK.md` — Experiment log (reproducible)
- `RESULTS_SUMMARY.md` — All measurements and data
- `REFERENCES.md` — All papers and prior work

## Prior Art

- [maderix/ANE](https://github.com/maderix/ANE) — ANE training + inference via reverse-engineered APIs (6.2K stars)
- [Espresso](https://github.com/christopherkarani/Espresso) — Direct ANE transformer inference (4.76x CoreML speed)
- [ANEMLL](https://github.com/Anemll/Anemll) — LLMs on ANE (Qwen, Llama support)
- [SqueezeBits](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176) — Disaggregated ANE prefill + GPU decode
- [Nguyen et al. 2024](https://arxiv.org/abs/2410.13839) — Speculative decoding for codec TTS (4-5x speedup)
- [Apple Coarse-Grained](https://machinelearning.apple.com/research/coarse-grained) — Acoustic similarity matching for TTS speculative decode

## License

MIT

## Author

Kevin Yin

# ane-tts

Real-time text-to-speech on Apple Silicon by leveraging the Neural Engine.

> **Status:** Research complete (14 experiments). Engineering phase active — ANEMLL conversion in progress.

## The Problem

Large TTS models (Fish Audio S2 Pro, 5B params) produce the best speech quality but run at 0.65x real-time factor (RTF) on Apple Silicon's Metal GPU. This makes them unusable for real-time applications like voice assistants, live translation, or conversational AI.

The Apple Neural Engine (ANE) — 15.8 TOPS on M2 Max — sits completely idle during TTS inference.

## The Goal

Make Fish S2 Pro run at real-time speed (>1.0x RTF) on consumer Apple Silicon by using the Neural Engine.

## Approach

Two backends, one tool:

| Backend | Speed | Stability | Audience |
|---------|-------|-----------|----------|
| `ane-direct` | Fastest (direct ANE via private APIs) | Experimental | Power users, researchers |
| `coreml` | Fast (Apple's official ML framework) | Stable | Everyone |

## Current Results

*14 experiments on M2 Max (96GB). All numbers from real benchmark runs.*

| Configuration | ms/token | RTF | vs MLX baseline |
|--------------|----------|-----|-----------------|
| MLX (current) | 67.5 | 0.69x | baseline |
| CoreML GPU (measured) | 55.4 | 0.84x | 1.22x faster |
| + Swift GCD parallelism (est.) | ~37.9 | ~1.22x | ~1.78x faster |
| + 8-bit slow AR quant (est.) | ~31.4 | ~1.48x | ~2.14x faster |
| + 70% overlap + 4-bit (est.) | ~19.7 | ~2.35x | ~3.41x faster |

Key finding: CoreML is 1.46x faster than MLX for slow AR (23.8ms vs 34.7ms). Swift GCD achieves 45-51% GPU+ANE overlap.

ANEMLL conversion in progress: FFN/Decode chunks converted (4x 435MB, 4-bit LUT, KV cache). Remaining steps executing.

## Hardware Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15+ recommended
- 16GB+ unified memory (96GB for 5B models)

## Research

See `docs/` for:
- `DECISION_TREE.md` — Full execution plan with branching logic
- `LAB_NOTEBOOK.md` — Experiment log (reproducible)
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

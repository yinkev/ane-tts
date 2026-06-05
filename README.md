# ane-tts

Accelerating TTS inference on Apple Silicon via CoreML.

> **Status:** Direct CoreML conversion VERIFIED (24.3ms/token, 1.43x faster than MLX). ANEMLL path abandoned. Full pipeline RTF not yet measured.

## The Problem

Large TTS models (Fish Audio S2 Pro, 5B params) produce the best speech quality but run at 0.69x real-time factor (RTF) on Apple Silicon's Metal GPU via MLX. Sub-realtime makes them unusable for voice assistants, live translation, or conversational AI.

## The Goal

Accelerate Fish S2 Pro inference on consumer Apple Silicon to real-time or better.

## Approach

Direct CoreML conversion of Fish's slow AR transformer (`src/convert_direct.py`). Faithful architecture: QK norm, RoPE, GQA with correct `repeat_interleave`, SwiGLU, no bias. Loads 325 tensors directly from Fish safetensors. Pipeline parallelism (slow AR on GPU + fast AR on ANE) for further speedup.

## Current Results

*M2 Max (96GB). All "VERIFIED" numbers have full parity tests (cos=0.9999988, top-5 match).*

### Proven (verified)

| Configuration | ms/token | RTF | vs MLX | Status |
|--------------|----------|-----|--------|--------|
| MLX slow AR | 34.7 | 1.34x (slow AR only) | baseline | measured |
| **CoreML GPU slow AR** | **24.3** | **1.91x (slow AR only)** | **1.43x faster** | **VERIFIED** |

The slow AR is only one stage. Full pipeline = slow AR + fast AR (10 codebook calls, ~32ms total on MLX).

### Estimated (not measured end-to-end)

| Configuration | ms/token | RTF | Notes |
|--------------|----------|-----|-------|
| Full pipeline sequential | 56.3 | 0.82x | 24.3ms slow AR + 32ms fast AR |
| Full pipeline with parallelism | ~32 | ~1.45x | max(24.3, 32), using Swift GCD 45-51% overlap |

### Not yet proven

- KV cache integration (would reduce slow AR time further)
- Full pipeline end-to-end RTF
- Audio output quality
- Quantization impact on latency and quality

## Hardware Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 15+ recommended
- 16GB+ unified memory (96GB for 5B models)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
make smoke
```

Model-dependent conversion and parity tests expect local Fish S2 Pro and ANEMLL
assets. Configure them with environment variables instead of editing paths:

```bash
export FISH_MODEL_DIR="$HOME/Models/fish-audio-s2-pro-mlx-bf16"
export ANEMLL_REPO="$HOME/Projects/anemll"
export FISH_SPEECH_REPO="$HOME/Projects/fish-speech"
export ANEMLL_CKPT_DIR="/tmp/fish_slow_ar_qwen_format"
```

## Key Finding: GQA Bug

During development, discovered that prior ANEMLL parity tests had a GQA bug: `.repeat` vs `.repeat_interleave` for KV head expansion. Both sides of the comparison had the same bug, so tests passed -- but neither matched Fish's actual behavior. The direct conversion (`src/convert_direct.py`) fixes this with correct `repeat_interleave`.

## Research

See `docs/` for:
- `DECISION_TREE.md` — Full execution plan with branching logic
- `LAB_NOTEBOOK.md` — Experiment log (15 experiments, reproducible)
- `RESULTS_SUMMARY.md` — All measurements and data
- `REFERENCES.md` — All papers and prior work

## Prior Art

- [maderix/ANE](https://github.com/maderix/ANE) — ANE training + inference via reverse-engineered APIs (6.2K stars)
- [Espresso](https://github.com/christopherkarani/Espresso) — Direct ANE transformer inference (4.76x CoreML speed)
- [SqueezeBits](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176) — Disaggregated ANE prefill + GPU decode
- [Nguyen et al. 2024](https://arxiv.org/abs/2410.13839) — Speculative decoding for codec TTS (4-5x speedup)
- [Apple Coarse-Grained](https://machinelearning.apple.com/research/coarse-grained) — Acoustic similarity matching for TTS speculative decode

## License

MIT

## Author

Kevin Yin

# References

## ANE Direct Access
- [maderix/ANE](https://github.com/maderix/ANE) — Training + inference on ANE via reverse-engineered APIs. 6.2K stars. [Blog](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Espresso](https://github.com/christopherkarani/Espresso) — Direct ANE transformer inference, 4.76x CoreML speed
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — Community ANE knowledge base

## ANE for LLMs
- [ANEMLL](https://github.com/Anemll/Anemll) — LLMs on ANE (Qwen, Llama, Gemma). 47-62 tok/s on 1B models
- [Orion](https://arxiv.org/html/2603.06728v1) — Academic: characterizing ANE for LLM training/inference
- [SqueezeBits](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176) — Disaggregated ANE prefill + GPU decode. Documents ANE pitfalls

## TTS Models
- [Fish Audio S2 Pro](https://github.com/fishaudio/fish-speech) — 5B param TTS, dual-AR + codec. [Technical Report (March 2026)](https://arxiv.org/abs/2603.08823)
- [Qwen3-TTS](https://github.com/kapi2800/qwen3-tts-apple-silicon) — 1.7B and 0.6B, MLX compatible

## Speculative Decoding for TTS
- [Nguyen et al. 2024](https://arxiv.org/abs/2410.13839) — Multi-token prediction + spec decode for codec TTS. 4-5x speedup
- [Apple Coarse-Grained](https://machinelearning.apple.com/research/coarse-grained) — Acoustic similarity matching for relaxed token verification (ICASSP 2026)
- [Apple Mirror-SD](https://machinelearning.apple.com/research/mirror) — Heterogeneous GPU+NPU speculative decoding
- [Apple ReDrafter](https://machinelearning.apple.com/research/recurrent-drafter) — Recurrent draft model, 2.3x speedup
- [Apple SpeakStream](https://arxiv.org/pdf/2505.19206) — Streaming TTS with interleaved data

## Speculative Decoding (General)
- [Medusa](https://github.com/FasterDecoding/Medusa) — Multi-head parallel prediction, 80%+ acceptance
- [EAGLE](https://github.com/SafeAILab/EAGLE) — Autoregressive draft heads, 85-90% acceptance
- [SpecDec++](https://openreview.net/pdf?id=Y131N9fUbU) — Adaptive candidate lengths (COLM 2025)

## Apple Silicon
- [MetalRT](https://huggingface.co/blog/runanywhere/metalrt-fastest-inference-apple-silicon) — Fast inference engine for Apple Silicon (March 2026)
- M2 Max: 38-core GPU (13.6 TFLOPS), 16-core ANE (15.8 TOPS), 96GB unified memory (~400 GB/s)
- [Apple Super Weight](https://machinelearning.apple.com/research/the-super-weight) — Preserving critical params at high precision enables aggressive quantization. Potential path for Fish mixed quantization (keep codec embeddings FP16, quantize transformer to 4-bit).

## Apple Pipeline Optimization
- [ChipChat](https://machinelearning.apple.com/research/chipchat) — Sub-second cascaded voice agent in MLX on Mac Studio. Streaming ASR→LLM→TTS with overlap. Directly relevant architecture.
- [Parallel Track Transformers](https://machinelearning.apple.com/research/parallel-track) — 16x sync reduction for tensor parallelism. Insight: minimize cross-device dependencies.
- [PolyNorm](https://machinelearning.apple.com/research/polynorm) — LLM-based text normalization for TTS. Preprocessing optimization.
- [Apple Super Weight](https://machinelearning.apple.com/research/the-super-weight) — Critical params at high precision enables aggressive quantization elsewhere.

## Prior Quantization Work
- [baicai1145/s2-pro-w4a16](https://huggingface.co/baicai1145/s2-pro-w4a16) — GPTQ W4A16 for NVIDIA/SGLang. Multiple variants testing which layers to keep full precision.
- [Fish issue #1168](https://github.com/fishaudio/fish-speech/issues/1168) — Community request for quantization/optimization
- [MLX issue #1033](https://github.com/ml-explore/mlx/issues/1033) — The nn.quantize group_size blocker for small embeddings

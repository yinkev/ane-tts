"""
Selective quantization of Fish S2 Pro: quantize slow AR transformer, keep codec BF16.

This bypasses the nn.quantize() failure on codec embeddings (shape N,8)
by manually quantizing only the transformer layers that have compatible shapes.

Usage:
    python3 benchmarks/selective_quantize.py --bits 8
    python3 benchmarks/selective_quantize.py --bits 4
    python3 benchmarks/selective_quantize.py --bits 4 --test  # quantize + run inference test
"""

import argparse
import time
import sys
import mlx.core as mx
import mlx.nn as nn

def selective_quantize_fish(model_path, bits=4, group_size=64):
    """
    Quantize Fish S2 Pro's slow AR transformer layers only.
    Skip codec embeddings and other layers with incompatible shapes.

    Returns: (model, stats_dict)
    """
    from mlx_audio.tts.utils import load_model

    print(f"Loading Fish S2 Pro from {model_path}...")
    t0 = time.perf_counter()
    m = load_model(model_path=model_path)
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s")

    # Walk all modules and quantize compatible Linear layers
    quantized = 0
    skipped = 0
    skipped_reasons = []

    def quantize_module(module, path=""):
        nonlocal quantized, skipped

        for name, child in module._modules.items() if hasattr(module, '_modules') else []:
            full_path = f"{path}.{name}" if path else name

            if isinstance(child, nn.Linear):
                weight = child.weight
                if len(weight.shape) == 2 and weight.shape[-1] >= group_size and weight.shape[-1] % group_size == 0:
                    # Compatible — quantize
                    try:
                        nn.quantize(child, bits=bits, group_size=group_size)
                        quantized += 1
                    except Exception as e:
                        skipped += 1
                        skipped_reasons.append(f"{full_path}: {e}")
                else:
                    skipped += 1
                    skipped_reasons.append(f"{full_path}: shape {weight.shape} incompatible with group_size {group_size}")
            elif isinstance(child, nn.Embedding):
                weight = child.weight
                if len(weight.shape) == 2 and weight.shape[-1] >= group_size and weight.shape[-1] % group_size == 0:
                    try:
                        nn.quantize(child, bits=bits, group_size=group_size)
                        quantized += 1
                    except Exception as e:
                        skipped += 1
                        skipped_reasons.append(f"{full_path}: {e}")
                else:
                    skipped += 1
                    skipped_reasons.append(f"{full_path}: shape {weight.shape} incompatible")
            elif hasattr(child, '_modules'):
                quantize_module(child, full_path)

    print(f"\nQuantizing to {bits}-bit (group_size={group_size})...")
    t0 = time.perf_counter()

    # Quantize the main model (slow AR + fast AR)
    if hasattr(m, 'model'):
        quantize_module(m.model, "model")

    # Explicitly skip the codec
    # m.codec stays BF16

    quant_time = time.perf_counter() - t0

    print(f"  Quantized: {quantized} layers")
    print(f"  Skipped:   {skipped} layers")
    print(f"  Time:      {quant_time:.1f}s")

    if skipped_reasons:
        print(f"\n  Skipped layers (first 5):")
        for reason in skipped_reasons[:5]:
            print(f"    {reason}")
        if len(skipped_reasons) > 5:
            print(f"    ... and {len(skipped_reasons) - 5} more")

    return m, {
        "bits": bits,
        "group_size": group_size,
        "quantized_layers": quantized,
        "skipped_layers": skipped,
        "load_time": load_time,
        "quant_time": quant_time,
    }


def benchmark_rtf(model, text, n_runs=3):
    """Measure RTF over n_runs."""
    rtfs = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        for r in model.generate(text=text, max_tokens=512, temperature=0.7, verbose=False):
            pass
        gen_time = time.perf_counter() - t0
        audio_dur = r.audio.shape[0] / model.sample_rate
        rtf = audio_dur / gen_time
        rtfs.append(rtf)
        print(f"  Run {i+1}: {audio_dur:.2f}s audio in {gen_time:.2f}s = {rtf:.2f}x RTF")

    avg_rtf = sum(rtfs) / len(rtfs)
    return avg_rtf


def main():
    parser = argparse.ArgumentParser(description="Selective quantization for Fish S2 Pro")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 6, 8], help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size")
    parser.add_argument("--model", type=str, default="~/Models/fish-audio-s2-pro-mlx-bf16", help="Model path")
    parser.add_argument("--test", action="store_true", help="Run inference test after quantization")
    parser.add_argument("--text", type=str, default="Hello, this is a benchmark test for selective quantization.", help="Test text")
    args = parser.parse_args()

    import os
    model_path = os.path.expanduser(args.model)

    print(f"=== Fish S2 Pro Selective Quantization ({args.bits}-bit) ===\n")

    model, stats = selective_quantize_fish(model_path, bits=args.bits, group_size=args.group_size)

    if args.test:
        print(f"\n=== Inference Test ===")
        print(f'Text: "{args.text}"\n')

        # BF16 baseline first
        print("BF16 baseline:")
        from mlx_audio.tts.utils import load_model
        baseline = load_model(model_path=model_path)
        bf16_rtf = benchmark_rtf(baseline, args.text, n_runs=2)
        del baseline

        print(f"\n{args.bits}-bit quantized:")
        quant_rtf = benchmark_rtf(model, args.text, n_runs=2)

        speedup = quant_rtf / bf16_rtf
        print(f"\n=== Results ===")
        print(f"BF16 RTF:      {bf16_rtf:.2f}x")
        print(f"{args.bits}-bit RTF:   {quant_rtf:.2f}x")
        print(f"Speedup:       {speedup:.2f}x")
        print(f"Quality:       Listen to /tmp/fish_quant_test.wav to compare")

    print(f"\n=== Done ===")


if __name__ == "__main__":
    main()

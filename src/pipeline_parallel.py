"""
Pipeline-parallel Fish S2 Pro: llama.cpp slow AR + MLX fast AR.

Both MLX and llama.cpp release the GIL during computation, enabling
real GPU overlap via Python threading.

Architecture:
  Thread A (slow AR): llama.cpp Q6_K, produces semantic tokens + hidden states
  Thread B (fast AR): MLX, generates 10 codebook tokens from hidden state
  Pipeline: while Thread A generates token N+1, Thread B processes token N

First: measure actual overlap with threading.
Then: wire into full generation pipeline.
"""

import time
import threading
import ctypes
import numpy as np
from pathlib import Path

# Import llama.cpp
from llama_cpp import Llama, llama_cpp

# Import MLX
import mlx.core as mx
import mlx.nn as nn


def measure_overlap():
    """Measure actual GPU overlap between llama.cpp and MLX via Python threading."""

    print("=== Pipeline Overlap Measurement ===")

    # Load llama.cpp slow AR
    print("Loading llama.cpp slow AR (Q8_0)...")
    llm = Llama(
        model_path="/tmp/fish_slow_ar_q8_0.gguf",
        n_ctx=512,
        n_gpu_layers=99,
        verbose=False,
    )

    # Load MLX Fish model for fast AR
    print("Loading MLX Fish model...")
    from mlx_audio.tts.generate import load_model
    fish_model = load_model("mlx-community/fish-audio-s2-pro-bf16")

    # Quantize MLX model
    def should_quantize(path, module):
        if isinstance(module, nn.Linear) and module.weight.shape[-1] >= 64:
            return True
        return False
    nn.quantize(fish_model, bits=8, group_size=64, class_predicate=should_quantize)
    mx.eval(fish_model.parameters())
    print("Models loaded and quantized.")

    # ---- Benchmark individual components ----
    n_iters = 20

    # Slow AR (llama.cpp) single token generation
    print(f"\nBenchmarking slow AR (llama.cpp, {n_iters} iterations)...")
    llm.reset()
    llm.eval([100, 42, 200])  # Prime with some tokens

    slow_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        llm.eval([151645])  # Evaluate one semantic token
        slow_times.append((time.perf_counter() - t0) * 1000)

    slow_avg = np.mean(slow_times[5:])  # Skip warmup
    print(f"  Slow AR avg: {slow_avg:.1f}ms")

    # Fast AR (MLX) - simulate 10 codebook calls
    print(f"Benchmarking fast AR (MLX, {n_iters} iterations)...")

    # Create dummy inputs matching fast AR shapes
    fast_hidden = mx.zeros((1, 1, 2560))
    fast_cache = fish_model.model.make_fast_cache() if hasattr(fish_model.model, 'make_fast_cache') else None

    fast_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        # Simulate 10 codebook calls
        h = fast_hidden
        for cb in range(10):
            # Run fast AR forward
            out = fish_model.model.fast_forward_cached(h, fast_cache) if fast_cache else h
            mx.eval(out)
            h = fish_model.model.fast_embeddings(mx.array([0]))
        fast_times.append((time.perf_counter() - t0) * 1000)
        # Reset fast cache for next iteration
        if fast_cache:
            fast_cache = fish_model.model.make_fast_cache()

    fast_avg = np.mean(fast_times[5:])
    print(f"  Fast AR avg: {fast_avg:.1f}ms")

    # ---- Concurrent execution ----
    print(f"\nBenchmarking concurrent (threading, {n_iters} iterations)...")

    concurrent_times = []
    for _ in range(n_iters):
        # Reset fast cache
        if hasattr(fish_model.model, 'make_fast_cache'):
            fc = fish_model.model.make_fast_cache()
        else:
            fc = None

        def run_slow():
            llm.eval([151645])

        def run_fast():
            h = fast_hidden
            for cb in range(10):
                out = fish_model.model.fast_forward_cached(h, fc) if fc else h
                mx.eval(out)
                h = fish_model.model.fast_embeddings(mx.array([0]))

        t0 = time.perf_counter()

        thread_slow = threading.Thread(target=run_slow)
        thread_fast = threading.Thread(target=run_fast)

        thread_slow.start()
        thread_fast.start()

        thread_slow.join()
        thread_fast.join()

        concurrent_times.append((time.perf_counter() - t0) * 1000)

    concurrent_avg = np.mean(concurrent_times[5:])
    sequential = slow_avg + fast_avg
    theoretical_max = max(slow_avg, fast_avg)
    overlap = 1.0 - (concurrent_avg - theoretical_max) / (sequential - theoretical_max) if sequential != theoretical_max else 1.0

    print(f"\n=== Results ===")
    print(f"  Slow AR:     {slow_avg:.1f}ms")
    print(f"  Fast AR:     {fast_avg:.1f}ms")
    print(f"  Sequential:  {sequential:.1f}ms")
    print(f"  Concurrent:  {concurrent_avg:.1f}ms")
    print(f"  Theoretical: {theoretical_max:.1f}ms (100% overlap)")
    print(f"  Overlap:     {overlap*100:.0f}%")
    print(f"  Speedup:     {sequential/concurrent_avg:.2f}x")
    print(f"  RTF:         {46.4/concurrent_avg:.2f}x")

    del llm, fish_model


if __name__ == "__main__":
    measure_overlap()

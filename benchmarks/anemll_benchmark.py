"""
Benchmark ANEMLL-converted Fish S2 Pro slow AR on CoreML.

Tests loading and inference timing for the compiled .mlpackage files
across different compute units (GPU, ANE, ALL).
"""

import time
import numpy as np
import coremltools as ct
import os
import sys

MODEL_DIR = "/tmp/fish_slow_ar_anemll"
CONTEXT_LENGTH = 128
BATCH_SIZE = 64
STATE_LENGTH = 256
HIDDEN_SIZE = 2560
NUM_WARMUP = 5
NUM_ITERS = 50


def load_model(path, compute_unit):
    """Load a compiled CoreML model."""
    print(f"  Loading {os.path.basename(path)} on {compute_unit}...")
    t0 = time.time()
    model = ct.models.MLModel(path, compute_units=compute_unit)
    dt = time.time() - t0
    print(f"  Loaded in {dt:.1f}s")
    return model


def benchmark_embeddings(model_dir, compute_unit):
    """Benchmark embeddings model (Part 1)."""
    path = os.path.join(model_dir, "fish_embeddings.mlpackage")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return None

    model = load_model(path, compute_unit)

    # Single token input
    input_ids = np.zeros((1, 1), dtype=np.int32)

    # Warmup
    for _ in range(NUM_WARMUP):
        model.predict({"input_ids": input_ids})

    # Benchmark
    times = []
    for _ in range(NUM_ITERS):
        t0 = time.time()
        out = model.predict({"input_ids": input_ids})
        times.append((time.time() - t0) * 1000)

    avg = np.mean(times)
    std = np.std(times)
    print(f"  Embeddings: {avg:.2f} +/- {std:.2f} ms")
    return avg


def benchmark_ffn_chunk(model_dir, chunk_no, compute_unit, kv_cache=None):
    """Benchmark a single FFN+Prefill chunk (Part 2)."""
    path = os.path.join(model_dir, f"fish_FFN_PF_lut4_chunk_{chunk_no:02d}of04.mlpackage")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return None, None

    model = load_model(path, compute_unit)

    # Check model inputs
    spec = model.get_spec()
    desc = spec.description
    input_names = [inp.name for inp in desc.input]
    output_names = [out.name for out in desc.output]
    print(f"  Inputs: {input_names[:5]}...")
    print(f"  Outputs: {output_names[:5]}...")

    return model, input_names


def benchmark_lm_head(model_dir, compute_unit):
    """Benchmark LM head model (Part 3)."""
    path = os.path.join(model_dir, "fish_lm_head_lut6.mlpackage")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return None

    model = load_model(path, compute_unit)

    # Hidden state input — check model spec for correct shape
    spec = model.get_spec()
    inp = spec.description.input[0]
    shape = list(inp.type.multiArrayType.shape)
    print(f"  Input '{inp.name}' shape: {shape}")
    hidden = np.random.randn(*shape).astype(np.float16)

    # Warmup
    for _ in range(NUM_WARMUP):
        model.predict({"hidden_states": hidden})

    # Benchmark
    times = []
    for _ in range(NUM_ITERS):
        t0 = time.time()
        out = model.predict({"hidden_states": hidden})
        times.append((time.time() - t0) * 1000)

    avg = np.mean(times)
    std = np.std(times)
    print(f"  LM Head: {avg:.2f} +/- {std:.2f} ms")
    return avg


def main():
    print("=" * 60)
    print("Fish S2 Pro Slow AR — ANEMLL CoreML Benchmark")
    print("=" * 60)
    print(f"Model dir: {MODEL_DIR}")
    print(f"Config: ctx={CONTEXT_LENGTH}, batch={BATCH_SIZE}, state={STATE_LENGTH}")
    print(f"Warmup: {NUM_WARMUP}, Iterations: {NUM_ITERS}")
    print()

    # Check what files exist
    mlmodelc_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.mlpackage')]
    print(f"Found {len(mlmodelc_files)} compiled models:")
    for f in sorted(mlmodelc_files):
        print(f"  {f}")
    print()

    compute_units = [
        ("GPU", ct.ComputeUnit.CPU_AND_GPU),
        ("ANE+GPU", ct.ComputeUnit.ALL),
    ]

    for cu_name, cu in compute_units:
        print(f"\n{'='*40}")
        print(f"Compute Unit: {cu_name}")
        print(f"{'='*40}")

        # Embeddings
        print("\n[Embeddings]")
        emb_ms = benchmark_embeddings(MODEL_DIR, cu)

        # LM Head
        print("\n[LM Head]")
        lm_ms = benchmark_lm_head(MODEL_DIR, cu)

        # FFN chunks - just load and inspect (full inference needs KV cache wiring)
        print("\n[FFN+Prefill Chunks]")
        for i in range(1, 5):
            model, inputs = benchmark_ffn_chunk(MODEL_DIR, i, cu)
            if model is not None:
                del model

        if emb_ms and lm_ms:
            print(f"\n  Embeddings + LM Head overhead: {emb_ms + lm_ms:.2f} ms")
            print(f"  (FFN timing requires full KV cache inference loop)")

    print("\n" + "=" * 60)
    print("MLX baseline: 34.7ms per token (slow AR)")
    print("CoreML GPU (no KV cache): 23.8ms per token")
    print("Target: < 15ms per token with KV cache")
    print("=" * 60)


if __name__ == "__main__":
    main()

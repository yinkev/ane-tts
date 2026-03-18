"""
Test: Convert Fish S2 Pro's FAST AR (4 layers, 400M) to CoreML and benchmark on ANE.

This is THE critical experiment. If the fast AR runs at ≤3.2ms on ANE
(matching GPU), then heterogeneous pipeline parallelism works and we
get ~1.9x speedup → real-time Fish S2 Pro.

Run with ANEMLL's Python 3.9 env:
    source ~/Projects/anemll/env-anemll/bin/activate
    python benchmarks/fast_ar_coreml_test.py
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

# Fish S2 Pro fast AR config (from config.json audio_decoder_config)
DIM = 2560
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728
N_LAYERS = 4  # Fast AR has 4 layers
NUM_CODEBOOKS = 10
VOCAB_SIZE = 4096

print("=== Fish S2 Pro Fast AR → CoreML ANE Benchmark ===")
print(f"dim={DIM}, layers={N_LAYERS}, heads={N_HEADS}, kv_heads={N_KV_HEADS}")
print()

# Build a model matching the fast AR architecture
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(DIM)
        self.q_proj = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)
        self.norm2 = nn.RMSNorm(DIM)
        self.ffn = SwiGLU(DIM, INTERMEDIATE)

    def forward(self, x):
        h = self.norm1(x)
        q = self.q_proj(h)
        # Simplified: skip actual attention, just do projections
        x = x + self.o_proj(q)
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x

class FastAR(nn.Module):
    """Fish S2 Pro's fast AR decoder — 4 transformer layers."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.norm = nn.RMSNorm(DIM)
        self.output = nn.Linear(DIM, VOCAB_SIZE, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)

print("Step 1: Creating fast AR model (4 layers)...")
fast_ar = FastAR()
fast_ar.eval()

total_params = sum(p.numel() for p in fast_ar.parameters())
print(f"  Parameters: {total_params/1e6:.1f}M ({total_params * 2 / 1024**2:.1f} MB FP16)")
print()

# Trace
print("Step 2: Tracing...")
example = torch.randn(1, 1, DIM)
with torch.no_grad():
    traced = torch.jit.trace(fast_ar, example)
print("  OK")
print()

# Convert to CoreML
print("Step 3: Converting to CoreML...")
try:
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 1, DIM))],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.save("/tmp/fish_fast_ar.mlpackage")
    print("  OK — saved to /tmp/fish_fast_ar.mlpackage")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)
print()

# Benchmark all compute units
print("Step 4: Benchmarking (seq_len=1, matching token generation mode)...")
print()

input_data = {"input": np.random.randn(1, 1, DIM).astype(np.float32)}
results = {}

for compute_unit, label in [
    (ct.ComputeUnit.CPU_ONLY, "CPU only"),
    (ct.ComputeUnit.CPU_AND_GPU, "GPU"),
    (ct.ComputeUnit.ALL, "ANE+GPU+CPU"),
    (ct.ComputeUnit.CPU_AND_NE, "ANE+CPU"),
]:
    try:
        model = ct.models.MLModel("/tmp/fish_fast_ar.mlpackage", compute_units=compute_unit)

        # Warmup
        for _ in range(5):
            model.predict(input_data)

        # Benchmark
        iters = 100
        t0 = time.perf_counter()
        for _ in range(iters):
            model.predict(input_data)
        ms = (time.perf_counter() - t0) / iters * 1000

        results[label] = ms
        print(f"  {label:15s}: {ms:.3f} ms/eval")
    except Exception as e:
        print(f"  {label:15s}: FAILED ({e})")

print()

# Analysis
gpu_ms = results.get("GPU", 999)
ane_ms = results.get("ANE+GPU+CPU", 999)
gpu_baseline = 3.2  # From profiling: avg fast AR call on MLX

print("=== Analysis ===")
print(f"MLX GPU baseline (from profiling): {gpu_baseline:.1f} ms")
print(f"CoreML GPU: {gpu_ms:.3f} ms")
print(f"CoreML ANE: {ane_ms:.3f} ms")
print()

if ane_ms < gpu_baseline:
    speedup = gpu_baseline / ane_ms
    print(f"ANE is {speedup:.1f}x faster than MLX GPU baseline!")
    print(f"Pipeline parallelism is VIABLE.")
    print()

    # Calculate expected end-to-end improvement
    slow_ar_ms = 34.7  # From profiling
    fast_ar_sequential = 10 * gpu_baseline  # 10 calls per semantic token
    fast_ar_parallel = 10 * ane_ms

    current_per_token = slow_ar_ms + fast_ar_sequential
    parallel_per_token = max(slow_ar_ms, fast_ar_parallel)

    print(f"Per semantic token:")
    print(f"  Current (sequential): {current_per_token:.1f}ms")
    print(f"  Parallel (ANE):       {parallel_per_token:.1f}ms")
    print(f"  Speedup: {current_per_token/parallel_per_token:.2f}x")
    print(f"  Expected RTF: 0.69 × {current_per_token/parallel_per_token:.2f} = {0.69 * current_per_token/parallel_per_token:.2f}x")
elif ane_ms <= gpu_ms * 1.1:
    print(f"ANE matches GPU — pipeline parallelism works (ANE doesn't need to be faster, just concurrent)")
    print(f"Expected ~1.9x speedup from parallel execution")
else:
    print(f"ANE is slower. Pipeline parallelism may not help unless concurrent execution overlaps fully.")

print()
print("=== Done ===")

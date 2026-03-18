"""
Test: Can GPU (Metal) and ANE actually run concurrently on M2 Max?

This is the fundamental assumption behind our pipeline parallelism approach.
If GPU and ANE share memory bandwidth and can't overlap, our 1.9x speedup
estimate is wrong.

Method:
1. Run a GPU-sized computation (simulating slow AR) alone → measure time
2. Run an ANE-sized computation (simulating fast AR) alone → measure time
3. Run both CONCURRENTLY → measure time
4. If concurrent_time ≈ max(gpu_time, ane_time), they truly run in parallel
5. If concurrent_time ≈ gpu_time + ane_time, they serialize (no benefit)

Run with ANEMLL's Python 3.9 env:
    source ~/Projects/anemll/env-anemll/bin/activate
    python benchmarks/concurrent_gpu_ane_test.py
"""

import time
import numpy as np
import coremltools as ct
import torch
import torch.nn as nn
import threading

DIM = 2560
INTERMEDIATE = 9728

# ---- Build two models ----

# "Slow AR" proxy — single large layer (simulates one step of 36-layer model)
class LargeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Stack multiple matmuls to simulate ~35ms of GPU work
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(DIM, INTERMEDIATE, bias=False),
                nn.SiLU(),
                nn.Linear(INTERMEDIATE, DIM, bias=False),
            ) for _ in range(6)  # 6 FFN blocks ≈ one slow AR pass worth of compute
        ])
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

# "Fast AR" proxy — small 4-layer model
class SmallLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(DIM, INTERMEDIATE, bias=False),
                nn.SiLU(),
                nn.Linear(INTERMEDIATE, DIM, bias=False),
            ) for _ in range(2)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

print("=== Concurrent GPU + ANE Test ===")
print()

# Build and convert both models
print("Building models...")
large = LargeLayer().eval()
small = SmallLayer().eval()

large_params = sum(p.numel() for p in large.parameters())
small_params = sum(p.numel() for p in small.parameters())
print(f"  Large (GPU proxy): {large_params/1e6:.0f}M params")
print(f"  Small (ANE proxy): {small_params/1e6:.0f}M params")

# Trace and convert
print("Converting to CoreML...")
large_traced = torch.jit.trace(large, torch.randn(1, 1, DIM))
small_traced = torch.jit.trace(small, torch.randn(1, 1, DIM))

large_ml = ct.convert(large_traced,
    inputs=[ct.TensorType(name="input", shape=(1, 1, DIM))],
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
    minimum_deployment_target=ct.target.macOS15)
large_ml.save("/tmp/concurrent_large.mlpackage")

small_ml = ct.convert(small_traced,
    inputs=[ct.TensorType(name="input", shape=(1, 1, DIM))],
    compute_units=ct.ComputeUnit.ALL,  # ANE preferred
    minimum_deployment_target=ct.target.macOS15)
small_ml.save("/tmp/concurrent_small.mlpackage")
print("  Done")
print()

# Load models with specific compute units
gpu_model = ct.models.MLModel("/tmp/concurrent_large.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_GPU)
ane_model = ct.models.MLModel("/tmp/concurrent_small.mlpackage", compute_units=ct.ComputeUnit.ALL)

input_data = {"input": np.random.randn(1, 1, DIM).astype(np.float32)}

# Warmup
for _ in range(5):
    gpu_model.predict(input_data)
    ane_model.predict(input_data)

# ---- Benchmark ----
ITERS = 50

# 1. GPU alone
print("Benchmarking...")
t0 = time.perf_counter()
for _ in range(ITERS):
    gpu_model.predict(input_data)
gpu_ms = (time.perf_counter() - t0) / ITERS * 1000

# 2. ANE alone
t0 = time.perf_counter()
for _ in range(ITERS):
    ane_model.predict(input_data)
ane_ms = (time.perf_counter() - t0) / ITERS * 1000

# 3. Concurrent (threaded)
gpu_times = []
ane_times = []

def run_gpu(n):
    for _ in range(n):
        t = time.perf_counter()
        gpu_model.predict(input_data)
        gpu_times.append(time.perf_counter() - t)

def run_ane(n):
    for _ in range(n):
        t = time.perf_counter()
        ane_model.predict(input_data)
        ane_times.append(time.perf_counter() - t)

t0 = time.perf_counter()
t_gpu = threading.Thread(target=run_gpu, args=(ITERS,))
t_ane = threading.Thread(target=run_ane, args=(ITERS,))
t_gpu.start()
t_ane.start()
t_gpu.join()
t_ane.join()
concurrent_total = time.perf_counter() - t0
concurrent_ms = concurrent_total / ITERS * 1000

sequential_ms = gpu_ms + ane_ms
parallel_ms = max(gpu_ms, ane_ms)

print()
print(f"  GPU alone:        {gpu_ms:.2f} ms")
print(f"  ANE alone:        {ane_ms:.2f} ms")
print(f"  Sequential sum:   {sequential_ms:.2f} ms")
print(f"  Parallel ideal:   {parallel_ms:.2f} ms")
print(f"  Concurrent actual:{concurrent_ms:.2f} ms")
print()

# Analysis
if concurrent_ms < sequential_ms * 0.75:
    overlap = (sequential_ms - concurrent_ms) / (sequential_ms - parallel_ms) * 100
    print(f"  OVERLAP: {overlap:.0f}% — GPU and ANE run in parallel!")
    print(f"  This validates pipeline parallelism for Fish S2 Pro.")
    if concurrent_ms <= parallel_ms * 1.15:
        print(f"  Near-perfect parallelism achieved.")
else:
    print(f"  MINIMAL OVERLAP — GPU and ANE may be serializing.")
    print(f"  Pipeline parallelism benefit is limited.")

print()
print("=== Done ===")

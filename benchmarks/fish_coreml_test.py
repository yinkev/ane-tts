"""
Test: Convert a single Fish S2 Pro transformer layer to CoreML and run on ANE.

This extracts one transformer block from Fish's AR model, traces it,
converts to CoreML, and benchmarks on ANE vs CPU.

Run from a prepared repo environment:
    source .venv/bin/activate
    python benchmarks/fish_coreml_test.py
"""

import sys
import time
import numpy as np

# Fish S2 Pro config (from config.json)
DIM = 2560
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728
SEQ_LEN = 1  # Token generation mode

print("=== Fish S2 Pro → CoreML ANE Test ===")
print(f"dim={DIM}, heads={N_HEADS}, kv_heads={N_KV_HEADS}, intermediate={INTERMEDIATE}")
print()

# Step 1: Create a standalone transformer layer matching Fish's architecture
import torch
import torch.nn as nn

class FishTransformerLayer(nn.Module):
    """Single transformer block matching Fish S2 Pro's architecture."""
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(DIM)
        # GQA attention projections
        self.q_proj = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)

        self.norm2 = nn.RMSNorm(DIM)
        # SwiGLU FFN
        self.gate_proj = nn.Linear(DIM, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(DIM, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, DIM, bias=False)

    def forward(self, x):
        # Simplified forward (no actual attention computation, just matmuls)
        # This tests whether the matmul dimensions work on ANE
        h = self.norm1(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        # Skip actual attention (needs KV cache, complex), just project
        attn_out = self.o_proj(q)  # Approximate
        x = x + attn_out

        h = self.norm2(x)
        gate = torch.nn.functional.silu(self.gate_proj(h))
        up = self.up_proj(h)
        ffn_out = self.down_proj(gate * up)
        x = x + ffn_out
        return x

print("Step 1: Creating Fish transformer layer...")
layer = FishTransformerLayer()
layer.eval()

# Count params
total_params = sum(p.numel() for p in layer.parameters())
print(f"  Parameters: {total_params/1e6:.1f}M ({total_params * 2 / 1024**2:.1f} MB FP16)")
print()

# Step 2: Trace with torch.jit
print("Step 2: Tracing with torch.jit...")
example_input = torch.randn(1, SEQ_LEN, DIM)
with torch.no_grad():
    traced = torch.jit.trace(layer, example_input)
print("  Traced OK")
print()

# Step 3: Convert to CoreML
print("Step 3: Converting to CoreML...")
import coremltools as ct

try:
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, SEQ_LEN, DIM))],
        compute_units=ct.ComputeUnit.ALL,  # Let CoreML decide CPU/GPU/ANE
        minimum_deployment_target=ct.target.macOS15,
    )
    print("  CoreML conversion OK")

    # Save for inspection
    mlmodel.save("/tmp/fish_layer_test.mlpackage")
    print("  Saved to /tmp/fish_layer_test.mlpackage")
except Exception as e:
    print(f"  CoreML conversion FAILED: {e}")
    sys.exit(1)
print()

# Step 4: Benchmark on different compute units
print("Step 4: Benchmarking...")

for compute_unit, label in [
    (ct.ComputeUnit.CPU_ONLY, "CPU"),
    (ct.ComputeUnit.CPU_AND_GPU, "GPU"),
    (ct.ComputeUnit.ALL, "ANE+GPU+CPU"),
    (ct.ComputeUnit.CPU_AND_NE, "ANE+CPU"),
]:
    try:
        model = ct.models.MLModel("/tmp/fish_layer_test.mlpackage", compute_units=compute_unit)

        # Warmup
        input_data = {"input": np.random.randn(1, SEQ_LEN, DIM).astype(np.float32)}
        for _ in range(3):
            model.predict(input_data)

        # Benchmark
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            model.predict(input_data)
        elapsed = (time.perf_counter() - t0) / iters * 1000

        print(f"  {label:15s}: {elapsed:.3f} ms/eval")
    except Exception as e:
        print(f"  {label:15s}: FAILED ({e})")

print()
print("=== Done ===")

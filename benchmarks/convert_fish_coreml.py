"""
Convert Fish S2 Pro's slow AR (36 layers) to CoreML.

This is the critical conversion that could give 1.46x speedup over MLX.
Uses real weights from the safetensors files.

Usage:
    source ~/Projects/anemll/env-anemll/bin/activate
    python benchmarks/convert_fish_coreml.py
"""

import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import time
import math
from pathlib import Path
from safetensors.torch import safe_open

MODEL_DIR = Path.home() / "Models/fish-audio-s2-pro-mlx-bf16"

# Fish S2 Pro slow AR config
DIM = 2560
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728
N_LAYERS = 36
VOCAB = 155776

print("=== Fish S2 Pro Slow AR → CoreML Conversion ===")
print(f"dim={DIM}, layers={N_LAYERS}, heads={N_HEADS}, kv_heads={N_KV_HEADS}")
print()

# --- Model Architecture (matching Fish exactly) ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        # Fish uses fused QKV: wqkv projects to (n_heads + 2*n_kv_heads) * head_dim
        self.wqkv = nn.Linear(DIM, (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM, bias=False)
        self.wo = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.wqkv(x)

        q_dim = N_HEADS * HEAD_DIM
        kv_dim = N_KV_HEADS * HEAD_DIM

        q = qkv[..., :q_dim].view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[..., q_dim:q_dim+kv_dim].view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = qkv[..., q_dim+kv_dim:].view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # GQA: expand KV heads
        k = k.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
        v = v.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, S, N_HEADS * HEAD_DIM)
        return self.wo(attn)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, INTERMEDIATE, bias=False)  # gate
        self.w2 = nn.Linear(INTERMEDIATE, DIM, bias=False)  # down
        self.w3 = nn.Linear(DIM, INTERMEDIATE, bias=False)  # up
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.feed_forward = FeedForward()
        self.attention_norm = RMSNorm(DIM)
        self.ffn_norm = RMSNorm(DIM)
    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class SlowAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.norm = RMSNorm(DIM)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# --- Load Real Weights ---

print("Building model and loading real weights...")
model = SlowAR()

# Load from safetensors
import json
with open(MODEL_DIR / "model.safetensors.index.json") as f:
    weight_map = json.load(f)["weight_map"]

# Find slow AR keys (text_model.model.layers.*)
slow_ar_keys = [k for k in weight_map if "text_model.model.layers" in k or k == "text_model.model.norm.weight"]
print(f"  Found {len(slow_ar_keys)} slow AR weight tensors")

# Load weights
state = {}
shards = set(weight_map[k] for k in slow_ar_keys)
for shard in shards:
    print(f"  Loading {shard}...")
    with safe_open(str(MODEL_DIR / shard), framework="pt") as f:
        for key in f.keys():
            if key in slow_ar_keys:
                # Remap key: text_model.model.layers.N.X → layers.N.X
                clean = key.replace("text_model.model.", "")
                state[clean] = f.get_tensor(key).float()

# Load
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"  Loaded: {len(state)} tensors")
print(f"  Missing: {len(missing)}")
if missing:
    for m in missing[:5]:
        print(f"    {m}")
print()

model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {total_params/1e9:.2f}B params ({total_params*2/1024**3:.1f} GB BF16)")
print()

# --- Trace ---
print("Tracing (seq_len=1)...")
inp = torch.randn(1, 1, DIM)
with torch.no_grad():
    traced = torch.jit.trace(model, inp)
    out = traced(inp)
    print(f"  Output: {out.shape}")
print()

# --- Convert to CoreML ---
print("Converting to CoreML (this may take a while for 36 layers)...")
t0 = time.perf_counter()
try:
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_state", shape=(1, 1, DIM))],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
        minimum_deployment_target=ct.target.macOS15,
    )
    convert_time = time.perf_counter() - t0
    print(f"  Conversion OK ({convert_time:.0f}s)")

    out_path = "/tmp/fish_slow_ar_real.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")
    print()

    # --- Benchmark ---
    print("Benchmarking with REAL Fish slow AR weights...")
    data = {"hidden_state": np.random.randn(1, 1, DIM).astype(np.float32)}

    for cu, label in [
        (ct.ComputeUnit.CPU_AND_GPU, "GPU"),
        (ct.ComputeUnit.ALL, "ANE+GPU"),
    ]:
        mod = ct.models.MLModel(out_path, compute_units=cu)
        for _ in range(3):
            mod.predict(data)
        iters = 30
        t0 = time.perf_counter()
        for _ in range(iters):
            mod.predict(data)
        ms = (time.perf_counter() - t0) / iters * 1000
        print(f"  {label:10s}: {ms:.1f} ms/eval")

    print()
    print("Compare with MLX profiled: 34.7 ms")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print()
print("=== Done ===")

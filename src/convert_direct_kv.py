"""
Direct CoreML conversion of Fish S2 Pro slow AR WITH KV cache.

The model takes:
  - hidden_states: [1, 1, 2560] (single token embedding)
  - cos, sin: [1, 1, 1, 64] (RoPE for current position)
  - position: [1] (current position index)
  - kv_cache: [72, 8, MAX_SEQ, 128] (unified KV cache for all layers)

Returns:
  - output: [1, 1, 2560] (post-norm hidden state)
  - kv_cache_out: [72, 8, MAX_SEQ, 128] (updated cache)

The cache layout: layers 0..35 are K, layers 36..71 are V.
Each layer i reads K from kv_cache[i] and V from kv_cache[36+i].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
import time
import math
import json
import os
from pathlib import Path
from safetensors.torch import safe_open

MODEL_DIR = Path(os.environ.get("FISH_MODEL_DIR", "~/Models/fish-audio-s2-pro-mlx-bf16")).expanduser()

DIM = 2560
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728
N_LAYERS = 36
ROPE_BASE = 1000000.0
NORM_EPS = 1e-6
MAX_SEQ = 256  # Maximum sequence length for KV cache


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


def apply_rope(x, cos, sin):
    x1 = x[..., :HEAD_DIM // 2]
    x2 = x[..., HEAD_DIM // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class AttentionKV(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.wqkv = nn.Linear(DIM, (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM, bias=False)
        self.wo = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, NORM_EPS)
        self.k_norm = RMSNorm(HEAD_DIM, NORM_EPS)
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, x, cos, sin, kv_cache, position, causal_mask):
        """
        x: [1, 1, DIM]
        cos, sin: [1, 1, 1, HEAD_DIM//2]
        kv_cache: [72, 8, MAX_SEQ, 128]
        position: [1] int
        causal_mask: [1, 1, 1, MAX_SEQ]
        """
        B, S, D = x.shape
        qkv = self.wqkv(x)

        q_dim = N_HEADS * HEAD_DIM
        kv_dim = N_KV_HEADS * HEAD_DIM

        q = qkv[..., :q_dim].view(B, S, N_HEADS, HEAD_DIM)
        k = qkv[..., q_dim:q_dim + kv_dim].view(B, S, N_KV_HEADS, HEAD_DIM)
        v = qkv[..., q_dim + kv_dim:].view(B, S, N_KV_HEADS, HEAD_DIM)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)  # [1, 32, 1, 128]
        k = k.transpose(1, 2)  # [1, 8, 1, 128]
        v = v.transpose(1, 2)  # [1, 8, 1, 128]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Write new K, V to cache at current position
        # Use scatter/index_put for traceability
        li = self.layer_idx
        # kv_cache[li, :, position, :] = k and kv_cache[36+li, :, position, :] = v
        # For tracing: use narrow + copy pattern
        kv_cache = kv_cache.clone()
        kv_cache[li:li+1, :, position:position+1, :] = k
        kv_cache[N_LAYERS+li:N_LAYERS+li+1, :, position:position+1, :] = v

        # Read full K, V from cache
        k_full = kv_cache[li]       # [8, MAX_SEQ, 128]
        v_full = kv_cache[N_LAYERS + li]  # [8, MAX_SEQ, 128]

        # GQA expand
        k_full = k_full.unsqueeze(0).repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)  # [1, 32, MAX_SEQ, 128]
        v_full = v_full.unsqueeze(0).repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)

        # Attention with causal mask
        attn_weights = torch.matmul(q, k_full.transpose(-1, -2)) * self.scale  # [1, 32, 1, MAX_SEQ]
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v_full)  # [1, 32, 1, 128]

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, N_HEADS * HEAD_DIM)
        return self.wo(attn_out), kv_cache


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, INTERMEDIATE, bias=False)
        self.w2 = nn.Linear(INTERMEDIATE, DIM, bias=False)
        self.w3 = nn.Linear(DIM, INTERMEDIATE, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlockKV(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.attention = AttentionKV(layer_idx)
        self.feed_forward = FeedForward()
        self.attention_norm = RMSNorm(DIM, NORM_EPS)
        self.ffn_norm = RMSNorm(DIM, NORM_EPS)

    def forward(self, x, cos, sin, kv_cache, position, causal_mask):
        attn_out, kv_cache = self.attention(self.attention_norm(x), cos, sin, kv_cache, position, causal_mask)
        x = x + attn_out
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, kv_cache


class FishSlowARWithKV(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlockKV(i) for i in range(N_LAYERS)])
        self.norm = RMSNorm(DIM, NORM_EPS)

    def forward(self, x, cos, sin, kv_cache, position, causal_mask):
        for layer in self.layers:
            x, kv_cache = layer(x, cos, sin, kv_cache, position, causal_mask)
        return self.norm(x), kv_cache


def load_weights(model):
    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    keys = [k for k in wm if "text_model.model.layers" in k or k == "text_model.model.norm.weight"]
    state = {}
    for shard in sorted(set(wm[k] for k in keys)):
        print(f"  Loading {shard}...")
        with safe_open(str(MODEL_DIR / shard), framework="pt") as f:
            for key in f.keys():
                if key in keys:
                    state[key.replace("text_model.model.", "")] = f.get_tensor(key).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded {len(state)} tensors, missing={len(missing)}")
    return len(missing) == 0


def get_rope(pos):
    freqs = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    angles = torch.outer(torch.tensor([float(pos)]), freqs)
    return torch.cos(angles).view(1, 1, 1, 64), torch.sin(angles).view(1, 1, 1, 64)


def make_causal_mask(position):
    """Create causal mask: 0 for positions <= current, -inf for future."""
    mask = torch.zeros(1, 1, 1, MAX_SEQ)
    mask[0, 0, 0, position + 1:] = -10000.0
    return mask


if __name__ == "__main__":
    print("=== Fish Slow AR with KV Cache → CoreML ===")
    print(f"Max sequence: {MAX_SEQ}")

    # Build and load
    model = FishSlowARWithKV()
    load_weights(model)
    model.eval()

    # Test PyTorch KV cache decode
    print("\nTesting PyTorch KV cache decode (8 steps)...")
    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    shard = wm["text_model.model.embeddings.weight"]
    with safe_open(str(MODEL_DIR / shard), framework="pt") as f:
        emb_w = f.get_tensor("text_model.model.embeddings.weight").float()

    kv = torch.zeros(72, 8, MAX_SEQ, 128)
    tokens = [100]
    with torch.no_grad():
        for step in range(8):
            tok = tokens[-1]
            h = emb_w[tok].unsqueeze(0).unsqueeze(0)
            cos, sin = get_rope(step)
            mask = make_causal_mask(step)
            out, kv = model(h, cos, sin, kv, torch.tensor([step], dtype=torch.int64), mask)
            logits = F.linear(out, emb_w)
            next_tok = logits[0, 0].argmax().item()
            print(f"  Step {step}: {tok} -> {next_tok}")
            tokens.append(next_tok)
    print(f"  Sequence: {tokens}")

    # Trace for CoreML
    print("\nTracing for CoreML...")
    h_sample = torch.randn(1, 1, DIM)
    cos_s, sin_s = get_rope(0)
    kv_s = torch.zeros(72, 8, MAX_SEQ, 128)
    pos_s = torch.tensor([0], dtype=torch.int64)
    mask_s = make_causal_mask(0)

    with torch.no_grad():
        traced = torch.jit.trace(model, (h_sample, cos_s, sin_s, kv_s, pos_s, mask_s))

    print("Converting to CoreML...")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, 1, DIM)),
            ct.TensorType(name="cos", shape=(1, 1, 1, 64)),
            ct.TensorType(name="sin", shape=(1, 1, 1, 64)),
            ct.TensorType(name="kv_cache", shape=(72, 8, MAX_SEQ, 128)),
            ct.TensorType(name="position", shape=(1,)),
            ct.TensorType(name="causal_mask", shape=(1, 1, 1, MAX_SEQ)),
        ],
        outputs=[
            ct.TensorType(name="output"),
            ct.TensorType(name="kv_cache_out"),
        ],
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  Conversion took {time.perf_counter()-t0:.0f}s")

    out_path = "/tmp/fish_slow_ar_direct_kv.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Benchmark
    print("\nBenchmarking...")
    for cu, label in [
        (ct.ComputeUnit.CPU_AND_GPU, "GPU"),
        (ct.ComputeUnit.ALL, "ANE+GPU"),
    ]:
        mod = ct.models.MLModel(out_path, compute_units=cu)
        data = {
            "hidden_states": np.random.randn(1, 1, DIM).astype(np.float32),
            "cos": cos_s.numpy().astype(np.float32),
            "sin": sin_s.numpy().astype(np.float32),
            "kv_cache": np.zeros((72, 8, MAX_SEQ, 128), dtype=np.float16),
            "position": np.array([0], dtype=np.int32),
            "causal_mask": make_causal_mask(0).numpy().astype(np.float16),
        }
        for _ in range(5):
            mod.predict(data)
        iters = 30
        t0 = time.perf_counter()
        for _ in range(iters):
            mod.predict(data)
        ms = (time.perf_counter() - t0) / iters * 1000
        print(f"  {label:10s}: {ms:.1f} ms/token")

    print(f"\n  No-KV baseline: 24.3 ms/token")
    print(f"  MLX baseline: 34.7 ms/token")

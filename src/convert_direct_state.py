"""
Direct CoreML conversion of Fish S2 Pro slow AR with GPU-resident KV cache.

Uses ct.StateType so the KV cache stays in GPU/ANE memory between predict()
calls — no 75MB copy per token. Expected ~25ms/token (vs 110ms with naive I/O).

Architecture: Same as convert_direct.py but with:
- KV cache as register_buffer (becomes CoreML state)
- In-place cache updates via scatter
- Causal mask passed as input (position-dependent)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import numpy as np
import time
import math
import json
from pathlib import Path
from safetensors.torch import safe_open

MODEL_DIR = Path.home() / "Models/fish-audio-s2-pro-mlx-bf16"

DIM = 2560
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728
N_LAYERS = 36
ROPE_BASE = 1000000.0
NORM_EPS = 1e-6
MAX_SEQ = 256


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


class AttentionStateful(nn.Module):
    """Attention with stateful KV cache via parent model's registered buffer."""

    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.wqkv = nn.Linear(DIM, (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM, bias=False)
        self.wo = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, NORM_EPS)
        self.k_norm = RMSNorm(HEAD_DIM, NORM_EPS)
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, x, cos, sin, kv_cache, causal_mask):
        """
        x: [1, 1, DIM]
        cos, sin: [1, 1, 1, 64]
        kv_cache: [72, N_KV_HEADS, MAX_SEQ, HEAD_DIM] — the model's buffer
        causal_mask: [1, 1, 1, MAX_SEQ]
        """
        qkv = self.wqkv(x)
        q_dim = N_HEADS * HEAD_DIM
        kv_dim = N_KV_HEADS * HEAD_DIM

        q = qkv[..., :q_dim].view(1, 1, N_HEADS, HEAD_DIM)
        k = qkv[..., q_dim:q_dim + kv_dim].view(1, 1, N_KV_HEADS, HEAD_DIM)
        v = qkv[..., q_dim + kv_dim:].view(1, 1, N_KV_HEADS, HEAD_DIM)

        q = self.q_norm(q).transpose(1, 2)  # [1, 32, 1, 128]
        k = self.k_norm(k).transpose(1, 2)  # [1, 8, 1, 128]
        v = v.transpose(1, 2)                # [1, 8, 1, 128]

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Write new K/V to cache using shift-left + append (static slice, ANE-safe)
        li = self.layer_idx
        k_store = k.to(kv_cache.dtype)
        v_store = v.to(kv_cache.dtype)
        old_k = kv_cache[li:li+1, :, 1:, :]
        new_k_cache = torch.cat([old_k, k_store], dim=2)
        old_v = kv_cache[N_LAYERS+li:N_LAYERS+li+1, :, 1:, :]
        new_v_cache = torch.cat([old_v, v_store], dim=2)

        # Update cache in-place
        kv_cache[li] = new_k_cache.squeeze(0)
        kv_cache[N_LAYERS + li] = new_v_cache.squeeze(0)

        # Read full cache for attention (cast to computation dtype)
        k_full = kv_cache[li].unsqueeze(0).to(q.dtype)              # [1, 8, MAX_SEQ, 128]
        v_full = kv_cache[N_LAYERS + li].unsqueeze(0).to(q.dtype)   # [1, 8, MAX_SEQ, 128]

        # GQA expand
        k_full = k_full.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
        v_full = v_full.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)

        # Attention
        attn_w = torch.matmul(q, k_full.transpose(-1, -2)) * self.scale
        attn_w = attn_w + causal_mask
        attn_w = torch.softmax(attn_w, dim=-1)
        attn_out = torch.matmul(attn_w, v_full)

        attn_out = attn_out.transpose(1, 2).contiguous().view(1, 1, N_HEADS * HEAD_DIM)
        return self.wo(attn_out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, INTERMEDIATE, bias=False)
        self.w2 = nn.Linear(INTERMEDIATE, DIM, bias=False)
        self.w3 = nn.Linear(DIM, INTERMEDIATE, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlockStateful(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.attention = AttentionStateful(layer_idx)
        self.feed_forward = FeedForward()
        self.attention_norm = RMSNorm(DIM, NORM_EPS)
        self.ffn_norm = RMSNorm(DIM, NORM_EPS)

    def forward(self, x, cos, sin, kv_cache, causal_mask):
        x = x + self.attention(self.attention_norm(x), cos, sin, kv_cache, causal_mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class FishSlowARStateful(nn.Module):
    """Fish S2 Pro slow AR with GPU-resident KV cache state."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlockStateful(i) for i in range(N_LAYERS)])
        self.norm = RMSNorm(DIM, NORM_EPS)

        # KV cache as registered buffer — becomes ct.StateType in CoreML
        # Layout: [0..35] = K caches, [36..71] = V caches
        self.register_buffer(
            "kv_cache",
            torch.zeros(N_LAYERS * 2, N_KV_HEADS, MAX_SEQ, HEAD_DIM, dtype=torch.float16)
        )

    def forward(self, x, cos, sin, causal_mask):
        """
        x: [1, 1, DIM]
        cos, sin: [1, 1, 1, 64]
        causal_mask: [1, 1, 1, MAX_SEQ]
        Returns: [1, 1, DIM]
        """
        for layer in self.layers:
            x = layer(x, cos, sin, self.kv_cache, causal_mask)
        return self.norm(x)


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
    # Filter expected missing (kv_cache buffer)
    missing = [m for m in missing if "kv_cache" not in m]
    print(f"  Loaded {len(state)} tensors, missing={len(missing)}")
    if missing:
        print(f"  Missing: {missing[:5]}")
    return len(missing) == 0


def get_rope(pos):
    freqs = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    angles = torch.outer(torch.tensor([float(pos)]), freqs)
    return torch.cos(angles).view(1, 1, 1, 64), torch.sin(angles).view(1, 1, 1, 64)


def make_causal_mask(num_valid_positions):
    """Mask: 0 for the last num_valid_positions, -inf for the rest."""
    mask = torch.full((1, 1, 1, MAX_SEQ), -10000.0)
    if num_valid_positions > 0:
        mask[0, 0, 0, -num_valid_positions:] = 0.0
    return mask


if __name__ == "__main__":
    print("=== Fish Slow AR with Stateful KV Cache → CoreML ===")

    # Build and load
    model = FishSlowARStateful()
    load_weights(model)
    model.eval()

    # Test PyTorch
    print("\nPyTorch test (3 steps)...")
    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    shard = wm["text_model.model.embeddings.weight"]
    with safe_open(str(MODEL_DIR / shard), framework="pt") as f:
        emb_w = f.get_tensor("text_model.model.embeddings.weight").float()

    tokens = [100]
    with torch.no_grad():
        for step in range(3):
            tok = tokens[-1]
            h = emb_w[tok].unsqueeze(0).unsqueeze(0)
            cos, sin = get_rope(step)
            mask = make_causal_mask(step + 1)
            out = model(h, cos, sin, mask)
            logits = F.linear(out, emb_w)
            next_tok = logits[0, 0].argmax().item()
            print(f"  Step {step}: {tok} -> {next_tok}")
            tokens.append(next_tok)

    # Trace
    print("\nTracing...")
    h_s = torch.randn(1, 1, DIM)
    cos_s, sin_s = get_rope(0)
    mask_s = make_causal_mask(1)

    with torch.no_grad():
        traced = torch.jit.trace(model, (h_s, cos_s, sin_s, mask_s))

    # Convert with StateType
    print("Converting to CoreML with StateType...")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, 1, DIM)),
            ct.TensorType(name="cos", shape=(1, 1, 1, 64)),
            ct.TensorType(name="sin", shape=(1, 1, 1, 64)),
            ct.TensorType(name="causal_mask", shape=(1, 1, 1, MAX_SEQ)),
        ],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(N_LAYERS * 2, N_KV_HEADS, MAX_SEQ, HEAD_DIM),
                    dtype=np.float16,
                ),
                name="kv_cache",
            ),
        ],
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  Conversion took {time.perf_counter()-t0:.0f}s")

    out_path = "/tmp/fish_slow_ar_state.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Benchmark
    print("\nBenchmarking...")
    for cu, label in [
        (ct.ComputeUnit.CPU_AND_GPU, "GPU"),
        (ct.ComputeUnit.ALL, "ANE+GPU"),
    ]:
        mod = ct.models.MLModel(out_path, compute_units=cu)
        state = mod.make_state()

        data = {
            "hidden_states": np.random.randn(1, 1, DIM).astype(np.float16),
            "cos": cos_s.numpy().astype(np.float16),
            "sin": sin_s.numpy().astype(np.float16),
            "causal_mask": mask_s.numpy().astype(np.float16),
        }

        # Warmup
        for _ in range(10):
            mod.predict(data, state)

        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            mod.predict(data, state)
        ms = (time.perf_counter() - t0) / iters * 1000
        print(f"  {label:10s}: {ms:.1f} ms/token")

    print(f"\n  No-KV direct:  24.3 ms/token")
    print(f"  Naive KV I/O: 110.7 ms/token")
    print(f"  MLX baseline:  34.7 ms/token")

    print("\n=== Done ===")

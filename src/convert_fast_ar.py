"""
Direct CoreML conversion of Fish S2 Pro's fast AR.

The fast AR takes slow AR hidden states and generates codebook tokens.
4 transformer layers, same dim as slow AR, no QK norm, max_seq=10.

Called 10 times per semantic token (1 prefill + 9 codebook iterations).
At 3.2ms per call on MLX, this is 32ms total — the remaining bottleneck.
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

# Fast AR config (from audio_decoder_config)
DIM = 2560
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728
N_LAYERS = 4
CODEBOOK_SIZE = 4096
NUM_CODEBOOKS = 10
ROPE_BASE = 1000000.0
NORM_EPS = 1e-6


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


class FastAttention(nn.Module):
    """Fast AR attention — NO QK norm, same GQA structure."""

    def __init__(self):
        super().__init__()
        self.wqkv = nn.Linear(DIM, (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM, bias=False)
        self.wo = nn.Linear(N_HEADS * HEAD_DIM, DIM, bias=False)
        self.scale = 1.0 / math.sqrt(HEAD_DIM)

    def forward(self, x, cos, sin):
        B, S, D = x.shape
        qkv = self.wqkv(x)
        q_dim = N_HEADS * HEAD_DIM
        kv_dim = N_KV_HEADS * HEAD_DIM

        q = qkv[..., :q_dim].view(B, S, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = qkv[..., q_dim:q_dim + kv_dim].view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim:].view(B, S, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        # NO QK norm for fast AR
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
        v = v.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)

        attn_w = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn_w = torch.softmax(attn_w, dim=-1)
        attn_out = torch.matmul(attn_w, v)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, N_HEADS * HEAD_DIM)
        return self.wo(attn_out)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, INTERMEDIATE, bias=False)
        self.w2 = nn.Linear(INTERMEDIATE, DIM, bias=False)
        self.w3 = nn.Linear(DIM, INTERMEDIATE, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class FastTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FastAttention()
        self.feed_forward = FeedForward()
        self.attention_norm = RMSNorm(DIM, NORM_EPS)
        self.ffn_norm = RMSNorm(DIM, NORM_EPS)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.attention_norm(x), cos, sin)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class FishFastAR(nn.Module):
    """Fish S2 Pro fast AR: 4 layers, no QK norm."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FastTransformerBlock() for _ in range(N_LAYERS)])
        self.norm = RMSNorm(DIM, NORM_EPS)
        self.output = nn.Linear(DIM, CODEBOOK_SIZE, bias=False)

    def forward(self, x, cos, sin):
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.output(x)


def load_fast_ar_weights(model):
    """Load fast AR weights from Fish safetensors."""
    with open(MODEL_DIR / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]

    # Fast AR weights are under audio_decoder namespace
    # But looking at the model, fast_layers are part of text_model
    # Let me check what keys exist
    fast_keys = [k for k in wm if "fast_layers" in k or "fast_norm" in k or "fast_output" in k]
    print(f"  Found {len(fast_keys)} fast AR keys")
    if fast_keys:
        print(f"  Sample: {fast_keys[:3]}")

    state = {}
    shards = set(wm[k] for k in fast_keys)
    for shard in sorted(shards):
        print(f"  Loading {shard}...")
        with safe_open(str(MODEL_DIR / shard), framework="pt") as f:
            for key in f.keys():
                if key not in fast_keys:
                    continue
                # Remap: text_model.fast_layers.N.X -> layers.N.X
                clean = key
                for prefix in ["text_model.", "audio_decoder."]:
                    clean = clean.replace(prefix, "")
                clean = clean.replace("fast_layers.", "layers.")
                clean = clean.replace("fast_norm.", "norm.")
                clean = clean.replace("fast_output.", "output.")
                clean = clean.replace(".attention.wqkv.", ".attention.wqkv.")
                clean = clean.replace(".attention.wo.", ".attention.wo.")
                clean = clean.replace(".feed_forward.w1.", ".feed_forward.w1.")
                clean = clean.replace(".feed_forward.w2.", ".feed_forward.w2.")
                clean = clean.replace(".feed_forward.w3.", ".feed_forward.w3.")
                clean = clean.replace(".attention_norm.", ".attention_norm.")
                clean = clean.replace(".ffn_norm.", ".ffn_norm.")
                state[clean] = f.get_tensor(key).float()

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded {len(state)} tensors")
    if missing:
        print(f"  Missing: {missing}")
    if unexpected:
        print(f"  Unexpected: {unexpected}")
    return len(missing) == 0


def get_rope(pos):
    freqs = 1.0 / (ROPE_BASE ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    angles = torch.outer(torch.tensor([float(pos)]), freqs)
    return torch.cos(angles).view(1, 1, 1, 64), torch.sin(angles).view(1, 1, 1, 64)


if __name__ == "__main__":
    print("=== Fish S2 Pro Fast AR → Direct CoreML ===")
    print(f"Architecture: {N_LAYERS} layers, dim={DIM}, no QK norm")
    print(f"Codebook: {CODEBOOK_SIZE} codes, {NUM_CODEBOOKS} codebooks")
    print()

    model = FishFastAR()
    load_fast_ar_weights(model)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {params / 1e6:.0f}M params")

    # Test
    print("\nPyTorch test...")
    x = torch.randn(1, 1, DIM)
    cos, sin = get_rope(0)
    with torch.no_grad():
        out = model(x, cos, sin)
    print(f"  Output: shape={out.shape}")
    print(f"  Top-5 codebook codes: {torch.topk(out[0, 0], 5).indices.tolist()}")

    # Trace & convert
    print("\nTracing and converting...")
    with torch.no_grad():
        traced = torch.jit.trace(model, (x, cos, sin))

    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, 1, DIM)),
            ct.TensorType(name="cos", shape=(1, 1, 1, 64)),
            ct.TensorType(name="sin", shape=(1, 1, 1, 64)),
        ],
        outputs=[ct.TensorType(name="codebook_logits")],
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"  Conversion took {time.perf_counter()-t0:.0f}s")

    out_path = "/tmp/fish_fast_ar_direct.mlpackage"
    mlmodel.save(out_path)
    print(f"  Saved to {out_path}")

    # Parity check
    print("\nCoreML parity...")
    cm_out = mlmodel.predict({
        "hidden_states": x.numpy().astype(np.float32),
        "cos": cos.numpy().astype(np.float32),
        "sin": sin.numpy().astype(np.float32),
    })
    cm_logits = torch.from_numpy(cm_out["codebook_logits"]).float()
    pt_logits = out.float()
    cos_sim = F.cosine_similarity(pt_logits.flatten().unsqueeze(0), cm_logits.flatten().unsqueeze(0)).item()
    max_err = (pt_logits.flatten() - cm_logits.flatten()).abs().max().item()
    top1_match = pt_logits[0,0].argmax().item() == cm_logits[0,0].argmax().item()
    print(f"  cos={cos_sim:.10f}, max_err={max_err:.6f}, top1_match={top1_match}")

    # Benchmark
    print("\nBenchmarking (10 calls = 1 semantic token)...")
    for cu, label in [
        (ct.ComputeUnit.CPU_AND_GPU, "GPU"),
        (ct.ComputeUnit.ALL, "ANE+GPU"),
        (ct.ComputeUnit.CPU_AND_NE, "ANE only"),
    ]:
        mod = ct.models.MLModel(out_path, compute_units=cu)
        data = {
            "hidden_states": np.random.randn(1, 1, DIM).astype(np.float32),
            "cos": cos.numpy().astype(np.float32),
            "sin": sin.numpy().astype(np.float32),
        }
        for _ in range(10):
            mod.predict(data)
        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            mod.predict(data)
        ms = (time.perf_counter() - t0) / iters * 1000
        print(f"  {label:10s}: {ms:.2f} ms/call × 10 = {ms*10:.1f} ms/semantic_tok")

    print(f"\n  MLX fast AR: 3.2 ms/call × 10 = 32 ms/semantic_tok")

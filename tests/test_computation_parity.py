"""
Computation parity test: Fish PyTorch vs ANEMLL Qwen2.5

Tests actual forward pass computation, not just weight loading.
Follows the other agent's minimum acceptable order:
1. Single embedding parity
2. Single-layer parity (q/k/v proj, q/k after norm, attention output, layer output)
3. One-step full-model parity (hidden states + logits)
4. Multi-step KV-cache parity (2-step, 8-step)
"""

import sys
import math
import json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, "/Users/kyin/Projects/anemll")

FISH_MODEL_DIR = "/Users/kyin/Models/fish-audio-s2-pro-mlx-bf16"
ANEMLL_CKPT_DIR = "/tmp/fish_slow_ar_qwen_format"

torch.set_grad_enabled(False)


def load_fish_weights():
    """Load raw Fish weights for manual computation."""
    from safetensors.torch import safe_open

    with open(f"{FISH_MODEL_DIR}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    weights = {}
    shards_loaded = set()

    def get(key):
        if key in weights:
            return weights[key]
        shard = weight_map[key]
        if shard not in shards_loaded:
            with safe_open(f"{FISH_MODEL_DIR}/{shard}", framework="pt") as f:
                for k in f.keys():
                    weights[k] = f.get_tensor(k)
            shards_loaded.add(shard)
        return weights[key]

    return get


def load_anemll_model():
    """Load ANEMLL model with Fish weights."""
    from anemll.models.qwen2_5_model import Qwen25Config, Qwen25ForCausalLM
    config = Qwen25Config.from_json(f"{ANEMLL_CKPT_DIR}/config.json")
    model = Qwen25ForCausalLM(config, enable_coreml=False)
    model.load_pretrained_weights(ANEMLL_CKPT_DIR)
    model.eval()
    return model, config


def fish_rms_norm(x, weight, eps=1e-6):
    """Fish's RMSNorm: standard implementation."""
    x_f32 = x.float()
    rms = torch.sqrt(x_f32.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f32 / rms * weight.float()).to(x.dtype)


def report(name, a, b, threshold=0.01):
    """Compare two tensors and report."""
    a_f32, b_f32 = a.float(), b.float()
    max_err = (a_f32 - b_f32).abs().max().item()
    mean_err = (a_f32 - b_f32).abs().mean().item()
    cos = F.cosine_similarity(a_f32.flatten().unsqueeze(0), b_f32.flatten().unsqueeze(0)).item()
    passed = max_err < threshold
    status = "PASS" if passed else "FAIL"
    print(f"  {name:>30}: max_err={max_err:.6f}, mean_err={mean_err:.6f}, cos={cos:.8f} [{status}]")
    return passed


# ============================================================
# Test 1: Single Embedding Parity
# ============================================================
def test_embedding():
    print("\n" + "=" * 70)
    print("Test 1: Single Embedding Parity")
    print("=" * 70)

    get_fish = load_fish_weights()
    anemll_model, _ = load_anemll_model()

    fish_emb_w = get_fish("text_model.model.embeddings.weight").float()
    anemll_emb_w = anemll_model.model.embed_tokens.weight.float()

    token_ids = [0, 42, 100, 1000, 50000, 100000, 155775]
    all_pass = True
    for tid in token_ids:
        fish_out = fish_emb_w[tid]
        anemll_out = anemll_emb_w[tid]
        p = report(f"Token {tid}", fish_out, anemll_out, threshold=1e-6)
        all_pass = all_pass and p

    print(f"\nTest 1: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ============================================================
# Test 2: Single-Layer Computation Parity
# ============================================================
def test_single_layer():
    print("\n" + "=" * 70)
    print("Test 2: Single-Layer Computation Parity (Layer 0)")
    print("=" * 70)

    get_fish = load_fish_weights()
    anemll_model, config = load_anemll_model()

    n_head = 32
    n_kv_head = 8
    head_dim = 128
    hidden_size = 2560
    eps = 1e-6

    # Create a deterministic input hidden state
    torch.manual_seed(42)
    x = torch.randn(1, 1, hidden_size, dtype=torch.float32)

    # ---- Fish manual computation ----

    # Step A: Input LayerNorm
    fish_ln_w = get_fish("text_model.model.layers.0.attention_norm.weight").float()
    fish_normed = fish_rms_norm(x, fish_ln_w, eps)

    # Step B: QKV projection (fused)
    fish_wqkv = get_fish("text_model.model.layers.0.attention.wqkv.weight").float()
    fish_qkv = F.linear(fish_normed, fish_wqkv)  # [1, 1, 6144]

    q_dim = n_head * head_dim       # 4096
    kv_dim = n_kv_head * head_dim   # 1024
    fish_q, fish_k, fish_v = fish_qkv.split([q_dim, kv_dim, kv_dim], dim=-1)

    # Reshape to per-head
    fish_q = fish_q.view(1, 1, n_head, head_dim)       # [1, 1, 32, 128]
    fish_k = fish_k.view(1, 1, n_kv_head, head_dim)    # [1, 1, 8, 128]
    fish_v = fish_v.view(1, 1, n_kv_head, head_dim)    # [1, 1, 8, 128]

    # Step C: QK Norm (after reshape, before RoPE)
    fish_qn_w = get_fish("text_model.model.layers.0.attention.q_norm.weight").float()
    fish_kn_w = get_fish("text_model.model.layers.0.attention.k_norm.weight").float()
    fish_q_normed = fish_rms_norm(fish_q, fish_qn_w, eps)
    fish_k_normed = fish_rms_norm(fish_k, fish_kn_w, eps)

    # ---- ANEMLL computation ----

    # ANEMLL uses Conv2d, so input needs to be [1, hidden, 1, seq] format
    # But let's use the model's actual forward path components

    layer = anemll_model.model.layers[0]
    attn = layer.self_attn

    # Step A: Input LayerNorm (ANEMLL's Qwen25RMSNorm)
    anemll_normed = layer.input_layernorm(x)

    # Step B: QKV projection (separate q/k/v_proj Conv2d)
    # ANEMLL expects [1, hidden, 1, seq] for Conv2d in float16
    h_conv = anemll_normed.half().permute(0, 2, 1).unsqueeze(2)  # [1, 2560, 1, 1] fp16
    anemll_q_raw = attn.q_proj(h_conv)  # [1, 4096, 1, 1]
    anemll_k_raw = attn.k_proj(h_conv)  # [1, 1024, 1, 1]
    anemll_v_raw = attn.v_proj(h_conv)  # [1, 1024, 1, 1]

    # Reshape to per-head (matching ANEMLL's view: [1, n_heads, 1, head_dim])
    anemll_q = anemll_q_raw.view(1, n_head, head_dim, 1).permute(0, 1, 3, 2)     # [1, 32, 1, 128]
    anemll_k = anemll_k_raw.view(1, n_kv_head, head_dim, 1).permute(0, 1, 3, 2)  # [1, 8, 1, 128]
    anemll_v = anemll_v_raw.view(1, n_kv_head, head_dim, 1).permute(0, 1, 3, 2)  # [1, 8, 1, 128]

    # Step C: QK Norm
    anemll_q_normed = attn.q_norm(anemll_q) if attn.use_qk_norm else anemll_q
    anemll_k_normed = attn.k_norm(anemll_k) if attn.use_qk_norm else anemll_k

    # ---- Compare at each stage ----

    all_pass = True

    # Input norm
    p = report("Input LayerNorm", fish_normed, anemll_normed, threshold=0.01)
    all_pass = all_pass and p

    # Q projection (reshape Fish to match ANEMLL: [1, n_heads, seq, head_dim] vs [1, seq, n_heads, head_dim])
    fish_q_cmp = fish_q.permute(0, 2, 1, 3)    # [1, 32, 1, 128]
    fish_k_cmp = fish_k.permute(0, 2, 1, 3)    # [1, 8, 1, 128]
    fish_v_cmp = fish_v.permute(0, 2, 1, 3)    # [1, 8, 1, 128]

    p = report("Q projection", fish_q_cmp, anemll_q, threshold=0.05)
    all_pass = all_pass and p
    p = report("K projection", fish_k_cmp, anemll_k, threshold=0.05)
    all_pass = all_pass and p
    p = report("V projection", fish_v_cmp, anemll_v, threshold=0.05)
    all_pass = all_pass and p

    # QK norm output
    fish_q_normed_cmp = fish_q_normed.permute(0, 2, 1, 3)  # [1, 32, 1, 128]
    fish_k_normed_cmp = fish_k_normed.permute(0, 2, 1, 3)  # [1, 8, 1, 128]

    p = report("Q after q_norm", fish_q_normed_cmp, anemll_q_normed, threshold=0.05)
    all_pass = all_pass and p
    p = report("K after k_norm", fish_k_normed_cmp, anemll_k_normed, threshold=0.05)
    all_pass = all_pass and p

    print(f"\nTest 2: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Fish S2 Pro vs ANEMLL — Computation Parity Test")
    print("=" * 70)

    results = {}
    results["1_embedding"] = test_embedding()
    results["2_single_layer"] = test_single_layer()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name:>20}: {'PASS' if passed else 'FAIL'}")
    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    if not all_pass:
        sys.exit(1)

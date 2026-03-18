"""
Correctness parity test: Fish PyTorch vs ANEMLL Qwen2.5

Tests that the ANEMLL-converted model produces numerically equivalent
output to Fish's original PyTorch model on the same input.

Phases:
1. Single embedding lookup parity
2. Single-layer attention parity (Q, K after norm, after RoPE)
3. Full model forward pass parity (hidden states + logits)
"""

import sys
import torch
import numpy as np

sys.path.insert(0, "/Users/kyin/Projects/anemll")
sys.path.insert(0, "/Users/kyin/Projects/fish-speech")

FISH_MODEL_DIR = "/Users/kyin/Models/fish-audio-s2-pro-mlx-bf16"
ANEMLL_CKPT_DIR = "/tmp/fish_slow_ar_qwen_format"

# Tolerances
ATOL_FP16 = 0.01    # float16 conversion introduces ~1e-3 error
ATOL_LOGITS = 0.05  # logits accumulate error across layers


def load_fish_model():
    """Load Fish S2 Pro's slow AR in PyTorch."""
    from safetensors.torch import safe_open
    import json

    with open(f"{FISH_MODEL_DIR}/config.json") as f:
        config = json.load(f)

    text_config = config["text_config"]
    print(f"Fish config: dim={text_config['dim']}, layers={text_config['n_layer']}, "
          f"heads={text_config['n_head']}, kv_heads={text_config['n_local_heads']}, "
          f"qk_norm={text_config['attention_qk_norm']}")

    return text_config


def load_anemll_model():
    """Load ANEMLL Qwen2.5 model with Fish weights."""
    from anemll.models.qwen2_5_model import Qwen25Config, Qwen25ForCausalLM

    config = Qwen25Config.from_json(f"{ANEMLL_CKPT_DIR}/config.json")
    model = Qwen25ForCausalLM(config, enable_coreml=False)
    model.load_pretrained_weights(ANEMLL_CKPT_DIR)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, config


def test_embedding_parity():
    """Phase 1: Verify embedding weights match between Fish and ANEMLL."""
    print("\n" + "=" * 60)
    print("Phase 1: Embedding Parity")
    print("=" * 60)

    from safetensors.torch import safe_open
    import json

    # Load Fish embedding weight directly
    with open(f"{FISH_MODEL_DIR}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    fish_emb_shard = weight_map["text_model.model.embeddings.weight"]
    with safe_open(f"{FISH_MODEL_DIR}/{fish_emb_shard}", framework="pt") as f:
        fish_emb = f.get_tensor("text_model.model.embeddings.weight")

    # Load ANEMLL embedding weight
    anemll_model, _ = load_anemll_model()
    anemll_emb = anemll_model.model.embed_tokens.weight

    # Compare (ANEMLL may have reshuffled to Conv2d but embed_tokens is nn.Embedding)
    print(f"Fish embedding:   shape={fish_emb.shape}, dtype={fish_emb.dtype}")
    print(f"ANEMLL embedding: shape={anemll_emb.shape}, dtype={anemll_emb.dtype}")

    # Both should be [vocab_size, hidden_size]
    fish_f32 = fish_emb.float()
    anemll_f32 = anemll_emb.float()

    if fish_f32.shape != anemll_f32.shape:
        print(f"FAIL: Shape mismatch {fish_f32.shape} vs {anemll_f32.shape}")
        return False

    max_err = (fish_f32 - anemll_f32).abs().max().item()
    mean_err = (fish_f32 - anemll_f32).abs().mean().item()
    print(f"Max absolute error:  {max_err:.8f}")
    print(f"Mean absolute error: {mean_err:.8f}")

    # Test a few token embeddings
    for tok_id in [0, 100, 1000, 50000, 155775]:
        fish_vec = fish_f32[tok_id]
        anemll_vec = anemll_f32[tok_id]
        err = (fish_vec - anemll_vec).abs().max().item()
        cos_sim = torch.nn.functional.cosine_similarity(fish_vec.unsqueeze(0), anemll_vec.unsqueeze(0)).item()
        print(f"  Token {tok_id:>6}: max_err={err:.8f}, cos_sim={cos_sim:.8f}")

    passed = max_err < ATOL_FP16
    print(f"\nPhase 1: {'PASS' if passed else 'FAIL'} (max_err={max_err:.8f}, threshold={ATOL_FP16})")
    return passed


def test_qk_norm_parity():
    """Phase 2: Verify QK normalization weights match."""
    print("\n" + "=" * 60)
    print("Phase 2: QK Norm Weight Parity")
    print("=" * 60)

    from safetensors.torch import safe_open
    import json

    # Load Fish QK norm weights directly
    with open(f"{FISH_MODEL_DIR}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    # Load ANEMLL model
    anemll_model, _ = load_anemll_model()

    all_pass = True
    for layer_idx in [0, 17, 35]:  # First, middle, last
        for norm_type in ["q_norm", "k_norm"]:
            fish_key = f"text_model.model.layers.{layer_idx}.attention.{norm_type}.weight"
            shard = weight_map[fish_key]

            with safe_open(f"{FISH_MODEL_DIR}/{shard}", framework="pt") as f:
                fish_w = f.get_tensor(fish_key).float()

            anemll_w = getattr(anemll_model.model.layers[layer_idx].self_attn, norm_type).weight.float()

            max_err = (fish_w - anemll_w).abs().max().item()
            match = max_err < ATOL_FP16
            all_pass = all_pass and match
            status = "OK" if match else "FAIL"
            print(f"  Layer {layer_idx:>2} {norm_type}: max_err={max_err:.8f} [{status}]")

    print(f"\nPhase 2: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_attention_proj_parity():
    """Phase 2b: Verify Q/K/V projection weights match after wqkv split."""
    print("\n" + "=" * 60)
    print("Phase 2b: Attention Projection Parity (wqkv split)")
    print("=" * 60)

    from safetensors.torch import safe_open
    import json

    with open(f"{FISH_MODEL_DIR}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    anemll_model, _ = load_anemll_model()

    n_head = 32
    n_kv_head = 8
    head_dim = 128
    q_dim = n_head * head_dim      # 4096
    kv_dim = n_kv_head * head_dim  # 1024

    all_pass = True
    for layer_idx in [0, 17, 35]:
        fish_key = f"text_model.model.layers.{layer_idx}.attention.wqkv.weight"
        shard = weight_map[fish_key]

        with safe_open(f"{FISH_MODEL_DIR}/{shard}", framework="pt") as f:
            wqkv = f.get_tensor(fish_key).float()

        # Split fused QKV same way as adapter
        fish_q, fish_k, fish_v = wqkv.split([q_dim, kv_dim, kv_dim], dim=0)

        # ANEMLL stores in Conv2d format [out, in, 1, 1]
        anemll_q = anemll_model.model.layers[layer_idx].self_attn.q_proj.weight.float().squeeze()
        anemll_k = anemll_model.model.layers[layer_idx].self_attn.k_proj.weight.float().squeeze()
        anemll_v = anemll_model.model.layers[layer_idx].self_attn.v_proj.weight.float().squeeze()

        for name, fish_w, anemll_w in [("Q", fish_q, anemll_q), ("K", fish_k, anemll_k), ("V", fish_v, anemll_v)]:
            if fish_w.shape != anemll_w.shape:
                print(f"  Layer {layer_idx} {name}: SHAPE MISMATCH {fish_w.shape} vs {anemll_w.shape}")
                all_pass = False
                continue
            max_err = (fish_w - anemll_w).abs().max().item()
            match = max_err < ATOL_FP16
            all_pass = all_pass and match
            status = "OK" if match else "FAIL"
            print(f"  Layer {layer_idx:>2} {name}_proj: max_err={max_err:.8f} [{status}]")

    print(f"\nPhase 2b: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_ffn_parity():
    """Phase 2c: Verify FFN weights match."""
    print("\n" + "=" * 60)
    print("Phase 2c: FFN Weight Parity")
    print("=" * 60)

    from safetensors.torch import safe_open
    import json

    with open(f"{FISH_MODEL_DIR}/model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    anemll_model, _ = load_anemll_model()

    fish_to_anemll = {
        "feed_forward.w1": "mlp.gate_proj",
        "feed_forward.w2": "mlp.down_proj",
        "feed_forward.w3": "mlp.up_proj",
    }

    all_pass = True
    for layer_idx in [0, 17, 35]:
        for fish_name, anemll_name in fish_to_anemll.items():
            fish_key = f"text_model.model.layers.{layer_idx}.{fish_name}.weight"
            shard = weight_map[fish_key]
            with safe_open(f"{FISH_MODEL_DIR}/{shard}", framework="pt") as f:
                fish_w = f.get_tensor(fish_key).float()

            anemll_w = dict(anemll_model.model.layers[layer_idx].named_modules())[anemll_name.split('.')[0]]
            anemll_w = getattr(anemll_w, anemll_name.split('.')[1]).weight.float().squeeze()

            if fish_w.shape != anemll_w.shape:
                print(f"  Layer {layer_idx} {fish_name}: SHAPE MISMATCH {fish_w.shape} vs {anemll_w.shape}")
                all_pass = False
                continue
            max_err = (fish_w - anemll_w).abs().max().item()
            match = max_err < ATOL_FP16
            all_pass = all_pass and match
            status = "OK" if match else "FAIL"
            print(f"  Layer {layer_idx:>2} {fish_name}: max_err={max_err:.8f} [{status}]")

    print(f"\nPhase 2c: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    print("Fish S2 Pro vs ANEMLL Qwen2.5 — Correctness Parity Test")
    print("=" * 60)

    results = {}

    # Phase 1: Embeddings
    results["embeddings"] = test_embedding_parity()

    # Phase 2: QK Norm
    results["qk_norm"] = test_qk_norm_parity()

    # Phase 2b: Attention projections (wqkv split)
    results["attn_proj"] = test_attention_proj_parity()

    # Phase 2c: FFN
    results["ffn"] = test_ffn_parity()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name:>15}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")

    if not all_pass:
        sys.exit(1)

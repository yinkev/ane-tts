"""
Fish S2 Pro → ANEMLL Qwen2.5 Adapter

Remaps Fish's slow AR weights to ANEMLL's Qwen2.5 format.

Key differences between Fish and standard Qwen 2.5:
1. Embeddings: Fish uses 'embeddings', ANEMLL uses 'embed_tokens'
2. QK Norm: Fish has per-head RMSNorm on Q and K (attention_qk_norm=true)
3. Fused QKV: Fish uses single 'wqkv', ANEMLL uses separate q_proj/k_proj/v_proj
4. Codebook embeddings: Fish-specific, not included (handled in Python)
5. No attention bias: Fish has attention_bias=false
"""

import os
import sys
import json
import torch
from pathlib import Path

# Fish S2 Pro text model config
FISH_CONFIG = {
    "dim": 2560,
    "n_layer": 36,
    "n_head": 32,
    "n_local_heads": 8,  # KV heads (GQA)
    "head_dim": 128,
    "intermediate_size": 9728,
    "norm_eps": 1e-6,
    "rope_base": 1000000,
    "vocab_size": 155776,
    "max_seq_len": 32768,
}

# Complete weight mapping: Fish key pattern → ANEMLL key pattern
# Per-layer weights use {n} for layer index
FISH_TO_ANEMLL = {
    # Embeddings (not per-layer)
    "text_model.model.embeddings.weight": "model.embed_tokens.weight",
    # Final norm (not per-layer)
    "text_model.model.norm.weight": "model.norm.weight",
    # Attention projections — wqkv is SPLIT into q/k/v (handled specially)
    "text_model.model.layers.{n}.attention.wqkv.weight": "SPLIT_QKV",
    "text_model.model.layers.{n}.attention.wo.weight": "model.layers.{n}.self_attn.o_proj.weight",
    # QK normalization
    "text_model.model.layers.{n}.attention.q_norm.weight": "model.layers.{n}.self_attn.q_norm.weight",
    "text_model.model.layers.{n}.attention.k_norm.weight": "model.layers.{n}.self_attn.k_norm.weight",
    # FFN (SwiGLU)
    "text_model.model.layers.{n}.feed_forward.w1.weight": "model.layers.{n}.mlp.gate_proj.weight",
    "text_model.model.layers.{n}.feed_forward.w2.weight": "model.layers.{n}.mlp.down_proj.weight",
    "text_model.model.layers.{n}.feed_forward.w3.weight": "model.layers.{n}.mlp.up_proj.weight",
    # Layer norms
    "text_model.model.layers.{n}.attention_norm.weight": "model.layers.{n}.input_layernorm.weight",
    "text_model.model.layers.{n}.ffn_norm.weight": "model.layers.{n}.post_attention_layernorm.weight",
}

# Expected tensor counts for verification
EXPECTED_PER_LAYER = 10  # wqkv(→3) + wo + q_norm + k_norm + w1 + w2 + w3 + attn_norm + ffn_norm = 12 output tensors from 10 input
EXPECTED_GLOBAL = 2  # embeddings + norm
EXPECTED_TOTAL_OUTPUT = FISH_CONFIG["n_layer"] * (EXPECTED_PER_LAYER + 2) + EXPECTED_GLOBAL  # +2 for split qkv


def check_weight_compatibility():
    """Verify ALL expected Fish slow AR weights exist in the checkpoint."""
    model_dir = Path.home() / "Models/fish-audio-s2-pro-mlx-bf16"
    index_file = model_dir / "model.safetensors.index.json"

    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    missing = []
    found = []

    # Check global weights
    for fish_key in ["text_model.model.embeddings.weight", "text_model.model.norm.weight"]:
        if fish_key in weight_map:
            found.append(fish_key)
        else:
            missing.append(fish_key)

    # Check per-layer weights
    per_layer_patterns = [k for k in FISH_TO_ANEMLL if "{n}" in k]
    for layer_idx in range(FISH_CONFIG["n_layer"]):
        for pattern in per_layer_patterns:
            fish_key = pattern.format(n=layer_idx)
            if fish_key in weight_map:
                found.append(fish_key)
            else:
                missing.append(fish_key)

    print(f"Weight compatibility check:")
    print(f"  Found: {len(found)} / {len(found) + len(missing)} expected weights")
    if missing:
        print(f"  MISSING ({len(missing)}):")
        for k in missing:
            print(f"    {k}")
        return False
    else:
        print(f"  All {len(found)} weights found")
        return True


def create_qwen_compatible_config():
    """Create a Qwen-format config.json from Fish's config."""
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": FISH_CONFIG["dim"],
        "num_hidden_layers": FISH_CONFIG["n_layer"],
        "num_attention_heads": FISH_CONFIG["n_head"],
        "num_key_value_heads": FISH_CONFIG["n_local_heads"],
        "head_dim": FISH_CONFIG["head_dim"],
        "intermediate_size": FISH_CONFIG["intermediate_size"],
        "rms_norm_eps": FISH_CONFIG["norm_eps"],
        "rope_theta": FISH_CONFIG["rope_base"],
        "vocab_size": FISH_CONFIG["vocab_size"],
        "max_position_embeddings": FISH_CONFIG["max_seq_len"],
        "attention_bias": False,
        "attention_qk_norm": True,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }


def split_fused_qkv(tensor):
    """Split Fish's fused wqkv weight into separate q, k, v projections.

    Fish wqkv shape: [(n_head + 2*n_kv_head) * head_dim, dim]
                   = [(32 + 8 + 8) * 128, 2560]
                   = [6144, 2560]

    Returns: (q_weight, k_weight, v_weight)
    """
    q_dim = FISH_CONFIG["n_head"] * FISH_CONFIG["head_dim"]       # 32*128 = 4096
    kv_dim = FISH_CONFIG["n_local_heads"] * FISH_CONFIG["head_dim"]  # 8*128 = 1024
    expected_total = q_dim + 2 * kv_dim  # 6144

    assert tensor.shape[0] == expected_total, \
        f"wqkv shape mismatch: got {tensor.shape[0]}, expected {expected_total}"

    q_w, k_w, v_w = tensor.split([q_dim, kv_dim, kv_dim], dim=0)
    return q_w, k_w, v_w


def remap_weights_fish_to_qwen(model_dir):
    """Load Fish S2 Pro weights and remap to ANEMLL's Qwen2.5 naming.

    Handles:
    - embeddings.weight → embed_tokens.weight
    - wqkv split into separate q_proj/k_proj/v_proj
    - QK norm weight remapping
    - All FFN and norm weights

    Returns a state_dict compatible with ANEMLL's Qwen2.5 model.
    """
    from safetensors.torch import safe_open

    model_dir = Path(model_dir)
    with open(model_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    # Collect ALL slow AR weights (layers + embeddings + final norm)
    slow_ar_keys = [k for k in weight_map if "text_model.model.layers" in k
                    or k == "text_model.model.norm.weight"
                    or k == "text_model.model.embeddings.weight"]

    remapped = {}
    shards_needed = set(weight_map[k] for k in slow_ar_keys)

    for shard in sorted(shards_needed):
        path = model_dir / shard
        print(f"  Loading {shard}...")
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                if key not in slow_ar_keys:
                    continue

                tensor = f.get_tensor(key)

                # Step 1: Strip Fish prefix
                new_key = key.replace("text_model.model.", "model.")

                # Step 2: Remap embedding name
                new_key = new_key.replace("model.embeddings.weight", "model.embed_tokens.weight")

                # Step 3: Remap attention projections
                if ".attention.wqkv." in new_key:
                    # Split fused QKV into separate projections
                    layer_prefix = new_key.replace(".self_attn.qkv_proj.weight", "")
                    # First do the attention rename
                    layer_prefix = new_key.split(".attention.wqkv.")[0]
                    q_w, k_w, v_w = split_fused_qkv(tensor)
                    remapped[f"{layer_prefix}.self_attn.q_proj.weight"] = q_w
                    remapped[f"{layer_prefix}.self_attn.k_proj.weight"] = k_w
                    remapped[f"{layer_prefix}.self_attn.v_proj.weight"] = v_w
                    continue  # Don't add the fused weight

                new_key = new_key.replace(".attention.wo.", ".self_attn.o_proj.")

                # Step 4: Remap QK norms
                new_key = new_key.replace(".attention.q_norm.", ".self_attn.q_norm.")
                new_key = new_key.replace(".attention.k_norm.", ".self_attn.k_norm.")

                # Step 5: Remap FFN
                new_key = new_key.replace(".feed_forward.w1.", ".mlp.gate_proj.")
                new_key = new_key.replace(".feed_forward.w2.", ".mlp.down_proj.")
                new_key = new_key.replace(".feed_forward.w3.", ".mlp.up_proj.")

                # Step 6: Remap layer norms
                new_key = new_key.replace(".attention_norm.", ".input_layernorm.")
                new_key = new_key.replace(".ffn_norm.", ".post_attention_layernorm.")

                remapped[new_key] = tensor

    # Verification
    print(f"\n  Remapped {len(remapped)} tensors")

    # Check critical weights
    critical = {
        "model.embed_tokens.weight": "Embeddings (+ tied LM head)",
        "model.norm.weight": "Final RMSNorm",
    }
    for layer in range(FISH_CONFIG["n_layer"]):
        critical[f"model.layers.{layer}.self_attn.q_proj.weight"] = f"Layer {layer} Q proj"
        critical[f"model.layers.{layer}.self_attn.k_proj.weight"] = f"Layer {layer} K proj"
        critical[f"model.layers.{layer}.self_attn.v_proj.weight"] = f"Layer {layer} V proj"
        critical[f"model.layers.{layer}.self_attn.o_proj.weight"] = f"Layer {layer} O proj"
        critical[f"model.layers.{layer}.self_attn.q_norm.weight"] = f"Layer {layer} Q norm"
        critical[f"model.layers.{layer}.self_attn.k_norm.weight"] = f"Layer {layer} K norm"
        critical[f"model.layers.{layer}.mlp.gate_proj.weight"] = f"Layer {layer} gate"
        critical[f"model.layers.{layer}.mlp.down_proj.weight"] = f"Layer {layer} down"
        critical[f"model.layers.{layer}.mlp.up_proj.weight"] = f"Layer {layer} up"
        critical[f"model.layers.{layer}.input_layernorm.weight"] = f"Layer {layer} input norm"
        critical[f"model.layers.{layer}.post_attention_layernorm.weight"] = f"Layer {layer} post-attn norm"

    missing_critical = []
    for key, desc in critical.items():
        if key not in remapped:
            missing_critical.append((key, desc))

    if missing_critical:
        print(f"\n  FATAL: {len(missing_critical)} critical weights missing from remapped checkpoint:")
        for key, desc in missing_critical[:10]:
            print(f"    {key} ({desc})")
        if len(missing_critical) > 10:
            print(f"    ... and {len(missing_critical) - 10} more")
        sys.exit(1)
    else:
        print(f"  All {len(critical)} critical weights verified present")

    # Shape audit
    print(f"\n  Shape audit:")
    print(f"    embed_tokens: {remapped['model.embed_tokens.weight'].shape}")
    print(f"    q_proj (L0):  {remapped['model.layers.0.self_attn.q_proj.weight'].shape}")
    print(f"    k_proj (L0):  {remapped['model.layers.0.self_attn.k_proj.weight'].shape}")
    print(f"    v_proj (L0):  {remapped['model.layers.0.self_attn.v_proj.weight'].shape}")
    print(f"    o_proj (L0):  {remapped['model.layers.0.self_attn.o_proj.weight'].shape}")
    print(f"    q_norm (L0):  {remapped['model.layers.0.self_attn.q_norm.weight'].shape}")
    print(f"    k_norm (L0):  {remapped['model.layers.0.self_attn.k_norm.weight'].shape}")
    print(f"    gate (L0):    {remapped['model.layers.0.mlp.gate_proj.weight'].shape}")
    print(f"    norm:         {remapped['model.norm.weight'].shape}")

    return remapped


def save_as_qwen_checkpoint(output_dir):
    """Save Fish's slow AR as a Qwen-compatible checkpoint for ANEMLL."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path.home() / "Models/fish-audio-s2-pro-mlx-bf16"

    # Save config (with attention_qk_norm=True)
    config = create_qwen_compatible_config()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json (attention_qk_norm={config['attention_qk_norm']})")

    # Remap and save weights
    print("Remapping weights...")
    weights = remap_weights_fish_to_qwen(model_dir)

    from safetensors.torch import save_file
    save_file(weights, str(output_dir / "model.safetensors"))
    print(f"\nSaved model.safetensors ({len(weights)} tensors)")

    # Minimal tokenizer config
    tokenizer_config = {
        "model_type": "qwen2",
        "vocab_size": FISH_CONFIG["vocab_size"],
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    print(f"\nCheckpoint saved to {output_dir}")


if __name__ == "__main__":
    print("=== Fish S2 Pro -> ANEMLL Qwen2.5 Checkpoint ===\n")

    # Verify all weights exist in Fish checkpoint
    if not check_weight_compatibility():
        print("\nFATAL: Weight compatibility check failed")
        sys.exit(1)

    print()

    # Save remapped checkpoint
    output_dir = Path("/tmp/fish_slow_ar_qwen_format")
    save_as_qwen_checkpoint(output_dir)

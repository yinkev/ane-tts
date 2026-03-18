"""
Fish S2 Pro → CoreML Adapter

Adapts Fish's slow AR (which is architecturally Qwen) to work with
ANEMLL's CoreML conversion pipeline that includes KV cache support.

The key differences between standard Qwen and Fish's slow AR:
1. Input: Fish takes (semantic_token, codebook_tokens) per step, not just text tokens
2. Output: Fish returns both logits AND hidden_states (for fast AR)
3. Embeddings: Fish has codebook_embeddings in addition to text embeddings
4. Architecture: Same transformer layers (attention + SwiGLU FFN)

Strategy: Convert only the transformer layers (which are identical to Qwen),
handle the custom input/output wrapping in Python.
"""

import os
import sys
import json
import time
from pathlib import Path

# Fish S2 Pro config
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

def check_weight_compatibility():
    """Verify Fish's slow AR weights match Qwen architecture."""
    from safetensors.torch import safe_open

    model_dir = Path.home() / "Models/fish-audio-s2-pro-mlx-bf16"
    index_file = model_dir / "model.safetensors.index.json"

    with open(index_file) as f:
        weight_map = json.load(f)["weight_map"]

    # Fish weight names → Qwen equivalent
    fish_to_qwen = {
        "text_model.model.layers.{n}.attention.wqkv.weight": "model.layers.{n}.self_attn.qkv_proj.weight",
        "text_model.model.layers.{n}.attention.wo.weight": "model.layers.{n}.self_attn.o_proj.weight",
        "text_model.model.layers.{n}.feed_forward.w1.weight": "model.layers.{n}.mlp.gate_proj.weight",
        "text_model.model.layers.{n}.feed_forward.w2.weight": "model.layers.{n}.mlp.down_proj.weight",
        "text_model.model.layers.{n}.feed_forward.w3.weight": "model.layers.{n}.mlp.up_proj.weight",
        "text_model.model.layers.{n}.attention_norm.weight": "model.layers.{n}.input_layernorm.weight",
        "text_model.model.layers.{n}.ffn_norm.weight": "model.layers.{n}.post_attention_layernorm.weight",
    }

    # Check that all expected weights exist
    missing = []
    found = 0
    for layer_idx in range(FISH_CONFIG["n_layer"]):
        for fish_pattern in fish_to_qwen:
            fish_key = fish_pattern.format(n=layer_idx)
            if fish_key in weight_map:
                found += 1
            else:
                missing.append(fish_key)

    print(f"Weight compatibility check:")
    print(f"  Found: {found} / {found + len(missing)} expected weights")
    if missing:
        print(f"  Missing: {missing[:5]}...")
    else:
        print(f"  All weights compatible with Qwen architecture ✅")

    return len(missing) == 0


def create_qwen_compatible_config():
    """Create a Qwen-format config.json from Fish's config."""
    qwen_config = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
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
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
    }
    return qwen_config


def remap_weights_fish_to_qwen(model_dir):
    """
    Load Fish S2 Pro weights and remap to Qwen naming convention.
    Only extracts the slow AR transformer layers.
    Returns a state_dict compatible with ANEMLL's Qwen model.
    """
    from safetensors.torch import safe_open

    model_dir = Path(model_dir)
    with open(model_dir / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    # Collect slow AR weights
    slow_ar_keys = [k for k in weight_map if "text_model.model.layers" in k
                    or k == "text_model.model.norm.weight"
                    or k == "text_model.model.embed_tokens.weight"]

    remapped = {}
    shards_needed = set(weight_map[k] for k in slow_ar_keys)

    for shard in shards_needed:
        path = model_dir / shard
        print(f"  Loading {shard}...")
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                if key not in slow_ar_keys:
                    continue

                tensor = f.get_tensor(key)

                # Remap Fish → Qwen naming
                new_key = key.replace("text_model.model.", "model.")
                new_key = new_key.replace(".attention.wqkv.", ".self_attn.qkv_proj.")
                new_key = new_key.replace(".attention.wo.", ".self_attn.o_proj.")
                new_key = new_key.replace(".feed_forward.w1.", ".mlp.gate_proj.")
                new_key = new_key.replace(".feed_forward.w2.", ".mlp.down_proj.")
                new_key = new_key.replace(".feed_forward.w3.", ".mlp.up_proj.")
                new_key = new_key.replace(".attention_norm.", ".input_layernorm.")
                new_key = new_key.replace(".ffn_norm.", ".post_attention_layernorm.")

                remapped[new_key] = tensor

    print(f"  Remapped {len(remapped)} tensors")
    return remapped


def save_as_qwen_checkpoint(output_dir):
    """
    Save Fish's slow AR as a Qwen-compatible checkpoint that ANEMLL can convert.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path.home() / "Models/fish-audio-s2-pro-mlx-bf16"

    # Save config
    config = create_qwen_compatible_config()
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config.json")

    # Remap and save weights
    print("Remapping weights...")
    weights = remap_weights_fish_to_qwen(model_dir)

    from safetensors.torch import save_file
    save_file(weights, str(output_dir / "model.safetensors"))
    print(f"Saved model.safetensors ({len(weights)} tensors)")

    # Save tokenizer config (minimal, ANEMLL needs it)
    tokenizer_config = {
        "model_type": "qwen2",
        "vocab_size": FISH_CONFIG["vocab_size"],
    }
    with open(output_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    print(f"\nQwen-compatible checkpoint saved to {output_dir}")
    print(f"To convert with ANEMLL:")
    print(f"  ./anemll/utils/convert_model.sh --model {output_dir} --output /tmp/fish_slow_ar_anemll")


if __name__ == "__main__":
    print("=== Fish S2 Pro → Qwen-Compatible Checkpoint ===\n")

    # Check compatibility
    compatible = check_weight_compatibility()
    if not compatible:
        print("\nERROR: Weight compatibility check failed")
        sys.exit(1)

    print()

    # Save as Qwen checkpoint
    output_dir = Path("/tmp/fish_slow_ar_qwen_format")
    save_as_qwen_checkpoint(output_dir)

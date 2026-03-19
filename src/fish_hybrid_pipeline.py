"""
Hybrid Fish S2 Pro pipeline: llama.cpp slow AR + MLX fast AR + codec.

Architecture:
  Python: text -> tokenize -> embed (text + codebook fusion) -> raw embedding
  llama.cpp: raw embedding -> 36 transformer layers -> logits + hidden state
  MLX: hidden state -> fast AR (10 codebook calls) -> codebook tokens
  MLX: codebook tokens -> codec decoder -> audio waveform

Uses llama.cpp's batch.embd to feed pre-computed embeddings,
bypassing the GGUF's embedding layer. This preserves Fish's
multi-codebook embedding fusion while using llama.cpp's optimized
Metal kernels for the transformer compute.
"""

import time
import numpy as np
import ctypes
from pathlib import Path

# Check imports
try:
    from llama_cpp import Llama, llama_cpp
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def test_embedding_bypass():
    """Test: can we feed raw embeddings to llama.cpp and get logits?"""
    if not HAS_LLAMA:
        print("llama-cpp-python not installed")
        return

    print("=== Test: Embedding Bypass via llama.cpp ===")
    gguf_path = "/tmp/fish_slow_ar_q8_0.gguf"

    # Load model
    print(f"Loading {gguf_path}...")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=512,
        n_gpu_layers=99,
        embedding=True,  # Enable embedding mode
        verbose=False,
    )

    # Get model's embedding dimension
    n_embd = llama_cpp.llama_n_embd(llm.model)
    print(f"Embedding dim: {n_embd}")

    # Test 1: Normal token input
    print("\nTest 1: Normal token forward pass...")
    llm.reset()
    llm.eval([100])  # Feed token 100
    logits_normal = llm.eval_logits
    if logits_normal:
        logits_arr = np.array(logits_normal[-1][:10])
        print(f"  Logits (first 10): {logits_arr}")

    # Test 2: Feed the same token's embedding directly
    print("\nTest 2: Embedding bypass...")
    # Get the embedding for token 100
    llm.reset()

    # Use low-level API to feed embedding
    # llama_batch supports embd field
    import ctypes

    # Create a batch with embedding instead of token
    batch = llama_cpp.llama_batch_init(1, n_embd, 1)

    # Get token 100's embedding from the model
    # First, eval token 100 to get its embedding
    # Actually, we need to feed a raw embedding vector
    # Let's create a random one first to test the API
    embd = np.random.randn(n_embd).astype(np.float32)

    # Set up batch with embedding
    batch.n_tokens = 1
    batch.embd = ctypes.cast(
        embd.ctypes.data,
        ctypes.POINTER(ctypes.c_float)
    )
    batch.token = None
    batch.pos[0] = 0
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = 0
    batch.logits[0] = True

    # Decode
    ret = llama_cpp.llama_decode(llm.ctx, batch)
    print(f"  llama_decode returned: {ret}")

    if ret == 0:
        # Get logits
        logits_ptr = llama_cpp.llama_get_logits(llm.ctx)
        n_vocab = llama_cpp.llama_n_vocab(llm.model)
        logits_embd = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
        print(f"  Logits shape: {logits_embd.shape}")
        print(f"  Logits (first 10): {logits_embd[:10]}")
        print(f"  Logits max: {logits_embd.max():.4f}, argmax: {logits_embd.argmax()}")
        print("  EMBEDDING BYPASS WORKS!")
    else:
        print(f"  FAILED with code {ret}")

    llama_cpp.llama_batch_free(batch)
    del llm


if __name__ == "__main__":
    test_embedding_bypass()

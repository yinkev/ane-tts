"""
Fish S2 Pro end-to-end pipeline: llama.cpp slow AR + MLX fast AR + codec.

This script generates actual audio from text using:
1. llama.cpp GGUF for the slow AR (12.9ms/tok Q6_K)
2. MLX (mlx-audio's Fish implementation) for the fast AR + codec
3. Outputs a WAV file

First milestone: produce audio and verify quality.
"""

import sys
import time
import numpy as np
from pathlib import Path

# We need to test if we can load the GGUF and get logits
def test_slow_ar_gguf():
    """Test that the slow AR GGUF loads and produces logits via llama-cpp-python."""
    from llama_cpp import Llama

    gguf_path = "/tmp/fish_slow_ar_q8_0.gguf"
    print(f"Loading {gguf_path}...")
    t0 = time.time()

    llm = Llama(
        model_path=gguf_path,
        n_ctx=512,        # Small context for testing
        n_gpu_layers=99,  # All layers on Metal GPU
        verbose=False,
    )
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Test: feed a token and get logits
    # Fish's semantic tokens are in the 151K-155K range
    # Token 100 is a text token (Qwen BPE)
    test_tokens = [100, 42, 200]

    print(f"\nTesting forward pass with tokens: {test_tokens}")

    # Reset KV cache
    llm.reset()

    # Evaluate tokens
    llm.eval(test_tokens)

    # Get logits for the last token
    # llama-cpp-python stores logits internally
    # We need to access them via the scores

    # Generate one token to test
    t0 = time.time()
    output = llm.create_completion(
        prompt=test_tokens,  # Raw token IDs
        max_tokens=1,
        temperature=0.8,
        top_p=0.9,
        # logprobs=5,  # Needs logits_all=True
    )
    dt = (time.time() - t0) * 1000

    print(f"Generated in {dt:.1f}ms")
    print(f"Output: {output}")

    # Now test autoregressive generation of semantic tokens
    print(f"\nGenerating 20 tokens autoregressively...")
    t0 = time.time()
    output = llm.create_completion(
        prompt=test_tokens,
        max_tokens=20,
        temperature=0.8,
        top_p=0.9,
    )
    dt = (time.time() - t0) * 1000

    text = output["choices"][0]["text"]
    tokens_generated = output["usage"]["completion_tokens"]
    ms_per_tok = dt / max(tokens_generated, 1)

    print(f"Generated {tokens_generated} tokens in {dt:.0f}ms ({ms_per_tok:.1f}ms/tok)")
    print(f"Output text: {repr(text[:200])}")

    return llm


def test_generation_speed():
    """Benchmark autoregressive generation speed."""
    from llama_cpp import Llama

    for quant, path in [
        ("Q8_0", "/tmp/fish_slow_ar_q8_0.gguf"),
        ("Q6_K", "/tmp/fish_slow_ar_q6_k.gguf"),
    ]:
        if not Path(path).exists():
            print(f"Skipping {quant} — file not found")
            continue

        print(f"\n=== {quant} ===")
        llm = Llama(
            model_path=path,
            n_ctx=512,
            n_gpu_layers=99,
            verbose=False,
        )

        # Warm up
        llm.create_completion(prompt=[100], max_tokens=5, temperature=0.8)

        # Benchmark
        prompt = [100, 42, 200, 300, 400]  # 5-token prompt

        t0 = time.time()
        output = llm.create_completion(
            prompt=prompt,
            max_tokens=50,
            temperature=0.8,
            top_p=0.9,
        )
        dt = time.time() - t0

        n_tok = output["usage"]["completion_tokens"]
        ms_per_tok = (dt / max(n_tok, 1)) * 1000
        tok_per_sec = n_tok / dt if dt > 0 else 0

        print(f"  {n_tok} tokens in {dt:.2f}s")
        print(f"  {tok_per_sec:.1f} tok/s ({ms_per_tok:.1f} ms/tok)")
        print(f"  Slow AR RTF: {46.4 / ms_per_tok:.2f}x (46.4ms audio / {ms_per_tok:.1f}ms gen)")

        del llm


if __name__ == "__main__":
    print("=== Fish S2 Pro Pipeline Test ===\n")

    # Test 1: Basic GGUF loading and forward pass
    print("--- Test 1: GGUF Forward Pass ---")
    llm = test_slow_ar_gguf()
    del llm

    # Test 2: Generation speed benchmark
    print("\n--- Test 2: Generation Speed ---")
    test_generation_speed()

    print("\n=== Done ===")

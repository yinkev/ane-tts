"""
Test: llama.cpp batch.embd API for feeding pre-computed embeddings.

Validates that feeding a token's embedding directly via batch.embd produces
the same logits as feeding the token id via batch.token.

Steps:
  1. Extract token 100's embedding from the GGUF embedding table (dequantized Q8_0)
  2. Load model, decode token 100 normally via batch.token -> baseline logits
  3. Clear KV cache, feed same embedding via batch.embd -> test logits
  4. Compare — should be bit-identical

Key API usage:
  - llama_batch_init(n_tokens, n_embd, n_seq_max) with n_embd > 0 allocates batch.embd
  - batch.token will be NULL; batch.embd is a float buffer of n_tokens * n_embd
  - Use ctypes.memmove to copy data INTO the allocated buffer (do NOT reassign the pointer)
  - Set batch.n_tokens, pos, n_seq_id, seq_id, logits as normal
"""

import ctypes
import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
from gguf.constants import GGMLQuantizationType
from llama_cpp import Llama
from llama_cpp import llama_cpp


GGUF_PATH = "/tmp/fish_slow_ar_q8_0.gguf"
TEST_TOKEN = 100
N_CTX = 512


def extract_token_embedding(gguf_path: str, token_id: int) -> np.ndarray:
    """Extract a single token's embedding from the GGUF file's embedding table.

    Reads the quantized token_embd.weight tensor, dequantizes it, and returns
    the row corresponding to token_id as a float32 vector of shape (n_embd,).
    """
    reader = GGUFReader(gguf_path)

    embd_tensor = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            embd_tensor = t
            break
    if embd_tensor is None:
        raise RuntimeError("token_embd.weight not found in GGUF")

    qtype = GGMLQuantizationType(embd_tensor.tensor_type)
    n_embd = embd_tensor.shape[0]   # embedding dimension
    n_vocab = embd_tensor.shape[1]  # vocabulary size
    print(f"  Embedding table: {n_vocab} tokens x {n_embd} dims ({qtype.name})")

    # Dequantize and extract the row for our token
    all_embeddings = dequantize(embd_tensor.data, qtype).reshape(n_vocab, n_embd)
    token_embd = all_embeddings[token_id].copy().astype(np.float32)
    print(f"  Token {token_id} embedding: norm={np.linalg.norm(token_embd):.4f}, "
          f"range=[{token_embd.min():.4f}, {token_embd.max():.4f}]")

    return token_embd


def read_logits(ctx, n_vocab: int) -> np.ndarray:
    """Copy logits from llama.cpp's internal buffer into a Python-owned numpy array.

    Uses ctypes.memmove for a true deep copy — np.ctypeslib.as_array().copy()
    can silently produce stale data on some numpy/Python version combinations.
    """
    logits_ptr = llama_cpp.llama_get_logits_ith(ctx, 0)
    logits = np.empty(n_vocab, dtype=np.float32)
    ctypes.memmove(logits.ctypes.data, logits_ptr, n_vocab * 4)
    return logits


def main():
    print("=" * 60)
    print("Test: llama.cpp batch.embd API")
    print("=" * 60)

    # ── Extract embedding ──
    print(f"\n--- Extracting embedding for token {TEST_TOKEN} from GGUF ---")
    token_embedding = extract_token_embedding(GGUF_PATH, TEST_TOKEN)

    # ── Load model once ──
    print(f"\nLoading model...")
    llm = Llama(model_path=GGUF_PATH, n_ctx=N_CTX, n_gpu_layers=99, verbose=False)

    n_embd = llama_cpp.llama_model_n_embd(llm.model)
    vocab = llama_cpp.llama_model_get_vocab(llm.model)
    n_vocab = llama_cpp.llama_vocab_n_tokens(vocab)
    print(f"  n_embd={n_embd}, n_vocab={n_vocab}")
    assert token_embedding.shape == (n_embd,)

    # ── Path A: Normal token decode ──
    print(f"\n--- Path A: decode token {TEST_TOKEN} via batch.token ---")

    batch_tok = llama_cpp.llama_batch_init(1, 0, 1)  # embd=0 -> token mode
    batch_tok.n_tokens = 1
    batch_tok.token[0] = TEST_TOKEN
    batch_tok.pos[0] = 0
    batch_tok.n_seq_id[0] = 1
    batch_tok.seq_id[0][0] = 0
    batch_tok.logits[0] = 1

    ret = llama_cpp.llama_decode(llm.ctx, batch_tok)
    assert ret == 0, f"llama_decode (token) failed: {ret}"
    logits_token = read_logits(llm.ctx, n_vocab)
    llama_cpp.llama_batch_free(batch_tok)

    print(f"  Logits[:8]: {logits_token[:8]}")
    print(f"  Argmax={int(logits_token.argmax())}, max={logits_token.max():.4f}")

    # Clear KV cache between runs
    llama_cpp.llama_kv_self_clear(llm.ctx)

    # ── Path B: Embedding decode via batch.embd ──
    print(f"\n--- Path B: decode embedding via batch.embd ---")

    batch_embd = llama_cpp.llama_batch_init(1, n_embd, 1)  # embd=n_embd -> embd mode
    print(f"  batch.token is NULL: {not batch_embd.token}")
    print(f"  batch.embd is allocated: {bool(batch_embd.embd)}")

    batch_embd.n_tokens = 1
    batch_embd.pos[0] = 0
    batch_embd.n_seq_id[0] = 1
    batch_embd.seq_id[0][0] = 0
    batch_embd.logits[0] = 1

    # Copy embedding data into the C-allocated buffer
    ctypes.memmove(
        batch_embd.embd,
        token_embedding.ctypes.data,
        n_embd * ctypes.sizeof(ctypes.c_float)
    )

    ret = llama_cpp.llama_decode(llm.ctx, batch_embd)
    assert ret == 0, f"llama_decode (embd) failed: {ret}"
    logits_embd = read_logits(llm.ctx, n_vocab)
    llama_cpp.llama_batch_free(batch_embd)

    print(f"  Logits[:8]: {logits_embd[:8]}")
    print(f"  Argmax={int(logits_embd.argmax())}, max={logits_embd.max():.4f}")

    # ── Compare ──
    print(f"\n--- Comparison ---")

    identical = np.array_equal(logits_token, logits_embd)
    print(f"  Bitwise identical: {identical}")

    if identical:
        print(f"  Argmax: {int(logits_embd.argmax())}")
        idx = np.argpartition(logits_embd, -5)[-5:]
        top5 = idx[np.argsort(logits_embd[idx])[::-1]].tolist()
        print(f"  Top-5 token ids: {top5}")
        cos_sim = 1.0
    else:
        abs_diff = np.abs(logits_token.astype(np.float64) - logits_embd.astype(np.float64))
        print(f"  Max absolute diff:  {abs_diff.max():.6e}")
        print(f"  Mean absolute diff: {abs_diff.mean():.6e}")
        print(f"  Argmax (token): {int(logits_token.argmax())}")
        print(f"  Argmax (embd):  {int(logits_embd.argmax())}")
        a, b = logits_token.astype(np.float64), logits_embd.astype(np.float64)
        cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  Cosine similarity: {cos_sim:.8f}")

    # Verdict
    if cos_sim > 0.99:
        print(f"\n  PASS -- batch.embd produces matching logits (cosine={cos_sim:.6f})")
    elif cos_sim > 0.95:
        print(f"\n  PARTIAL -- logits roughly match (cosine={cos_sim:.6f})")
    else:
        print(f"\n  FAIL -- logits diverge (cosine={cos_sim:.6f})")

    del llm


if __name__ == "__main__":
    main()

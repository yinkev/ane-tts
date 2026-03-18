"""
CoreML Inference Engine for Fish S2 Pro Slow AR

Loads ANEMLL-converted CoreML models and provides an inference interface
compatible with Fish's generation loop. The FFN (transformer layers) run
on CoreML while embeddings and sampling stay in Python/PyTorch.

Architecture:
  Python: text → tokenize → embed (Fish's multi-codebook combo)
  CoreML: hidden_states → FFN chunk 1→2→3→4 → output_hidden_states
  CoreML: output_hidden_states → LM head → logits
  Python: logits → sample → fast AR → codebook codes

Usage:
  engine = CoreMLSlowAR("/tmp/fish_slow_ar_anemll")
  engine.setup()
  # Prefill
  engine.prefill(hidden_states, position_ids, causal_mask)
  # Decode loop
  for pos in range(start_pos, max_tokens):
      logits, hidden = engine.decode_one(hidden_states, pos)
      next_token = sample(logits)
"""

import os
import time
import glob
import numpy as np

try:
    import coremltools as ct
    HAS_COREML = True
except ImportError:
    HAS_COREML = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class CoreMLSlowAR:
    """CoreML inference engine for Fish S2 Pro's slow AR transformer."""

    def __init__(self, model_dir, compute_unit="ALL"):
        self.model_dir = model_dir
        self.compute_unit = getattr(ct.ComputeUnit, compute_unit) if HAS_COREML else None

        # Models
        self.embed_model = None
        self.ffn_chunks = []  # List of (infer_model, prefill_model) per chunk
        self.lm_head_model = None

        # State
        self.states = []  # KV cache state per chunk
        self.num_chunks = 0
        self.context_length = 128
        self.batch_size = 64
        self.state_length = 256
        self.hidden_size = 2560
        self.vocab_size = 155776
        self.num_logit_chunks = 16

        # Timing
        self.timings = {"embed": [], "ffn": [], "lm_head": [], "total": []}

    def setup(self):
        """Load all models and create KV cache state."""
        if not HAS_COREML:
            raise RuntimeError("coremltools not installed")

        # Load meta.yaml if it exists
        meta_path = os.path.join(self.model_dir, "meta.yaml")
        if HAS_YAML and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            params = meta.get("model_info", {}).get("parameters", {})
            self.context_length = params.get("context_length", self.context_length)
            self.batch_size = params.get("batch_size", self.batch_size)
            self.num_chunks = params.get("num_chunks", 4)
            self.num_logit_chunks = params.get("split_lm_head", 16)

        # Load embeddings
        embed_path = os.path.join(self.model_dir, "fish_embeddings.mlpackage")
        if os.path.exists(embed_path):
            print(f"Loading embeddings from {embed_path}...")
            self.embed_model = ct.models.MLModel(embed_path, compute_units=self.compute_unit)
            print(f"  Embeddings loaded")

        # Load FFN chunks (combined FFN+Prefill multi-function models)
        chunk_pattern = os.path.join(self.model_dir, "fish_FFN_PF_lut4_chunk_*of*.mlpackage")
        chunk_paths = sorted(glob.glob(chunk_pattern))
        if not chunk_paths:
            # Try non-combined chunks
            chunk_pattern = os.path.join(self.model_dir, "fish_FFN_lut4_chunk_*of*.mlpackage")
            chunk_paths = sorted(glob.glob(chunk_pattern))

        self.num_chunks = len(chunk_paths)
        print(f"Loading {self.num_chunks} FFN chunks...")
        for path in chunk_paths:
            print(f"  Loading {os.path.basename(path)}...")
            # Multi-function: load with function_name for decode vs prefill
            try:
                infer_model = ct.models.MLModel(path, compute_units=self.compute_unit,
                                                 function_name="infer")
                prefill_model = ct.models.MLModel(path, compute_units=self.compute_unit,
                                                   function_name="prefill")
                self.ffn_chunks.append({"infer": infer_model, "prefill": prefill_model})
            except Exception:
                # Single function model (no multi-function support)
                model = ct.models.MLModel(path, compute_units=self.compute_unit)
                self.ffn_chunks.append({"infer": model, "prefill": model})
            print(f"  Loaded")

        # Load LM head
        lm_path = os.path.join(self.model_dir, "fish_lm_head_lut6.mlpackage")
        if os.path.exists(lm_path):
            print(f"Loading LM head from {lm_path}...")
            self.lm_head_model = ct.models.MLModel(lm_path, compute_units=self.compute_unit)
            print(f"  LM head loaded")

        # Create KV cache state for each chunk
        print("Creating KV cache state...")
        self.states = []
        for chunk in self.ffn_chunks:
            # Use prefill model to create state (has same state structure)
            model = chunk.get("prefill") or chunk.get("infer")
            try:
                state = model.make_state()
                self.states.append(state)
            except Exception as e:
                print(f"  Warning: Could not create state: {e}")
                self.states.append(None)

        print(f"Setup complete: {self.num_chunks} FFN chunks, "
              f"ctx={self.context_length}, batch={self.batch_size}")

    def reset_state(self):
        """Reset KV cache state for a new sequence."""
        for i, chunk in enumerate(self.ffn_chunks):
            model = chunk.get("prefill") or chunk.get("infer")
            try:
                self.states[i] = model.make_state()
            except Exception:
                pass

    def _run_ffn_decode(self, hidden_states, position, causal_mask=None):
        """Run hidden states through all FFN chunks in decode (single token) mode.

        Args:
            hidden_states: numpy array [1, 1, hidden_size] float16
            position: int, current position in sequence
            causal_mask: numpy array [1, 1, 1, state_length] float16, or None for auto

        Returns:
            output_hidden_states: numpy array [1, 1, hidden_size] float16
        """
        if causal_mask is None:
            causal_mask = np.zeros((1, 1, 1, self.state_length), dtype=np.float16)
            # Mask future positions
            if position + 1 < self.state_length:
                causal_mask[0, 0, 0, position + 1:] = -1e4  # Large negative for softmax

        h = hidden_states
        for i, chunk in enumerate(self.ffn_chunks):
            inputs = {
                "hidden_states": h.astype(np.float16),
                "position_ids": np.array([position], dtype=np.int32),
                "causal_mask": causal_mask.astype(np.float16),
                "current_pos": np.array([position], dtype=np.int32),
            }
            state = self.states[i] if i < len(self.states) else None
            if state is not None:
                output = chunk["infer"].predict(inputs, state)
            else:
                output = chunk["infer"].predict(inputs)
            h = output["output_hidden_states"]

        return h

    def _run_ffn_prefill(self, hidden_states, position_ids, causal_mask, current_pos):
        """Run hidden states through all FFN chunks in prefill (batch) mode.

        Args:
            hidden_states: numpy array [1, batch_size, hidden_size] float16
            position_ids: numpy array [batch_size] int32
            causal_mask: numpy array [1, 1, batch_size, state_length] float16
            current_pos: int, starting position of this batch

        Returns:
            output_hidden_states: numpy array [1, batch_size, hidden_size] float16
        """
        h = hidden_states
        for i, chunk in enumerate(self.ffn_chunks):
            inputs = {
                "hidden_states": h.astype(np.float16),
                "position_ids": position_ids.astype(np.int32),
                "causal_mask": causal_mask.astype(np.float16),
                "current_pos": np.array([current_pos], dtype=np.int32),
            }
            state = self.states[i] if i < len(self.states) else None
            if state is not None:
                output = chunk["prefill"].predict(inputs, state)
            else:
                output = chunk["prefill"].predict(inputs)
            h = output["output_hidden_states"]

        return h

    def _run_lm_head(self, hidden_states):
        """Run hidden states through LM head to get logits.

        Args:
            hidden_states: numpy array [1, 1, hidden_size] float16

        Returns:
            logits: numpy array [1, 1, vocab_size] float32
        """
        output = self.lm_head_model.predict({
            "hidden_states": hidden_states.astype(np.float16)
        })

        # Combine split logits
        logit_parts = []
        for i in range(1, self.num_logit_chunks + 1):
            key = f"logits{i}"
            if key in output:
                logit_parts.append(output[key])

        if logit_parts:
            logits = np.concatenate(logit_parts, axis=-1)
        elif "output_logits" in output:
            logits = output["output_logits"]
        else:
            # Try first output key
            logits = list(output.values())[0]

        return logits.astype(np.float32)

    def embed(self, input_ids):
        """Run embeddings model.

        Args:
            input_ids: numpy array [1, seq_len] int32

        Returns:
            hidden_states: numpy array [1, seq_len, hidden_size] float16
        """
        if self.embed_model is None:
            raise RuntimeError("Embeddings model not loaded")
        output = self.embed_model.predict({"input_ids": input_ids.astype(np.int32)})
        return output["hidden_states"]

    def decode_one(self, hidden_states, position):
        """Run a single decode step: FFN + LM head.

        This is the hot path called for every token.

        Args:
            hidden_states: numpy array [1, 1, hidden_size] float16
                (output of Fish's embedding combination)
            position: int, current position in KV cache

        Returns:
            logits: numpy array [1, 1, vocab_size] float32
            output_hidden: numpy array [1, 1, hidden_size] float16
                (for passing to fast AR)
        """
        t0 = time.time()

        # FFN chunks (transformer layers with KV cache)
        t1 = time.time()
        output_hidden = self._run_ffn_decode(hidden_states, position)
        t2 = time.time()

        # LM head (logits)
        logits = self._run_lm_head(output_hidden)
        t3 = time.time()

        self.timings["ffn"].append((t2 - t1) * 1000)
        self.timings["lm_head"].append((t3 - t2) * 1000)
        self.timings["total"].append((t3 - t0) * 1000)

        return logits, output_hidden

    def prefill(self, hidden_states, start_pos=0):
        """Prefill the KV cache with a sequence of hidden states.

        Processes in batches of self.batch_size.

        Args:
            hidden_states: numpy array [1, seq_len, hidden_size] float16
            start_pos: int, starting position in KV cache
        """
        seq_len = hidden_states.shape[1]
        pos = start_pos

        while pos < start_pos + seq_len:
            batch_end = min(pos + self.batch_size, start_pos + seq_len)
            batch_len = batch_end - pos
            batch_hidden = hidden_states[:, pos - start_pos:batch_end - start_pos, :]

            # Pad if needed
            if batch_len < self.batch_size:
                padded = np.zeros((1, self.batch_size, self.hidden_size), dtype=np.float16)
                padded[:, :batch_len, :] = batch_hidden
                batch_hidden = padded

            # Position IDs
            position_ids = np.arange(pos, pos + self.batch_size, dtype=np.int32)

            # Causal mask
            causal_mask = np.zeros((1, 1, self.batch_size, self.state_length), dtype=np.float16)
            for i in range(self.batch_size):
                actual_pos = pos + i
                if actual_pos + 1 < self.state_length:
                    causal_mask[0, 0, i, actual_pos + 1:] = -1e4

            self._run_ffn_prefill(batch_hidden, position_ids, causal_mask, pos)
            pos = batch_end

    def get_timing_stats(self):
        """Return timing statistics."""
        stats = {}
        for key, times in self.timings.items():
            if times:
                stats[key] = {
                    "mean_ms": np.mean(times),
                    "std_ms": np.std(times),
                    "min_ms": np.min(times),
                    "max_ms": np.max(times),
                    "count": len(times),
                }
        return stats

    def print_timing_stats(self):
        """Print timing statistics."""
        stats = self.get_timing_stats()
        print("\nTiming Statistics:")
        print(f"{'Component':<12} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Count':>6}")
        print("-" * 54)
        for key in ["ffn", "lm_head", "total"]:
            if key in stats:
                s = stats[key]
                print(f"{key:<12} {s['mean_ms']:>7.2f}ms {s['std_ms']:>7.2f}ms "
                      f"{s['min_ms']:>7.2f}ms {s['max_ms']:>7.2f}ms {s['count']:>6}")


def benchmark_decode_loop(engine, num_tokens=100, warmup=10):
    """Run a decode loop benchmark with dummy data.

    This measures the core token generation speed without
    Fish's embedding logic or sampling overhead.
    """
    print(f"\n{'='*60}")
    print(f"Decode Loop Benchmark ({num_tokens} tokens, {warmup} warmup)")
    print(f"{'='*60}")

    # Dummy hidden state (as if from Fish's embedding combination)
    hidden = np.random.randn(1, 1, engine.hidden_size).astype(np.float16)

    # Warmup
    print("Warming up...")
    for i in range(warmup):
        engine.decode_one(hidden, position=i)

    # Reset timings
    engine.timings = {"embed": [], "ffn": [], "lm_head": [], "total": []}

    # Benchmark
    print(f"Running {num_tokens} decode steps...")
    t_start = time.time()
    for i in range(num_tokens):
        logits, output_hidden = engine.decode_one(hidden, position=warmup + i)
        # In real usage, we'd sample from logits and re-embed
    t_total = time.time() - t_start

    tokens_per_sec = num_tokens / t_total
    ms_per_token = (t_total / num_tokens) * 1000

    print(f"\nResults:")
    print(f"  Total time: {t_total:.2f}s")
    print(f"  Tokens/sec: {tokens_per_sec:.1f}")
    print(f"  ms/token: {ms_per_token:.2f}")
    print(f"  RTF estimate: {46.4 / ms_per_token:.2f}x "
          f"(46.4ms audio per semantic token / {ms_per_token:.2f}ms generation)")

    engine.print_timing_stats()

    print(f"\nComparison:")
    print(f"  MLX baseline:     34.7 ms/token  (0.69x RTF)")
    print(f"  CoreML GPU (raw): 23.8 ms/token  (0.84x RTF)")
    print(f"  This run:         {ms_per_token:.1f} ms/token  ({46.4/ms_per_token:.2f}x RTF)")


if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fish_slow_ar_anemll"
    compute = sys.argv[2] if len(sys.argv) > 2 else "ALL"

    engine = CoreMLSlowAR(model_dir, compute_unit=compute)
    engine.setup()
    benchmark_decode_loop(engine, num_tokens=100, warmup=10)

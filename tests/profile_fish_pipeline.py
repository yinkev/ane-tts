"""
Profile Fish S2 Pro BF16 generation pipeline to understand where time is spent.

Monkey-patches _generate_codes_for_batch to add per-component timing:
  - Slow AR prefill (initial forward pass)
  - Slow AR decode (per-token next-step forward)
  - Semantic sampling (logit bias + top-p/top-k + RAS)
  - Fast AR total (fast prefill + 9 codebook iterations)
  - Each individual fast AR codebook call
  - Codec decode (DAC vocoder)

Also checks whether nn.quantize with default predicate reaches fast_layers.
"""

import time
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
class Timer:
    """Wall-clock timer that calls mx.eval() before stopping to include GPU work."""

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self, *eval_args):
        """Stop timer. Optionally mx.eval() tensors to sync GPU before recording."""
        if eval_args:
            mx.eval(*eval_args)
        elapsed = time.perf_counter() - self.start_time
        self.start_time = None
        return elapsed


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading Fish S2 Pro BF16...")
t0 = time.perf_counter()
from mlx_audio.tts.utils import load
model = load("mlx-community/fish-audio-s2-pro-bf16")
mx.eval(model.parameters())
print(f"Model loaded in {time.perf_counter() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Check quantization reachability of fast_layers
# ---------------------------------------------------------------------------
def check_quantization_reach():
    """
    Simulate what nn.quantize would do with the default class_predicate
    (the one used in mlx_audio convert) and report which fast_layers
    modules would be quantized.
    """
    print("\n" + "=" * 70)
    print("QUANTIZATION REACHABILITY ANALYSIS")
    print("=" * 70)

    # Default predicate from mlx_audio/utils.py apply_quantization
    # plus the convert() predicate: hasattr(m, "to_quantized") and weight % 64 == 0
    group_size = 64

    def default_predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if hasattr(module, "weight") and module.weight.shape[-1] % group_size != 0:
            return False
        return True

    # Walk model tree
    fast_quantizable = []
    slow_quantizable = []
    other_quantizable = []

    for path, module in model.model.named_modules():
        if not default_predicate(path, module):
            continue
        if not hasattr(module, "weight"):
            continue
        shape = tuple(module.weight.shape)
        info = f"  {path:60s} weight={shape}"

        if "fast_layers" in path or path.startswith("fast_"):
            fast_quantizable.append(info)
        elif "layers." in path:
            slow_quantizable.append(info)
        else:
            other_quantizable.append(info)

    print(f"\nFast AR modules that WOULD be quantized ({len(fast_quantizable)}):")
    for s in fast_quantizable:
        print(s)
    print(f"\nSlow AR modules that WOULD be quantized ({len(slow_quantizable)}):")
    print(f"  (total: {len(slow_quantizable)} -- not listing individually)")
    print(f"\nOther quantizable modules ({len(other_quantizable)}):")
    for s in other_quantizable:
        print(s)

    if fast_quantizable:
        print("\n--> CONCLUSION: fast_layers ARE reached by default quantization.")
        print("    nn.quantize(model, bits=8, group_size=64) WILL quantize them.")
    else:
        print("\n--> CONCLUSION: fast_layers are NOT reached by default quantization.")

    # Also check fast_project_in and fast_output
    for name in ["fast_project_in", "fast_output"]:
        mod = getattr(model.model, name, None)
        if mod is not None and hasattr(mod, "weight"):
            shape = tuple(mod.weight.shape)
            q = default_predicate(name, mod)
            print(f"    {name}: weight={shape}, quantizable={q}")


check_quantization_reach()


# ---------------------------------------------------------------------------
# Monkey-patched generation with timing
# ---------------------------------------------------------------------------
timing_data = defaultdict(list)
fast_ar_per_codebook = defaultdict(list)  # codebook_index -> [times]


def patched_generate_codes_for_batch(
    self,
    conversation,
    batch_text,
    max_new_tokens,
    top_p,
    top_k,
    temperature,
):
    """Drop-in replacement with timing instrumentation."""
    from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import (
        IM_END_TOKEN,
        RAS_WIN_SIZE,
        _sample_logits,
    )

    if self.tokenizer is None:
        raise ValueError("Tokenizer not loaded.")

    from mlx_audio.tts.models.fish_qwen3_omni.prompt import Conversation, Message

    prompt_conversation = Conversation(list(conversation.messages))
    prompt_conversation.append(
        Message(
            role="assistant",
            parts=[],
            modality="voice",
            add_im_start=True,
            add_im_end=False,
        )
    )
    prompt = prompt_conversation.encode_for_inference(
        self.tokenizer, num_codebooks=self.model.num_codebooks
    )
    prompt = prompt[None, :, :]

    # --- Slow AR prefill ---
    timer = Timer()
    cache = self.model.make_cache()
    timer.start()
    result = self.model(prompt, cache=cache)
    logits = result.logits[:, -1]
    hidden_state = result.hidden_states[:, -1]
    t_prefill = timer.stop(logits, hidden_state)
    timing_data["slow_ar_prefill"].append(t_prefill)

    previous_semantic_tokens = []
    generated_steps = []
    im_end_id = self.tokenizer.get_token_id(IM_END_TOKEN)
    text_token_count = len(self.tokenizer.encode(batch_text))
    semantic_token_budget = min(
        max_new_tokens,
        max(32, text_token_count * 12),
    )

    for step in range(semantic_token_budget):
        # --- Semantic sampling ---
        timer.start()
        semantic_token = self._sample_semantic(
            logits=logits,
            previous_semantic_tokens=previous_semantic_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        mx.eval(semantic_token)
        t_sample = timer.stop()
        timing_data["semantic_sampling"].append(t_sample)

        semantic_token_id = int(semantic_token[0].item())
        if semantic_token_id == im_end_id:
            break

        previous_semantic_tokens.append(semantic_token_id)
        previous_semantic_tokens = previous_semantic_tokens[-RAS_WIN_SIZE:]

        semantic_code = (
            semantic_token - self.config.semantic_start_token_id
        ).astype(mx.int32)
        semantic_code = mx.clip(
            semantic_code, 0, self.config.audio_decoder_config.vocab_size - 1
        )
        previous_codebooks = semantic_code[:, None]

        # --- Fast AR total (prefill + 9 codebook steps) ---
        timer_fast_total = Timer()
        timer_fast_total.start()

        # Fast AR prefill
        timer.start()
        fast_cache = self.model.make_fast_cache()
        fast_prefill = self.model.fast_forward_cached(hidden_state, fast_cache)
        t_fp = timer.stop(fast_prefill)
        fast_ar_per_codebook["fast_prefill"].append(t_fp)

        # Fast AR embedding of semantic code
        fast_hidden = self.model.fast_embeddings(semantic_code)

        for cb_idx in range(self.model.num_codebooks - 1):
            timer.start()
            residual_logits = self.model.fast_forward_cached(
                fast_hidden, fast_cache
            )
            residual_token = _sample_logits(
                residual_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            t_cb = timer.stop(residual_token)
            fast_ar_per_codebook[f"codebook_{cb_idx + 1}"].append(t_cb)

            previous_codebooks = mx.concatenate(
                [previous_codebooks, residual_token[:, None]], axis=1
            )
            fast_hidden = self.model.fast_embeddings(residual_token)

        t_fast_total = timer_fast_total.stop()
        timing_data["fast_ar_total"].append(t_fast_total)

        generated_steps.append(previous_codebooks[0])

        # --- Slow AR decode step ---
        timer.start()
        next_input = mx.concatenate(
            [semantic_token[:, None].astype(mx.int32), previous_codebooks], axis=1
        )
        next_result = self.model(next_input[:, :, None], cache=cache)
        logits = next_result.logits[:, -1]
        hidden_state = next_result.hidden_states[:, -1]
        t_decode = timer.stop(logits, hidden_state)
        timing_data["slow_ar_decode"].append(t_decode)

    if not generated_steps:
        raise RuntimeError(f"No audio tokens generated for: {batch_text!r}")

    return mx.stack(generated_steps, axis=1).astype(mx.int32)


# Apply monkey-patch
from mlx_audio.tts.models.fish_qwen3_omni.fish_speech import Model
Model._generate_codes_for_batch = patched_generate_codes_for_batch


# ---------------------------------------------------------------------------
# Generate with timing
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("GENERATING: 'Hello world'")
print("=" * 70)

overall_start = time.perf_counter()

# Time code generation vs codec decode separately
gen_start = time.perf_counter()
for result in model.generate("Hello world", temperature=0.7, top_p=0.7, top_k=30):
    audio = result.audio
    sr = result.sample_rate
gen_elapsed = time.perf_counter() - gen_start

# The codec decode is already inside generate() after _generate_codes_for_batch.
# We can compute it as: gen_elapsed - sum(all component times)
total_components = (
    sum(timing_data["slow_ar_prefill"])
    + sum(timing_data["semantic_sampling"])
    + sum(timing_data["fast_ar_total"])
    + sum(timing_data["slow_ar_decode"])
)
# Codec decode time is the remainder (includes mx.eval(audio, codes) at end)
codec_time = gen_elapsed - total_components

overall_elapsed = time.perf_counter() - overall_start
audio_duration = float(audio.shape[0]) / float(sr)
n_steps = len(timing_data["slow_ar_decode"])

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("TIMING BREAKDOWN")
print("=" * 70)

print(f"\nAudio duration:  {audio_duration:.2f}s")
print(f"Total wall time: {gen_elapsed:.3f}s")
print(f"Real-time factor: {audio_duration / gen_elapsed:.2f}x")
print(f"Semantic tokens generated: {n_steps}")
print()

def report(name, times):
    total = sum(times)
    avg = total / len(times) if times else 0
    pct = total / gen_elapsed * 100
    print(f"  {name:30s}  total={total:7.3f}s  avg={avg*1000:7.2f}ms  "
          f"calls={len(times):4d}  share={pct:5.1f}%")

report("Slow AR prefill", timing_data["slow_ar_prefill"])
report("Slow AR decode", timing_data["slow_ar_decode"])
report("Semantic sampling", timing_data["semantic_sampling"])
report("Fast AR total", timing_data["fast_ar_total"])

print()
print("Fast AR breakdown (per codebook):")
for key in ["fast_prefill"] + [f"codebook_{i}" for i in range(1, 10)]:
    if key in fast_ar_per_codebook:
        times = fast_ar_per_codebook[key]
        total = sum(times)
        avg = total / len(times) if times else 0
        pct = total / gen_elapsed * 100
        print(f"    {key:25s}  total={total:7.3f}s  avg={avg*1000:7.2f}ms  "
              f"calls={len(times):4d}  share={pct:5.1f}%")

print()
report("Codec decode (estimated)", [codec_time])

print()
print("=" * 70)
print("PER-STEP AVERAGES (one semantic token)")
print("=" * 70)

if n_steps > 0:
    avg_slow_decode = sum(timing_data["slow_ar_decode"]) / n_steps * 1000
    avg_semantic = sum(timing_data["semantic_sampling"]) / n_steps * 1000
    avg_fast_total = sum(timing_data["fast_ar_total"]) / n_steps * 1000
    avg_step = (
        sum(timing_data["slow_ar_decode"])
        + sum(timing_data["semantic_sampling"])
        + sum(timing_data["fast_ar_total"])
    ) / n_steps * 1000

    print(f"  Slow AR decode:      {avg_slow_decode:7.2f} ms")
    print(f"  Semantic sampling:   {avg_semantic:7.2f} ms")
    print(f"  Fast AR total:       {avg_fast_total:7.2f} ms")
    print(f"  ---")
    print(f"  Total per step:      {avg_step:7.2f} ms")
    print(f"  Steps/sec:           {1000 / avg_step:7.1f}")
    print(f"  Theoretical RTF at 21.5ms/frame: {(21.5 / avg_step):.2f}x")
    print()
    print(f"  Fast AR share of per-step time: {avg_fast_total / avg_step * 100:.1f}%")
    print(f"  Slow AR share of per-step time: {avg_slow_decode / avg_step * 100:.1f}%")

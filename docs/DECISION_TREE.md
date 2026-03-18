# ANE TTS — Decision Tree

Every step has success/failure criteria and a pre-planned next action.
Execute top to bottom. At each decision point, take the branch that matches your result.

---

## Phase 0: Understand the Tools (2-3 hours)

### Step 0.1: Clone and build maderix/ANE
```bash
cd ~/Projects
git clone https://github.com/maderix/ANE.git maderix-ane
cd maderix-ane
# Follow their build instructions
```

**If build succeeds →** Go to Step 0.2
**If build fails →**
- Check macOS version (needs 15+)
- Check Xcode version
- Open an issue on their repo
- **Fallback:** Skip maderix, go to Step 0.3 (Espresso) or Step 0.5 (CoreML directly)

### Step 0.2: Run maderix/ANE benchmarks on M2 Max
```bash
# Run their included benchmark suite
# Record: throughput (TOPS utilized), max tensor dims, latency per operation
```

**Record these numbers:**
- [ ] Max matrix size that runs on ANE: _____ × _____
- [ ] Throughput at dim=768: _____ GOPS
- [ ] Throughput at dim=2048: _____ GOPS
- [ ] Throughput at dim=4096: _____ GOPS
- [ ] Supported data types: FP16 / INT8 / INT4 (circle which work)
- [ ] ANE SRAM apparent size: _____ (infer from when throughput drops)

**If dim=4096 works →** Fish S2 Pro's hidden_dim is compatible. Go to Phase 1.
**If dim=4096 fails or is very slow →**
- Fish S2 Pro won't run well on ANE directly
- **Pivot:** Use ANE only for the small draft model (0.6B, dim likely ≤2048). Go to Phase 2 directly.

### Step 0.3: (Backup) Clone and build Espresso
Only if maderix/ANE doesn't work.
```bash
cd ~/Projects
git clone https://github.com/christopherkarani/Espresso.git
cd Espresso
swift build
./espresso doctor  # Confirm ANE access
```

### Step 0.4: Profile Fish S2 Pro architecture
```python
# Add timing to mlx-audio's Fish model generate()
# Run: python3 profile_fish.py "Hello, this is a test."
```

**Record these numbers:**
- [ ] Time in text encoder: _____ ms
- [ ] Time per AR transformer step (slow AR, 4B): _____ ms
- [ ] Time per fast AR step (400M): _____ ms
- [ ] Time in codec decoder: _____ ms
- [ ] Audio tokens per second of output: _____
- [ ] Total tokens for 3s of audio: _____

**If AR transformer is >80% of time →** ANE acceleration of AR is the target
**If codec decoder is >30% of time →** ANE acceleration of codec is also worth trying (convolutions are ANE's strength)
**If fast AR is significant time →** The fast AR (400M) is a natural draft model candidate

### Step 0.5: CoreML baseline conversion (do this regardless)
```python
import coremltools as ct
# Convert Qwen3-TTS 0.6B to CoreML
# This is the "guaranteed to work" baseline
```

**Record:** CoreML conversion success, ANE delegation %, inference speed

---

## Phase 1: Fish S2 Pro on ANE — Direct Inference (the big swing)

### Step 1.1: Convert Fish S2 Pro AR transformer to maderix/ANE format
- Extract just the AR transformer weights (not the codec)
- Convert to whatever format maderix/ANE accepts
- Try running a single forward pass

**If it runs →** Benchmark it. Record tok/s. Go to Step 1.2.
**If conversion fails →**
- Which layers failed? Record them.
- **If attention fails:** ANE may not support Fish's attention pattern. Try replacing with ANE-compatible attention (see SqueezeBits blog for SDPA workarounds).
- **If FFN fails:** Check tensor dimensions against Phase 0 max dims.
- **If codec layers fail:** Expected. Keep codec on CPU/GPU, only AR on ANE. Go to Step 1.3.
- **If everything fails → Go to Phase 2** (speculative decode approach)

### Step 1.2: Benchmark Fish AR on ANE
```bash
# Run 100 tokens of generation, measure average ms/token
```

**Record:**
- [ ] ANE tok/s for Fish AR: _____
- [ ] GPU tok/s baseline: _____
- [ ] Speedup ratio: _____x

**If ANE is faster than GPU →** 🎉 This is the headline. Full Fish on ANE (AR on ANE, codec on CPU). Go to Phase 3.
**If ANE is similar to GPU (0.8x - 1.2x) →** Not worth it alone. But ANE + GPU parallel might help. Go to Phase 2.
**If ANE is slower than GPU →** ANE isn't the path for the big model. Go to Phase 2.

### Step 1.3: Hybrid — AR on ANE, codec on GPU
If the AR transformer works on ANE but the codec doesn't:
- Pipeline: Text → AR transformer on ANE → codec tokens → codec decoder on GPU → audio
- Measure end-to-end RTF

**If RTF > 1.0 →** Ship it. This is the result.
**If RTF < 1.0 →** Go to Phase 2.

---

## Phase 2: Speculative TTS Decoding (the reliable path)

### Step 2.1: Run Qwen3-TTS 0.6B on ANE
Convert the small model to run on ANE (maderix or CoreML).

**Record:**
- [ ] ANE tok/s for Qwen 0.6B: _____
- [ ] GPU tok/s baseline: _____

**If ANE runs Qwen 0.6B fast (>100 tok/s) →** Perfect draft model candidate. Go to Step 2.2.
**If ANE is slow for Qwen 0.6B →**
- Try CoreML instead of maderix
- If still slow → **Pivot to Phase 4** (pure GPU optimization, no ANE)

### Step 2.2: Wire speculative decode loop
```
Loop:
  1. Qwen 0.6B on ANE generates N draft audio tokens (fast)
  2. Fish S2 Pro on GPU verifies all N tokens in one forward pass
  3. Accept matching tokens, reject mismatches
  4. Repeat from last accepted token
```

**Record:**
- [ ] Draft generation time for N=5 tokens: _____ ms
- [ ] Verification time (1 GPU forward pass): _____ ms
- [ ] Acceptance rate: _____%
- [ ] Effective tok/s: _____
- [ ] End-to-end RTF: _____

**If acceptance rate > 60% and RTF > 1.0 →** 🎉 Ship it. Go to Phase 3.
**If acceptance rate < 40% →**
- Models are too different. Tokens don't match.
- **Try:** Acoustic similarity matching (Apple's Coarse-Grained technique). Go to Step 2.3.
- **Or:** Train a distilled draft model from Fish. Go to Step 2.4.

### Step 2.3: Acoustic similarity matching
- Extract Fish S2 Pro's codec embeddings
- Cluster acoustically similar tokens into groups
- Modify verification to accept group-level matches instead of exact

**If acceptance rate improves to > 60% →** Ship it.
**If still low →** The architectures are too different. Go to Step 2.4.

### Step 2.4: Distill a draft model from Fish
- Train a tiny model (200-500M params) to predict Fish's audio token distribution
- Run this on ANE as the draft
- This requires training data and compute but gives the highest acceptance rate (70-85%)

**This is a bigger project.** Estimate: 1-2 weeks. Only do this if Steps 2.2 and 2.3 fail.

---

## Phase 3: Polish and Ship

### Step 3.1: Package the tool
```
ane-tts/
├── ane_tts/
│   ├── __init__.py
│   ├── backends/
│   │   ├── ane_direct.py    # maderix/ANE backend
│   │   └── coreml.py        # CoreML backend
│   ├── models/
│   │   ├── fish_s2_pro.py   # Fish model adapter
│   │   └── qwen_tts.py      # Qwen model adapter
│   ├── speculative.py        # Spec decode loop (if needed)
│   └── server.py             # FastAPI server (drop-in replacement for tts_bridge.py)
├── benchmarks/
│   ├── benchmark.py
│   └── results/
├── docs/
├── tests/
└── README.md
```

### Step 3.2: Write benchmarks
- GPU baseline vs ANE (direct and CoreML)
- Multiple models (Fish 5B, Qwen 1.7B, Qwen 0.6B)
- Multiple text lengths
- RTF chart (the money shot for the README)

### Step 3.3: CoreML backend (the "for everyone" version)
If you haven't done this yet, now convert everything to CoreML.
This is the stable version anyone can use.

### Step 3.4: Release
1. Push to GitHub
2. Write a blog post with benchmarks and the story
3. Post on r/LocalLLaMA, r/MachineLearning, Hacker News
4. Submit to HuggingFace (model cards, spaces demo)

---

## Phase 4: Fallback — Pure GPU Optimization (no ANE)

If ANE doesn't work for TTS at all:

### Option 4A: Fix Fish S2 Pro quantization
- Pad (N,8) embeddings to (N,32)
- Quantize to 4-bit
- Slice back at inference
- Expected: 2-3x speedup (0.65x → 1.3-2.0x RTF)

### Option 4B: Multi-token prediction (Nguyen et al.)
- Add prediction heads to Fish S2 Pro
- Generate 4-5 tokens per forward pass
- No ANE needed, pure GPU technique
- Expected: 4-5x speedup (if trained)

### Option 4C: Use Qwen3-TTS and accept the quality tradeoff
- Already running at 1.64x RTF
- Good enough for most use cases
- Focus energy elsewhere

---

## Quick Reference: Expected Outcomes

| Scenario | RTF | What we'd ship | Community impact |
|----------|-----|---------------|-----------------|
| Fish on ANE direct, works fast | >1.5x | `ane-tts --backend ane-direct` | 🔥 Massive. HN front page. |
| Fish hybrid (AR on ANE, codec on GPU) | >1.0x | `ane-tts --backend hybrid` | 🔥 Strong. Novel architecture. |
| Spec decode (Qwen draft ANE + Fish verify GPU) | >1.0x | `ane-tts --backend speculative` | 🔥 Strong. First heterogeneous TTS spec decode. |
| CoreML only speedup | 0.9-1.1x | `ane-tts --backend coreml` | 👍 Useful but not exciting. |
| Nothing works for Fish, Qwen on ANE fast | >3x for 0.6B | `ane-tts` (Qwen only) | 👍 Modest. Small model on ANE isn't novel. |
| Total failure | — | Blog post: "What ANE can't do for TTS" | 📝 Still valuable data. Nobody's published this. |

---

*Every outcome produces something publishable or shippable. There is no wasted path.*

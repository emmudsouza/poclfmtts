# PocketTTS × LFM2 backbone swap — feasibility analysis

**Question:** Will replacing PocketTTS's Transformer backbone with the LFM2 hybrid
(conv+attention) architecture improve inference speed / RTF on a memory-bandwidth-constrained
edge CPU (Raspberry Pi 5 class, ~17 GB/s)?

**Short answer:** Not from the swap itself at equal model size. On a bandwidth-bound device,
per-step speed is set by **bytes read per step**, which is dominated by **parameter count ×
precision** — the same whether a layer is attention or convolution. LFM2's structural win
(no growing KV cache) only removes ~2–6% of per-step traffic at TTS sequence lengths. The
**only** path to a real LFM2 speed win on the Pi is reaching the quality bar with **fewer
parameters**; that is an empirical training result, not an architectural guarantee. The
guaranteed, architecture-independent lever is **quantization** (int8 ≈ 2×, int4 ≈ 4×), which
PocketTTS does not yet support.

---

## 1. How PocketTTS 100M is actually built

Verified from Kyutai's `pocket-tts` repo configs and the CALM paper (arXiv 2509.06926).

- **Backbone = 6-layer, d_model=1024, 16-head, RoPE, *full causal attention* transformer
  (~75M params).** This is the dominant cost. CALM states generation is *"bottlenecked by the
  cost of generating the conditioning variable via the large causal transformer."*
- It is **not** the older RQ/depth-transformer + 16-codebook Mimi design. PocketTTS emits a
  **continuous 32-dim VAE latent at 12.5 Hz**; the per-step head is a **1-step MLP
  (`SimpleMLPAdaLN`, ~3% of runtime)**; the Mimi VAE decode is comparatively cheap.
- → The backbone is correctly identified as the thing to attack. There is no depth transformer
  to optimize away.
- ~6× real-time on an M4 (2 CPU cores). **INT8 not yet supported** (headroom). Code is MIT;
  weights are open; **training code is not public** (repo is inference-only).

Key files: `pocket_tts/models/flow_lm.py`, `pocket_tts/modules/transformer.py` /
`mimi_transformer.py` (the `StreamingTransformer` + `_LinearKVCacheBackend` that would be
replaced), `pocket_tts/models/mimi.py`, `pocket_tts/config/english.yaml`.

## 2. What LFM2 is

Verified from Liquid AI configs and the LFM2 technical report (arXiv 2511.23404).

- 16 layers = **10 gated short-convolution (kernel-3) blocks + 6 grouped-query-attention
  blocks**. Conv blocks keep a **constant-size state** (no KV cache that grows with sequence
  length) — its headline edge-CPU advantage. GQA uses 8 KV heads.
- A conv mixing block (in-proj `d→3d` + out-proj `d→d` ≈ 4·d²) has roughly the **same weight
  count** as an attention block's Q/K/V/O (≈ 4·d²), and both carry the same SwiGLU FFN (≈ 8·d²).
- **Smallest official checkpoint is 350M** (3.5× the 75M backbone). No sub-100M model exists,
  so a small TTS backbone must be a **custom config trained from scratch**.
- Precedent: **LFM2-Audio-1.5B** already uses an LFM2 backbone for audio/TTS.

## 3. The roofline (why the answer is what it is)

Batch-1 autoregressive decode on a bandwidth-bound CPU:

```
time/step ≈ (weight_bytes + KV_cache_bytes) / effective_bandwidth
steps/s   ≈ effective_bandwidth / bytes_read_per_step
```

**Weight traffic dominates** and is architecture-independent at equal params/precision:

| Backbone size | Precision | Model bytes | steps/s @17 GB/s | RTF ceiling ÷12.5 Hz |
|---|---|---|---|---|
| 75M | fp16 | ~150 MB | ~113 | ~9× (≈5× at real ~10 GB/s) |
| 75M | int8 | ~75 MB  | ~227 | ~18× (≈10× at real ~10 GB/s) |

**KV-cache traffic is small at TTS sequence lengths** (6 layers, d=1024, full MHA):

| seqlen | audio @12.5 Hz | KV bytes/step (fp16) | KV ÷ weight traffic |
|---|---|---|---|
| 100  | 8 s  | 2.5 MB | ~1.7% |
| 375  | 30 s | 9.2 MB | ~6.1% |
| 1000 | 80 s | 24.6 MB | ~16% |

KV traffic does not equal weight traffic until **~6,000 frames (~8 min of continuous audio)** —
far beyond chunked TTS. So at realistic chunk lengths the backbone is firmly **weight-bound**.

Confirmed pattern from llama.cpp / ARM benchmarks: small-model batch-1 decode is
memory-bandwidth-bound, and tokens/sec scales ~inversely with quantized model bytes at a
near-constant effective bandwidth (~9–13 GB/s measured on a Pi 5).

## 4. Verdict

| Lever | Effect on Pi RTF | Architecture-dependent? |
|---|---|---|
| Swap attention → LFM2 **at equal params**, short sequences | ~2–6% (only the KV slice) | n/a — same weight bytes |
| LFM2 reaching quality at **fewer params** | proportional to param reduction | **yes — the real hypothesis** |
| Long, unchunked streams (minutes) | large (KV grows) | yes — but not typical TTS |
| **int8 quantization** | ~2× | no (guaranteed) |
| **int4 quantization** | ~4× | no (guaranteed) |
| Fewer layers / smaller d_model | proportional | no |

**LFM2 is worth adopting on speed grounds iff a smaller-than-75M LFM2 config matches the
attention baseline's expressive quality** (Liquid claims better quality-per-param; unproven for
small TTS). Otherwise the swap yields only the ~2–6% KV saving and is justified, if at all, by
streaming/long-context or peak-memory reasons rather than RTF.

## 5. Recommended path (validate before rewriting)

1. **Phase 0 — baseline + instrument.** Reproduce PocketTTS inference; measure RTF on the
   target ARM board; profile the per-step split (backbone vs MLP head vs Mimi decode; within
   the backbone, attention vs FFN) to confirm weight-bound behavior on real hardware.
2. **Phase 1 — quantization (guaranteed win).** Quantize the existing backbone to int8/int4;
   measure RTF + quality. Build the quality harness here (WER, speaker-sim, UTMOS/DNSMOS +
   small expressive listening set).
3. **Phase 2 — test the hypothesis.** Wrap an LFM2 backbone behind the existing streaming
   interface (HF `Lfm2Model` → 1024-dim embedding/step → unchanged flow head + Mimi VAE).
   Train 2–3 custom sub-100M LFM2 configs (~35M/50M/75M) on expressive data (distill from the
   released PocketTTS as teacher — lower risk, mirrors how Kyutai distilled 24→6 layers).
   Plot quality-per-param vs the 75M baseline, all at int8.

**Decision gate:** adopt LFM2 only if a <75M config meets baseline expressive quality.

## Open items

- Exact ARM board + its memory bandwidth (anchors the roofline).
- Source / licensing of the expressive training corpus.

## Sources

- PocketTTS: https://github.com/kyutai-labs/pocket-tts · https://huggingface.co/kyutai/pocket-tts ·
  CALM paper https://arxiv.org/abs/2509.06926 · https://kyutai.org/blog/2026-01-13-pocket-tts
- LFM2: https://www.liquid.ai/blog/liquid-foundation-models-v2-our-second-series-of-generative-ai-models ·
  report https://arxiv.org/abs/2511.23404 · https://huggingface.co/LiquidAI/LFM2-350M ·
  LFM2-Audio https://huggingface.co/LiquidAI/LFM2-Audio-1.5B
- Roofline / Pi5: https://www.stratosphereips.org/blog/2025/6/5/how-well-do-llms-perform-on-a-raspberry-pi-5 ·
  https://github.com/ggml-org/llama.cpp/discussions/4167 · https://www.raspberrypi.com/news/benchmarking-raspberry-pi-5/

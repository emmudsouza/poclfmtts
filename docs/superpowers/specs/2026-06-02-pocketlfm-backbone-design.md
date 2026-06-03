# PocketLFM — PocketTTS with an LFM2 backbone (design)

Date: 2026-06-02

## Purpose

Build `pocketlfm.py`: a single-file, OOP reimplementation of Kyutai PocketTTS that
**replaces the Transformer backbone with LFM2-style layers** (gated short convolutions +
grouped-query attention), while **keeping the total model size ~100M** and reusing the
released PocketTTS weights for everything except the backbone.

This is the implementation vehicle for the feasibility hypothesis in
`docs/lfm2-backbone-feasibility.md`: on a memory-bandwidth-bound edge CPU, an LFM2 backbone
only wins if it matches quality at fewer params; the swap itself is RTF-neutral at equal
size. `pocketlfm.py` lets us assemble the model, confirm ~100M params, run it end-to-end,
and benchmark RTF — then later distill/train the backbone on expressive data.

## Decisions (locked)

- **Reuse upstream codec/head, new LFM backbone.** Import `pocket-tts`; reimplement only the
  backbone in our file.
- **Wrap upstream Mimi** for real audio out (no codec reimplementation).
- **Authentic LFM2 blocks**: RMSNorm + SwiGLU FFN + double-gated LIV short conv.
- **6 layers, 4 conv + 2 GQA** (attention at layer indices `[2, 5]`); d_model=1024 fixed.
- **`pocket-tts` as a real dependency** (added to `pyproject.toml` via `uv`).

## Integration strategy: "assemble & swap"

The upstream `StreamingTransformer` backbone exposes a minimal contract:
`forward(x: [B, S, 1024], model_state: dict | None) -> [B, S, 1024]`, called incrementally
during streaming. Per-layer streaming state is auto-collected from any `StatefulModule`
submodule (`init_states` walks `named_modules()`; `tts_model.py:228` assigns
`_module_absolute_name`).

`PocketLFM.from_pretrained(language="english")`:
1. Build the upstream `TTSModel` via its normal loader → pretrained Mimi VAE,
   `SimpleMLPAdaLN` flow head, `LUTConditioner`, `input_linear`, `out_norm`, `out_eos`
   (all `strict=True` weight load succeeds).
2. Replace `tts_model.flow_lm.transformer` with our `LFMBackbone(LFMConfig)` (random init).
3. Re-run the `_module_absolute_name` assignment loop over `flow_lm` so the new LFM
   stateful submodules are keyed for streaming.
4. Return a `PocketLFM` wrapping `tts_model`.

Generation, the streaming loop, EOS handling, and audio decode are **reused unchanged**.
Result: the model is RTF-benchmarkable immediately; audio is noise until the backbone is
trained/distilled (separate phase, out of scope here).

## Components (all in `pocketlfm.py`, OOP)

- `LFMConfig` (dataclass): `d_model=1024`, `n_layers=6`, `attn_layer_indices=(2,5)`,
  `n_heads=16`, `n_kv_heads=8`, `conv_kernel=3`, `ff_dim` (tuned to size), `max_period=10000`.
- `RMSNorm(nn.Module)`.
- `SwiGLUFFN(nn.Module)`: `w1: d→ff`, `w3: d→ff`, `SiLU(w1·x) ⊙ (w3·x)`, `w2: ff→d`.
- `LFMShortConv(nn.Module)`: `in_proj Linear(d→3d)` → split `(B, C, x)` →
  `x = B ⊙ x` → depthwise causal conv via **reused upstream `StreamingConv1d`**
  (`groups=d, kernel=3, bias=False`) → `x = C ⊙ x` → `out_proj Linear(d→d)`.
  Constant streaming state (k−1=2 frames); no growing KV cache.
- `LFMAttention(StatefulModule)`: GQA (16 q-heads / 8 kv-heads), RoPE, streaming KV cache.
  Reuse `RotaryEmbedding`, `complete_kv`, `_build_attention_mask` from upstream;
  `repeat_kv` for SDPA. `init_state`/`increment_step` mirror `_LinearKVCacheBackend` but
  sized for `n_kv_heads`.
- `LFMLayer(nn.Module)`: pre-norm residual mixer (conv or attn by index) + pre-norm
  residual SwiGLU FFN.
- `LFMBackbone(nn.Module)`: `ModuleList[LFMLayer]`; `forward(x, model_state)` matching
  `StreamingTransformer`.
- `PocketLFM`: assembly wrapper. `from_pretrained(...)`, `generate(text, ...)` (delegates
  upstream), `num_parameters()`, `save_backbone()/load_backbone()` for the new weights.

## Size budget

d_model=1024 fixed (flow head + `input_linear`/`out_norm` expect it). Per-layer ≈ `12d²`
(mixer ≈ `4d²`, SwiGLU FFN ≈ `3·d·ff`). Targets:

| Part | Approx params |
|---|---|
| Conv layer (×4) | ~12.5–13.5M each |
| GQA layer (×2)  | ~11.5–12.5M each |
| **LFM backbone total** | **~75–80M** |
| Reused: text LUT + flow head + Mimi VAE | ~22M |
| **Total** | **~95–105M** |

`ff_dim` (and, if needed, layer count) tuned so a build-time assertion `90e6 < total < 110e6`
holds. The original backbone it replaces is also ~75M, so this is size-matched.

## Dependencies

Add to `pyproject.toml` via `uv add`: `pocket-tts` (PyPI v2.1.0; fall back to git/local path
if unavailable) and its transitive `torch` (CPU index), `safetensors`, `sentencepiece`, etc.

## Verification

1. **Param count**: `assert 90e6 < PocketLFM.from_pretrained().num_parameters() < 110e6`.
2. **Shape**: random `[B, S, 1024]` through `LFMBackbone` → `[B, S, 1024]`.
3. **Streaming equivalence**: full-sequence forward vs. step-by-step (S=1) streaming produce
   matching outputs (within fp tolerance) for both conv and GQA paths — proves the cache
   logic.
4. **End-to-end**: `PocketLFM.from_pretrained().generate("hello world")` runs and emits a
   24 kHz waveform (noise pre-training) without errors.
5. **RTF**: time `generate(...)` over a fixed sentence; report RTF = audio_s / wall_s on the
   dev machine (Pi/ARM later per the feasibility plan).

## Out of scope (later phases)

- Distillation/training of the backbone on expressive data.
- int8/int4 quantization.
- Pi5/ARM benchmarking (covered by the feasibility plan, Phase 0/1).

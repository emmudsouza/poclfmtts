from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PocketLFMConfig:
    d_model: int = 1024
    n_layers: int = 6
    attn_layer_indices: tuple[int, ...] = (2, 5)
    n_heads: int = 16
    n_kv_heads: int = 8
    conv_kernel: int = 3
    ff_dim: int = 2816
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    latent_dim: int = 32
    text_vocab: int = 4000
    text_dim: int = 512
    text_heads: int = 8
    text_encoder_layers: int = 2
    text_ff_dim: int = 1536
    max_text_len: int = 512

    mtp_horizon: int = 4
    head_dim: int = 192
    head_depth: int = 2

    attn_window: int | None = 384
    attn_sink: int = 128
    eos_loss_weight: float = 1.0
    input_noise: float = 0.0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert self.mtp_horizon >= 1

    @property
    def head_dim_per_head(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    dh = x.shape[-1]
    half = dh // 2
    inv_freq = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class ShortConvMixer(nn.Module):
    def __init__(self, cfg: PocketLFMConfig):
        super().__init__()
        d, k = cfg.d_model, cfg.conv_kernel
        self.k = k
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(d, d, k, groups=d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)

    def forward(self, x, conv_state):
        b, t, d = x.shape
        gate_b, gate_c, val = self.in_proj(x).chunk(3, dim=-1)
        y = (gate_b * val).transpose(1, 2)
        pad = conv_state if conv_state is not None else y.new_zeros(b, d, self.k - 1)
        y_in = torch.cat([pad, y], dim=-1)
        z = self.conv(y_in).transpose(1, 2)
        new_state = y_in[..., -(self.k - 1):] if self.k > 1 else y.new_zeros(b, d, 0)
        return self.out_proj(gate_c * z), new_state


class WindowedSelfAttention(nn.Module):
    def __init__(self, cfg: PocketLFMConfig):
        super().__init__()
        self.nh = cfg.n_heads
        self.nkv = cfg.n_kv_heads
        self.dh = cfg.head_dim_per_head
        self.theta = cfg.rope_theta
        self.window = cfg.attn_window
        self.sink = cfg.attn_sink if cfg.attn_window is not None else 0
        self.capacity = None if self.window is None else self.sink + self.window
        d = cfg.d_model
        self.q_proj = nn.Linear(d, self.nh * self.dh, bias=False)
        self.k_proj = nn.Linear(d, self.nkv * self.dh, bias=False)
        self.v_proj = nn.Linear(d, self.nkv * self.dh, bias=False)
        self.o_proj = nn.Linear(self.nh * self.dh, d, bias=False)

    def _mask(self, q_pos, k_pos):
        causal = k_pos[None, :] <= q_pos[:, None]
        if self.window is None:
            return causal
        within = (q_pos[:, None] - k_pos[None, :]) < self.window
        is_sink = k_pos[None, :] < self.sink
        return causal & (within | is_sink)

    def _evict(self, k, v, pos):
        if self.window is None or k.shape[2] <= self.capacity:
            return k, v, pos
        keep = self.capacity - self.sink
        s = self.sink
        k = torch.cat([k[:, :, :s], k[:, :, -keep:]], dim=2)
        v = torch.cat([v[:, :, :s], v[:, :, -keep:]], dim=2)
        pos = torch.cat([pos[:s], pos[-keep:]])
        return k, v, pos

    def forward(self, x, kv_state, positions, streaming):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.nh, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.nkv, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.nkv, self.dh).transpose(1, 2)
        q = apply_rope(q, positions, self.theta)
        k = apply_rope(k, positions, self.theta)

        if not streaming:
            mask = self._mask(positions, positions)[None, None]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            return self.o_proj(out.transpose(1, 2).reshape(b, t, self.nh * self.dh)), None

        if kv_state is not None:
            k = torch.cat([kv_state["k"], k], dim=2)
            v = torch.cat([kv_state["v"], v], dim=2)
            pos_all = torch.cat([kv_state["pos"], positions])
        else:
            pos_all = positions
        mask = self._mask(positions, pos_all)[None, None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        out = self.o_proj(out.transpose(1, 2).reshape(b, t, self.nh * self.dh))
        k_s, v_s, pos_s = self._evict(k, v, pos_all)
        return out, {"k": k_s, "v": v_s, "pos": pos_s}


class BackboneLayer(nn.Module):
    def __init__(self, cfg: PocketLFMConfig, is_attn: bool):
        super().__init__()
        self.is_attn = is_attn
        self.norm_mix = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = WindowedSelfAttention(cfg) if is_attn else ShortConvMixer(cfg)
        self.norm_ff = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ff = SwiGLU(cfg.d_model, cfg.ff_dim)

    def forward(self, x, layer_state, positions, streaming):
        mix_state = layer_state["mix"] if layer_state is not None else None
        h = self.norm_mix(x)
        if self.is_attn:
            update, new_mix = self.mixer(h, mix_state, positions, streaming)
        else:
            update, new_mix = self.mixer(h, mix_state)
        x = x + update
        x = x + self.ff(self.norm_ff(x))
        if streaming:
            return x, {"mix": new_mix}
        return x, None


class LFMBackbone(nn.Module):
    def __init__(self, cfg: PocketLFMConfig):
        super().__init__()
        self.cfg = cfg
        attn = set(cfg.attn_layer_indices)
        self.layers = nn.ModuleList(
            [BackboneLayer(cfg, is_attn=(i in attn)) for i in range(cfg.n_layers)]
        )

    def init_state(self) -> dict:
        return {"offset": 0, "layers": [None] * len(self.layers)}

    def forward(self, x, state=None):
        t = x.shape[1]
        offset = 0 if state is None else state["offset"]
        positions = torch.arange(offset, offset + t, device=x.device)
        streaming = state is not None
        for i, layer in enumerate(self.layers):
            ls = None if state is None else state["layers"][i]
            x, new_state = layer(x, ls, positions, streaming)
            if state is not None:
                state["layers"][i] = new_state
        if state is not None:
            state["offset"] = offset + t
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, causal: bool, eps: float = 1e-5):
        super().__init__()
        self.causal = causal
        self.nh = n_heads
        self.dh = dim // n_heads
        self.norm1 = RMSNorm(dim, eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim, eps)
        self.ff = SwiGLU(dim, ff_dim)

    def forward(self, x, key_mask=None):
        b, t, _ = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).view(b, t, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        mask = None
        if key_mask is not None:
            mask = key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=self.causal)
        x = x + self.proj(out.transpose(1, 2).reshape(b, t, self.nh * self.dh))
        return x + self.ff(self.norm2(x))


class TextEncoder(nn.Module):
    def __init__(self, cfg: PocketLFMConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.text_vocab, cfg.text_dim)
        self.pos = nn.Parameter(torch.randn(cfg.max_text_len, cfg.text_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.text_dim, cfg.text_heads, cfg.text_ff_dim, causal=False,
                              eps=cfg.norm_eps)
             for _ in range(cfg.text_encoder_layers)]
        )
        self.proj = nn.Linear(cfg.text_dim, cfg.d_model, bias=False)
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, tokens, mask):
        x = self.embed(tokens) + self.pos[: tokens.shape[1]][None]
        for block in self.blocks:
            x = block(x, key_mask=mask)
        return self.norm(self.proj(x))


class TimestepEmbed(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)

    def forward(self, t):
        half = self.hidden // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        ang = t.float() * freqs[None, :]
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.lin2(F.silu(self.lin1(emb)))


class AdaLNBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.ada = nn.Linear(hidden, 3 * hidden)

    def forward(self, x, c):
        scale, shift, gate = self.ada(F.silu(c)).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class FlowHead(nn.Module):
    def __init__(self, cfg: PocketLFMConfig, cond_dim: int):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        hidden = cfg.head_dim
        self.in_proj = nn.Linear(cfg.latent_dim, hidden)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        self.t_embed = TimestepEmbed(hidden)
        self.blocks = nn.ModuleList([AdaLNBlock(hidden) for _ in range(cfg.head_depth)])
        self.out = nn.Linear(hidden, cfg.latent_dim)

    def velocity(self, cond, t, x):
        c = self.cond_proj(cond) + self.t_embed(t)
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, c)
        return self.out(h)

    @torch.no_grad()
    def sample(self, cond, steps, temp):
        b = cond.shape[0]
        x = torch.randn(b, self.latent_dim, device=cond.device) * math.sqrt(temp)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((b, 1), i * dt, device=cond.device)
            x = x + self.velocity(cond, t, x) * dt
        return x

    def loss(self, cond, target):
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], 1, device=target.device)
        x_t = (1 - t) * noise + t * target
        return F.mse_loss(self.velocity(cond, t, x_t), target - noise)


class MTPHeads(nn.Module):
    def __init__(self, cfg: PocketLFMConfig):
        super().__init__()
        self.horizon = cfg.mtp_horizon
        self.latent_dim = cfg.latent_dim
        self.head_embed = nn.Parameter(torch.randn(cfg.mtp_horizon, cfg.d_model) * 0.02)
        self.head = FlowHead(cfg, cfg.d_model)

    def loss(self, hidden, targets, mask):
        total = hidden.new_zeros(())
        count = 0
        for k in range(self.horizon):
            m = mask[:, k]
            if m.any():
                total = total + self.head.loss(hidden[m] + self.head_embed[k], targets[m, k])
                count += 1
        return total / max(1, count)

    def sample(self, hidden, steps, temp):
        return torch.stack(
            [self.head.sample(hidden + self.head_embed[k], steps, temp) for k in range(self.horizon)],
            dim=1,
        )


class PocketLFM(nn.Module):
    def __init__(self, cfg: PocketLFMConfig):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TextEncoder(cfg)
        self.latent_in = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.bos = nn.Parameter(torch.randn(cfg.latent_dim))
        self.backbone = LFMBackbone(cfg)
        self.out_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mtp = MTPHeads(cfg)
        self.eos_head = nn.Linear(cfg.d_model, 1)
        self.register_buffer("latent_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("latent_std", torch.ones(cfg.latent_dim))

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_std.copy_(std.clamp(min=1e-4).to(self.latent_std))

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latent_mean) / self.latent_std

    def denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latent_std + self.latent_mean

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters())

        total = self.num_parameters()
        parts = {
            "text_encoder": count(self.text_encoder),
            "backbone": count(self.backbone),
            "mtp": count(self.mtp),
        }
        parts["other"] = total - sum(parts.values())
        parts["total"] = total
        return parts

    def _text_prefix(self, text_tokens, text_lens):
        device = text_tokens.device
        mask = torch.arange(text_tokens.shape[1], device=device)[None, :] < text_lens.to(device)[:, None]
        return self.text_encoder(text_tokens, mask)

    def compute_losses(self, text_tokens, text_lens, latents, lat_lens) -> dict[str, torch.Tensor]:
        b, t_max, _ = latents.shape
        k = self.cfg.mtp_horizon
        d = self.cfg.d_model
        device = latents.device
        text_lens = text_lens.to(device)
        lat_lens = lat_lens.to(device)
        latents = self.normalize(latents)

        text_emb = self._text_prefix(text_tokens, text_lens)

        ctx = latents
        if self.training and self.cfg.input_noise > 0:
            ctx = latents + self.cfg.input_noise * torch.randn_like(latents)

        seqs = []
        for i in range(b):
            lt, t = int(text_lens[i]), int(lat_lens[i])
            bos = self.bos.unsqueeze(0)
            audio_in = torch.cat([bos, ctx[i, : t - 1]], dim=0)
            seqs.append(torch.cat([text_emb[i, :lt], self.latent_in(audio_in)], dim=0))
        packed = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        hidden = self.out_norm(self.backbone(packed))

        audio_h = latents.new_zeros(b, t_max, d)
        for i in range(b):
            lt, t = int(text_lens[i]), int(lat_lens[i])
            audio_h[i, :t] = hidden[i, lt: lt + t]

        lat_mask = torch.arange(t_max, device=device)[None, :] < lat_lens[:, None]
        targets = latents.new_zeros(b, t_max, k, self.cfg.latent_dim)
        mask = torch.zeros(b, t_max, k, dtype=torch.bool, device=device)
        for j in range(k):
            if t_max - j > 0:
                targets[:, : t_max - j, j] = latents[:, j:]
                mask[:, : t_max - j, j] = lat_mask[:, j:] & lat_mask[:, : t_max - j]

        mtp = self.mtp.loss(
            audio_h.reshape(b * t_max, -1),
            targets.reshape(b * t_max, k, self.cfg.latent_dim),
            mask.reshape(b * t_max, k),
        )

        eos_logits = self.eos_head(audio_h).squeeze(-1)
        eos_target = torch.zeros(b, t_max, device=device)
        eos_target[torch.arange(b, device=device), (lat_lens - 1).clamp(min=0)] = 1.0
        eos = F.binary_cross_entropy_with_logits(eos_logits[lat_mask], eos_target[lat_mask])

        total = mtp + self.cfg.eos_loss_weight * eos
        return {"mtp": mtp, "eos": eos, "total": total}

    @torch.no_grad()
    def stream_latents(self, text_tokens, max_frames, flow_steps=1, temp=1.0, eos_threshold=0.0):
        b = text_tokens.shape[0]
        device = text_tokens.device
        text_lens = torch.full((b,), text_tokens.shape[1], device=device)
        text_emb = self._text_prefix(text_tokens, text_lens)

        state = self.backbone.init_state()
        self.backbone(text_emb, state)
        bos = self.bos.view(1, 1, -1).expand(b, 1, -1)
        hidden = self.out_norm(self.backbone(self.latent_in(bos), state))

        produced = 0
        while produced < max_frames:
            block = self.mtp.sample(hidden[:, -1], flow_steps, temp)
            for k in range(self.cfg.mtp_horizon):
                yield self.denormalize(block[:, k])
                produced += 1
                if produced >= max_frames:
                    return
            hidden = self.out_norm(self.backbone(self.latent_in(block), state))
            if (self.eos_head(hidden[:, -1:]) > eos_threshold).all():
                return

    @torch.no_grad()
    def generate_latents(self, text_tokens, max_frames, flow_steps=1, temp=1.0, eos_threshold=0.0):
        frames = list(self.stream_latents(text_tokens, max_frames, flow_steps, temp, eos_threshold))
        return torch.stack(frames, dim=1)[:, :max_frames]


def load_pretrained_codec(language: str = "english"):
    try:
        from pocket_tts.models.tts_model import TTSModel
    except ImportError as exc:
        raise ImportError(
            "load_pretrained_codec requires the 'pocket-tts' package (e.g. `uv add pocket-tts`)."
        ) from exc
    tts = TTSModel.load_model(language=language)
    tts.mimi.eval()
    return tts.mimi, tts.flow_lm.conditioner.tokenizer, int(tts.mimi.sample_rate), float(
        tts.config.mimi.frame_rate
    )


class PocketLFMPipeline:
    def __init__(self, model: PocketLFM, mimi, tokenizer, frame_rate: float = 12.5):
        self.model = model
        self.mimi = mimi
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate

    @classmethod
    def from_pretrained(cls, language: str = "english", cfg: PocketLFMConfig | None = None):
        mimi, tokenizer, _sr, frame_rate = load_pretrained_codec(language)
        return cls(PocketLFM(cfg or PocketLFMConfig()), mimi, tokenizer, frame_rate)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, language: str = "english", device: str = "cpu"):
        ckpt = torch.load(ckpt_path, map_location=device)
        model = PocketLFM(PocketLFMConfig(**ckpt["cfg"]))
        model.load_state_dict(ckpt["model"])
        model.eval()
        mimi, tokenizer, _sr, frame_rate = load_pretrained_codec(language)
        return cls(model, mimi, tokenizer, frame_rate).to(device)

    @property
    def sample_rate(self) -> int:
        return int(self.mimi.sample_rate)

    def to(self, device: str):
        self.model.to(device)
        self.mimi.to(device)
        return self

    def quantize(self):
        self.model = torch.ao.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        return self

    def _tokens(self, text):
        device = next(self.model.parameters()).device
        return torch.tensor([self.tokenizer.sp.encode(text, out_type=int)], dtype=torch.long,
                            device=device)

    def _decode(self, latent_frames, mimi_state):
        chan = torch.stack(latent_frames, dim=1).transpose(1, 2)
        if chan.shape[1] == self.mimi.quantizer.dimension:
            chan = self.mimi.quantizer(chan)
        return self.mimi.decode_from_latent(chan, mimi_state)

    @torch.no_grad()
    def generate(self, text: str, max_seconds: float = 10.0, flow_steps: int = 1,
                temp: float = 1.0, eos_threshold: float = 0.0):
        from pocket_tts.modules.stateful_module import init_states

        tokens = self._tokens(text)
        max_frames = int(max_seconds * self.frame_rate)
        latents = self.model.generate_latents(tokens, max_frames, flow_steps, temp, eos_threshold)
        latent_chan = latents.transpose(1, 2)
        if latent_chan.shape[1] == self.mimi.quantizer.dimension:
            latent_chan = self.mimi.quantizer(latent_chan)
        mimi_state = init_states(self.mimi, batch_size=latents.shape[0], sequence_length=10_000)
        return self.mimi.decode_from_latent(latent_chan, mimi_state)

    @torch.no_grad()
    def generate_stream(self, text: str, max_seconds: float = 10.0, flow_steps: int = 1,
                       temp: float = 1.0, eos_threshold: float = 0.0, decode_every: int = 1):
        from pocket_tts.modules.stateful_module import init_states

        tokens = self._tokens(text)
        max_frames = int(max_seconds * self.frame_rate)
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=10_000)
        buffer = []
        for z in self.model.stream_latents(tokens, max_frames, flow_steps, temp, eos_threshold):
            buffer.append(z)
            if len(buffer) >= decode_every:
                yield self._decode(buffer, mimi_state)
                buffer = []
        if buffer:
            yield self._decode(buffer, mimi_state)


if __name__ == "__main__":
    model = PocketLFM(PocketLFMConfig())
    for key, val in model.param_breakdown().items():
        print(f"{key:14s}: {val / 1e6:7.2f}M")

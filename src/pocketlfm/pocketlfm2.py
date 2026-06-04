from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LFM2Config:
    d_model: int = 1024
    n_layers: int = 6
    attn_layers: tuple[int, ...] = (2, 5)
    n_heads: int = 16
    n_kv_heads: int = 4
    conv_kernel: int = 4
    ff_dim: int = 2816
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    attn_window: int | None = None

    latent_dim: int = 32
    text_vocab: int = 4000
    text_dim: int = 512

    mtp_horizon: int = 4
    flow_dim: int = 768
    flow_depth: int = 6
    flow_freq_dim: int = 256

    eos_weight: float = 1.0
    input_noise: float = 0.0

    @property
    def head_dim(self) -> int:
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
    half = x.shape[-1] // 2
    inv_freq = theta ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)
    ang = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return (x.float() * cos + rot.float() * sin).to(x.dtype)


class ShortConv(nn.Module):
    def __init__(self, cfg: LFM2Config):
        super().__init__()
        d, k = cfg.d_model, cfg.conv_kernel
        self.k = k
        self.proj_in = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(d, d, k, groups=d, bias=False)
        self.proj_out = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None):
        b, _, d = x.shape
        gate_in, gate_out, val = self.proj_in(x).chunk(3, dim=-1)
        y = (gate_in * val).transpose(1, 2)
        pad = state if state is not None else y.new_zeros(b, d, self.k - 1)
        y = torch.cat([pad, y], dim=-1)
        out = self.conv(y).transpose(1, 2)
        new_state = y[..., -(self.k - 1):] if self.k > 1 else None
        return self.proj_out(gate_out * out), new_state


class Attention(nn.Module):
    def __init__(self, cfg: LFM2Config):
        super().__init__()
        self.nh, self.nkv, self.dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.theta = cfg.rope_theta
        self.window = cfg.attn_window
        d = cfg.d_model
        self.q_proj = nn.Linear(d, self.nh * self.dh, bias=False)
        self.k_proj = nn.Linear(d, self.nkv * self.dh, bias=False)
        self.v_proj = nn.Linear(d, self.nkv * self.dh, bias=False)
        self.o_proj = nn.Linear(self.nh * self.dh, d, bias=False)

    def _mask(self, q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor:
        delta = q_pos[:, None] - k_pos[None, :]
        mask = delta >= 0
        if self.window is not None:
            mask = mask & (delta < self.window)
        return mask[None, None]

    def forward(self, x, state, positions, streaming):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.nh, self.dh).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.nkv, self.dh).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.nkv, self.dh).transpose(1, 2)
        q = apply_rope(q, positions, self.theta)
        k = apply_rope(k, positions, self.theta)

        if not streaming:
            mask = self._mask(positions, positions)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
            return self.o_proj(out.transpose(1, 2).reshape(b, t, -1)), None

        if state is not None:
            k = torch.cat([state["k"], k], dim=2)
            v = torch.cat([state["v"], v], dim=2)
            k_pos = torch.cat([state["pos"], positions])
        else:
            k_pos = positions
        if self.window is not None and k.shape[2] > self.window:
            k, v, k_pos = k[:, :, -self.window:], v[:, :, -self.window:], k_pos[-self.window:]
        mask = self._mask(positions, k_pos)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        out = self.o_proj(out.transpose(1, 2).reshape(b, t, -1))
        return out, {"k": k, "v": v, "pos": k_pos}


class Block(nn.Module):
    def __init__(self, cfg: LFM2Config, is_attn: bool):
        super().__init__()
        self.is_attn = is_attn
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = Attention(cfg) if is_attn else ShortConv(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ff = SwiGLU(cfg.d_model, cfg.ff_dim)

    def forward(self, x, state, positions, streaming):
        mix_state = state["m"] if state is not None else None
        if self.is_attn:
            update, new = self.mixer(self.norm1(x), mix_state, positions, streaming)
        else:
            update, new = self.mixer(self.norm1(x), mix_state)
        x = x + update
        x = x + self.ff(self.norm2(x))
        return x, ({"m": new} if streaming else None)


class LFM2Backbone(nn.Module):
    def __init__(self, cfg: LFM2Config):
        super().__init__()
        attn = set(cfg.attn_layers)
        self.layers = nn.ModuleList([Block(cfg, i in attn) for i in range(cfg.n_layers)])

    def init_state(self) -> dict:
        return {"offset": 0, "layers": [None] * len(self.layers)}

    def forward(self, x, state=None):
        t = x.shape[1]
        offset = 0 if state is None else state["offset"]
        positions = torch.arange(offset, offset + t, device=x.device)
        streaming = state is not None
        for i, layer in enumerate(self.layers):
            ls = None if state is None else state["layers"][i]
            x, new = layer(x, ls, positions, streaming)
            if state is not None:
                state["layers"][i] = new
        if state is not None:
            state["offset"] = offset + t
        return x


class TimestepEmbed(nn.Module):
    def __init__(self, dim: int, freq_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        half = freq_dim // 2
        self.register_buffer(
            "freqs", torch.exp(-math.log(10_000.0) * torch.arange(half) / half), persistent=False
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t * self.freqs.to(t.dtype)
        return self.mlp(torch.cat([args.cos(), args.sin()], dim=-1))


class FlowResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))

    def forward(self, x, c):
        shift, scale, gate = self.ada(c).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class FlowHead(nn.Module):
    def __init__(self, cfg: LFM2Config):
        super().__init__()
        self.latent_dim = cfg.latent_dim
        d = cfg.flow_dim
        self.input_proj = nn.Linear(cfg.latent_dim, d)
        self.cond_proj = nn.Linear(cfg.d_model, d)
        self.t_start = TimestepEmbed(d, cfg.flow_freq_dim)
        self.t_end = TimestepEmbed(d, cfg.flow_freq_dim)
        self.blocks = nn.ModuleList([FlowResBlock(d) for _ in range(cfg.flow_depth)])
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(d, cfg.latent_dim)

    def velocity(self, c, s, t, x):
        y = (self.t_start(s) + self.t_end(t)) / 2 + self.cond_proj(c)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, y)
        return self.out(self.norm(h))

    def loss(self, c, target):
        noise = torch.randn_like(target)
        s = torch.rand(c.shape[0], 1, device=c.device)
        t = torch.rand(c.shape[0], 1, device=c.device)
        lo, hi = torch.minimum(s, t), torch.maximum(s, t)
        x_lo = (1 - lo) * noise + lo * target
        return F.mse_loss(self.velocity(c, lo, hi, x_lo), target - noise)

    @torch.no_grad()
    def sample(self, c, steps, temp):
        x = torch.randn(c.shape[0], self.latent_dim, device=c.device) * math.sqrt(temp)
        for i in range(steps):
            s = torch.full((c.shape[0], 1), i / steps, device=c.device)
            t = torch.full((c.shape[0], 1), (i + 1) / steps, device=c.device)
            x = x + self.velocity(c, s, t, x) / steps
        return x


class MTPHeads(nn.Module):
    def __init__(self, cfg: LFM2Config):
        super().__init__()
        self.horizon = cfg.mtp_horizon
        self.latent_dim = cfg.latent_dim
        self.offsets = nn.Parameter(torch.randn(cfg.mtp_horizon, cfg.d_model) * 0.02)
        self.head = FlowHead(cfg)

    def loss(self, hidden, targets, mask):
        cond = hidden[:, None, :] + self.offsets[None, :, :]
        return self.head.loss(cond[mask], targets[mask])

    def sample(self, hidden, steps, temp):
        return torch.stack(
            [self.head.sample(hidden + self.offsets[k], steps, temp) for k in range(self.horizon)],
            dim=1,
        )


class TextEmbedder(nn.Module):
    def __init__(self, cfg: LFM2Config):
        super().__init__()
        self.pad = cfg.text_vocab
        self.embed = nn.Embedding(cfg.text_vocab + 1, cfg.text_dim)
        self.proj = nn.Linear(cfg.text_dim, cfg.d_model, bias=False)

    def forward(self, tokens, lengths):
        tokens = tokens.clone()
        if lengths is not None:
            pad = torch.arange(tokens.shape[1], device=tokens.device)[None, :] >= lengths[:, None]
            tokens[pad] = self.pad
        return self.proj(self.embed(tokens))


class PocketLFM2(nn.Module):
    def __init__(self, cfg: LFM2Config):
        super().__init__()
        self.cfg = cfg
        self.text = TextEmbedder(cfg)
        self.latent_in = nn.Linear(cfg.latent_dim, cfg.d_model, bias=False)
        self.bos = nn.Parameter(torch.randn(cfg.latent_dim))
        self.backbone = LFM2Backbone(cfg)
        self.out_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mtp = MTPHeads(cfg)
        self.eos = nn.Linear(cfg.d_model, 1)
        self.register_buffer("latent_mean", torch.zeros(cfg.latent_dim))
        self.register_buffer("latent_std", torch.ones(cfg.latent_dim))

    def set_latent_stats(self, mean, std):
        self.latent_mean.copy_(mean.to(self.latent_mean))
        self.latent_std.copy_(std.clamp(min=1e-4).to(self.latent_std))

    def normalize(self, x):
        return (x - self.latent_mean) / self.latent_std

    def denormalize(self, x):
        return x * self.latent_std + self.latent_mean

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_breakdown(self) -> dict[str, float]:
        def m(module):
            return sum(p.numel() for p in module.parameters()) / 1e6

        return {
            "backbone": m(self.backbone),
            "mtp": m(self.mtp),
            "text": m(self.text),
            "total": self.num_parameters() / 1e6,
        }

    def _left_align(self, tokens, lengths):
        lt = tokens.shape[1]
        idx = torch.arange(lt, device=tokens.device)[None, :] - (lt - lengths[:, None])
        rolled = torch.gather(tokens, 1, idx.clamp(min=0))
        return torch.where(idx >= 0, rolled, torch.full_like(rolled, self.text.pad))

    def loss(self, text_tokens, text_lens, latents, lat_lens) -> dict[str, torch.Tensor]:
        device = latents.device
        text_lens, lat_lens = text_lens.to(device), lat_lens.to(device)
        data = self.normalize(latents)
        b, s, ld = data.shape
        k = self.cfg.mtp_horizon
        lt_max = text_tokens.shape[1]

        text_emb = self.text(self._left_align(text_tokens.to(device), text_lens), None)
        ctx = data
        if self.training and self.cfg.input_noise > 0:
            ctx = data + self.cfg.input_noise * torch.randn_like(data)

        audio_in = torch.cat([self.bos[None, None, :].expand(b, 1, ld), ctx[:, :-1]], dim=1)
        packed = torch.cat([text_emb, self.latent_in(audio_in)], dim=1)
        hidden = self.out_norm(self.backbone(packed))
        audio_h = hidden[:, lt_max:]

        lat_mask = torch.arange(s, device=device)[None, :] < lat_lens[:, None]
        targets = data.new_zeros(b, s, k, ld)
        mask = torch.zeros(b, s, k, dtype=torch.bool, device=device)
        for j in range(k):
            if s - j > 0:
                targets[:, : s - j, j] = data[:, j:]
                mask[:, : s - j, j] = lat_mask[:, j:] & lat_mask[:, : s - j]

        flow_loss = self.mtp.loss(
            audio_h.reshape(b * s, -1), targets.reshape(b * s, k, ld), mask.reshape(b * s, k)
        )

        eos_logits = self.eos(audio_h).squeeze(-1)
        eos_target = torch.zeros(b, s, device=device)
        eos_target[torch.arange(b, device=device), (lat_lens - 1).clamp(min=0)] = 1.0
        eos_loss = F.binary_cross_entropy_with_logits(eos_logits[lat_mask], eos_target[lat_mask])

        total = flow_loss + self.cfg.eos_weight * eos_loss
        return {"flow": flow_loss, "eos": eos_loss, "total": total}

    @torch.no_grad()
    def generate(self, text_tokens, max_frames, num_steps=8, temp=1.0, eos_threshold=0.0):
        device = text_tokens.device
        b = text_tokens.shape[0]
        text_emb = self.text(text_tokens, torch.full((b,), text_tokens.shape[1], device=device))
        state = self.backbone.init_state()
        self.backbone(text_emb, state)

        prev = self.bos[None, :].expand(b, -1)[:, None, :]
        produced = 0
        frames = []
        while produced < max_frames:
            hidden = self.out_norm(self.backbone(self.latent_in(prev), state))[:, -1]
            block = self.mtp.sample(hidden, num_steps, temp)
            for j in range(self.cfg.mtp_horizon):
                frames.append(self.denormalize(block[:, j]))
                produced += 1
                if produced >= max_frames:
                    break
            prev = block
            if (self.eos(hidden) > eos_threshold).all():
                break
        return torch.stack(frames, dim=1)


class LFM2Pipeline:
    def __init__(self, model: PocketLFM2, mimi, tokenizer, frame_rate: float = 12.5):
        self.model = model
        self.mimi = mimi
        self.tokenizer = tokenizer
        self.frame_rate = frame_rate

    @classmethod
    def from_checkpoint(cls, path: str, language: str = "english", device: str = "cpu"):
        from pocketlfm.pocketlfm import load_pretrained_codec

        ckpt = torch.load(path, map_location=device)
        model = PocketLFM2(LFM2Config(**ckpt["cfg"]))
        model.load_state_dict(ckpt["model"])
        model.eval()
        mimi, tokenizer, _, frame_rate = load_pretrained_codec(language)
        pipe = cls(model, mimi, tokenizer, frame_rate)
        pipe.model.to(device)
        pipe.mimi.to(device)
        return pipe

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def sample_rate(self) -> int:
        return int(self.mimi.sample_rate)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        return self.mimi.encode_to_latent(audio)[0].transpose(0, 1).contiguous()

    @torch.no_grad()
    def synthesize(self, text: str, max_seconds: float = 10.0, num_steps: int = 8,
                   temp: float = 1.0, eos_threshold: float = 0.0) -> torch.Tensor:
        from pocket_tts.modules.stateful_module import init_states

        tokens = torch.tensor([self.tokenizer.sp.encode(text, out_type=int)],
                              dtype=torch.long, device=self.device)
        max_frames = int(max_seconds * self.frame_rate)
        latents = self.model.generate(tokens, max_frames, num_steps, temp, eos_threshold)
        chan = latents.transpose(1, 2)
        if chan.shape[1] == self.mimi.quantizer.dimension:
            chan = self.mimi.quantizer(chan)
        mimi_state = init_states(self.mimi, batch_size=latents.shape[0], sequence_length=10_000)
        return self.mimi.decode_from_latent(chan, mimi_state)


if __name__ == "__main__":
    model = PocketLFM2(LFM2Config())
    for key, val in model.param_breakdown().items():
        print(f"{key:10s}: {val:6.1f}M")
    tt = torch.randint(0, 4000, (2, 12))
    tl = torch.tensor([12, 9])
    lat = torch.randn(2, 40, 32)
    ll = torch.tensor([40, 30])
    out = model.loss(tt, tl, lat, ll)
    print({k: round(float(v), 3) for k, v in out.items()})
    g = model.generate(tt[:1], max_frames=8, num_steps=4, eos_threshold=1e9)
    print("generate:", tuple(g.shape))

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time

import torch

FRAME_RATE = 12.5
SAMPLE_RATE = 24000


def _decode(mimi, denorm_latents):
    from pocket_tts.modules.stateful_module import init_states

    chan = denorm_latents.transpose(1, 2)
    if chan.shape[1] == mimi.quantizer.dimension:
        chan = mimi.quantizer(chan)
    state = init_states(mimi, batch_size=1, sequence_length=10_000)
    return mimi.decode_from_latent(chan, state)


def _load_lfm2(args):
    from pocketlfm.pocketlfm import load_pretrained_codec
    from pocketlfm.pocketlfm2 import LFM2Config, PocketLFM2

    mimi, tok, _, _ = load_pretrained_codec(args.language)
    mimi = mimi.float().eval()
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu")
        model = PocketLFM2(LFM2Config(**ck["cfg"]))
        model.load_state_dict(ck["model"])
    else:
        model = PocketLFM2(LFM2Config())
    model = model.float().eval()
    tokens = torch.tensor([tok.sp.encode(args.text, out_type=int)], dtype=torch.long)

    def gen(n):
        return model.generate(tokens, max_frames=n, num_steps=args.num_steps, eos_threshold=1e9)

    params = model.num_parameters()
    weight = sum(p.numel() * p.element_size() for p in model.parameters())
    return mimi, gen, params, weight


def _load_pockettts(args):
    from pocket_tts.conditioners.base import TokenizedText
    from pocket_tts.models.tts_model import TTSModel
    from pocket_tts.modules.stateful_module import increment_steps, init_states

    tts = TTSModel.load_model(language=args.language)
    flow = tts.flow_lm.float().eval()
    mimi = tts.mimi.float().eval()
    tok = flow.conditioner.tokenizer
    tokens = torch.tensor([tok.sp.encode(args.text, out_type=int)], dtype=torch.long)
    text_emb = flow.conditioner(TokenizedText(tokens))

    def gen(n):
        state = init_states(flow, batch_size=1, sequence_length=n + tokens.shape[1] + 4)
        seq = torch.full((1, 1, flow.ldim), float("nan"))
        frames, first = [], True
        for _ in range(n):
            te = text_emb if first else text_emb.new_zeros((1, 0, flow.dim))
            out, _ = flow._sample_next_latent(
                sequence=seq, text_embeddings=te, model_state=state,
                lsd_decode_steps=args.num_steps, temp=1.0, noise_clamp=None, eos_threshold=1e9)
            increment_steps(flow, state, increment=te.shape[1] + seq.shape[1])
            first = False
            frames.append(out)
            seq = out[:, None, :]
        return torch.stack(frames, dim=1) * flow.emb_std + flow.emb_mean

    params = sum(p.numel() for p in flow.parameters())
    weight = sum(p.numel() * p.element_size() for p in flow.parameters())
    return mimi, gen, params, weight


def run_worker(args) -> None:
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    mimi, gen, params, weight = (_load_lfm2 if args.worker == "lfm2" else _load_pockettts)(args)

    n_frames = int(args.seconds * FRAME_RATE)
    gen(8)
    t = time.monotonic()
    lat = gen(n_frames)
    lm_s = time.monotonic() - t
    t = time.monotonic()
    wav = _decode(mimi, lat)
    dec_s = time.monotonic() - t

    audio_s = wav.shape[-1] / SAMPLE_RATE
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print("RESULT " + json.dumps({
        "model": args.worker, "params_m": params / 1e6, "weight_mb": weight / 1e6,
        "peak_rss_mb": peak, "lm_s": lm_s, "dec_s": dec_s, "audio_s": audio_s,
        "lm_rtf": lm_s / audio_s, "e2e_rtf": (lm_s + dec_s) / audio_s}))


def print_table(results: dict) -> None:
    a, b = results.get("lfm2"), results.get("pockettts")
    rows = [
        ("params (M)", "params_m", "{:.1f}"),
        ("LM weights (MB)", "weight_mb", "{:.0f}"),
        ("peak RSS (MB)", "peak_rss_mb", "{:.0f}"),
        ("LM gen (s)", "lm_s", "{:.2f}"),
        ("decode (s)", "dec_s", "{:.2f}"),
        ("audio (s)", "audio_s", "{:.2f}"),
        ("LM RTF", "lm_rtf", "{:.3f}"),
        ("end-to-end RTF", "e2e_rtf", "{:.3f}"),
    ]
    print(f"\n{'metric':<18}{'PocketLFM2':>14}{'PocketTTS':>14}{'ratio':>10}")
    print("-" * 56)
    for label, key, fmt in rows:
        va = fmt.format(a[key]) if a else "-"
        vb = fmt.format(b[key]) if b else "-"
        ratio = f"{b[key] / a[key]:.2f}x" if (a and b and a[key]) else "-"
        print(f"{label:<18}{va:>14}{vb:>14}{ratio:>10}")
    print("\n(RTF < 1 = faster than real-time. ratio = PocketTTS / PocketLFM2.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="RTF + memory benchmark: PocketLFM2 vs original PocketTTS.")
    ap.add_argument("--ckpt", default=None, help="PocketLFM2 checkpoint (else random weights)")
    ap.add_argument("--text", default="the quick brown fox jumps over the lazy dog and then runs away")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--num-steps", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--language", default="english")
    ap.add_argument("--worker", default=None, choices=[None, "lfm2", "pockettts"])
    args = ap.parse_args()

    if args.worker:
        run_worker(args)
        return

    print(f"benchmark | {args.seconds:.0f}s target | {args.num_steps} flow steps | "
          f"{args.threads} threads\n", flush=True)
    results = {}
    for model in ("lfm2", "pockettts"):
        cmd = [sys.executable, __file__, "--worker", model, "--text", args.text,
               "--seconds", str(args.seconds), "--num-steps", str(args.num_steps),
               "--threads", str(args.threads), "--language", args.language]
        if args.ckpt and model == "lfm2":
            cmd += ["--ckpt", args.ckpt]
        print(f"running {model} (isolated process) ...", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
        if line is None:
            print(proc.stdout[-1500:])
            print(proc.stderr[-1500:])
            continue
        results[model] = json.loads(line[len("RESULT "):])
    print_table(results)


if __name__ == "__main__":
    main()

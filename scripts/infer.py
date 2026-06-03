"""Synthesize speech from text with a trained PocketLFM checkpoint.

Examples:
    # use a trained checkpoint
    python infer.py --ckpt runs/pocketlfm_ljspeech/pocketlfm_best.pt \
        --text "Hello, this is the LFM backbone speaking." --out sample.wav

    # quick architecture smoke (random weights -> noise) without a checkpoint
    python infer.py --text "testing one two three" --out noise.wav

    # fixed-length RTF benchmark (ignore the EOS head)
    python infer.py --ckpt runs/.../pocketlfm_best.pt --text "..." \
        --force-length --max-seconds 5 --benchmark
"""

from __future__ import annotations

import argparse
import time

import torch

from pocketlfm import PocketLFMConfig, PocketLFMPipeline


def build_pipeline(args: argparse.Namespace, device: str) -> PocketLFMPipeline:
    if args.ckpt:
        print(f"loading checkpoint {args.ckpt}")
        pipe = PocketLFMPipeline.from_checkpoint(args.ckpt, language=args.language, device=device)
    else:
        print("no --ckpt given: using a RANDOM backbone (output will be noise)")
        pipe = PocketLFMPipeline.from_pretrained(
            language=args.language, cfg=PocketLFMConfig()
        ).to(device)
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser(description="PocketLFM text-to-speech inference.")
    ap.add_argument("--ckpt", default=None, help="trained checkpoint (.pt); omit for random weights")
    ap.add_argument("--text", default="Hello world, this is the LFM backbone speaking.")
    ap.add_argument("--out", default="sample.wav", help="output WAV path")
    ap.add_argument("--language", default="english")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--flow-steps", type=int, default=1, help="ODE sampling steps per frame")
    ap.add_argument("--temp", type=float, default=1.0, help="sampling temperature")
    ap.add_argument("--eos-threshold", type=float, default=0.0)
    ap.add_argument("--force-length", action="store_true",
                    help="ignore EOS and generate exactly --max-seconds (for benchmarking)")
    ap.add_argument("--benchmark", action="store_true", help="warm up + report RTF")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    eos_threshold = float("inf") if args.force_length else args.eos_threshold

    pipe = build_pipeline(args, device)
    sr = pipe.sample_rate
    print(f"device={device} | sample_rate={sr} | frame_rate={pipe.frame_rate} | "
          f"model={pipe.model.num_parameters() / 1e6:.1f}M params")

    if args.benchmark:
        pipe.generate("warm up", max_seconds=1.0, flow_steps=args.flow_steps,
                      eos_threshold=float("inf"))

    t0 = time.time()
    wav = pipe.generate(
        args.text,
        max_seconds=args.max_seconds,
        flow_steps=args.flow_steps,
        temp=args.temp,
        eos_threshold=eos_threshold,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    wav = wav.detach().to("cpu", torch.float32)[0]  # [C, samples]
    secs = wav.shape[-1] / sr
    print(f'generated {secs:.2f}s of audio in {dt:.2f}s wall | RTF {secs / dt:.2f}x '
          f'("{args.text[:50]}")')

    try:
        import torchaudio

        torchaudio.save(args.out, wav, sr)
    except Exception:  # pragma: no cover - torchaudio backend fallback
        import scipy.io.wavfile

        scipy.io.wavfile.write(args.out, sr, wav.numpy().T)
    print(f"saved {args.out}")
    if args.ckpt is None:
        print("note: random backbone -> this WAV is noise. Train first, then pass --ckpt.")


if __name__ == "__main__":
    main()

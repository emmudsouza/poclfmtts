from __future__ import annotations

import argparse

import scipy.io.wavfile
import torch

from pocketlfm.pocketlfm2 import LFM2Pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize speech with a trained PocketLFM2 model.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", default="hello there, how are you today")
    ap.add_argument("--out", default="lfm2.wav")
    ap.add_argument("--language", default="english")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-steps", type=int, default=8)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--eos-threshold", type=float, default=0.0)
    args = ap.parse_args()

    pipe = LFM2Pipeline.from_checkpoint(args.ckpt, language=args.language, device=args.device)
    wav = pipe.synthesize(args.text, max_seconds=args.max_seconds, num_steps=args.num_steps,
                          temp=args.temp, eos_threshold=args.eos_threshold)
    arr = wav.squeeze().detach().cpu().numpy()
    scipy.io.wavfile.write(args.out, pipe.sample_rate, arr)
    print(f"wrote {args.out} ({arr.shape[-1] / pipe.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()

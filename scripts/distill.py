"""Distill PocketTTS into the PocketLFM latent space.

Synthesize a single fixed voice over a text corpus with the PocketTTS teacher and cache
the resulting Mimi latents in the exact format `data_ljspeech.py` produces — so `train.py`
consumes it unchanged. Distilled targets are clean and text-aligned, so the student learns
to speak far faster than training from scratch on real audio.

The voice defaults to a ~20s reference built from a few LJSpeech clips (a single female
English speaker). Pass --voice to use any wav / hf:// URL instead.

Usage:
    python distill.py --out data/distill_cache --limit 2000
    python train.py --cache data/distill_cache --out runs/distill
"""

from __future__ import annotations

import argparse
import pathlib

import torch


def build_voice_prompt(root: str, sr: int, n_clips: int, device: str) -> torch.Tensor:
    import torchaudio

    from data_ljspeech import read_metadata

    rows = read_metadata(root)
    chunks = []
    for cid, _ in rows[:n_clips]:
        wav, file_sr = torchaudio.load(str(pathlib.Path(root) / "wavs" / f"{cid}.wav"))
        wav = wav.mean(0, keepdim=True)
        if file_sr != sr:
            wav = torchaudio.functional.resample(wav, file_sr, sr)
        chunks.append(wav)
    return torch.cat(chunks, dim=1).to(device)  # [1, samples]


def main() -> None:
    ap = argparse.ArgumentParser(description="Distill PocketTTS into a single-voice latent cache.")
    ap.add_argument("--root", default="data/LJSpeech-1.1", help="LJSpeech root (for text + voice)")
    ap.add_argument("--out", default="data/distill_cache")
    ap.add_argument("--voice", default=None, help="voice prompt wav / hf:// URL; default builds one from LJSpeech")
    ap.add_argument("--voice-clips", type=int, default=6, help="LJSpeech clips to concat for the default voice")
    ap.add_argument("--texts", default=None, help="text file (one per line); default = LJSpeech transcripts")
    ap.add_argument("--limit", type=int, default=2000, help="number of utterances to synthesize")
    ap.add_argument("--device", default=None)
    ap.add_argument("--language", default="english")
    args = ap.parse_args()

    from data_ljspeech import read_metadata
    from pocket_tts.models.tts_model import TTSModel

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tts = TTSModel.load_model(language=args.language).to(device)
    mimi = tts.mimi
    sr = int(mimi.sample_rate)
    tokenizer = tts.flow_lm.conditioner.tokenizer

    if args.voice:
        voice_state = tts.get_state_for_audio_prompt(args.voice)
    else:
        prompt = build_voice_prompt(args.root, sr, args.voice_clips, device)
        voice_state = tts.get_state_for_audio_prompt(prompt)
    print(f"voice prompt ready | device={device} | sr={sr}")

    if args.texts:
        texts = [ln.strip() for ln in open(args.texts, encoding="utf-8") if ln.strip()]
        ids = [f"D{i:06d}" for i in range(len(texts))]
    else:
        rows = read_metadata(args.root)
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
    if args.limit:
        ids, texts = ids[: args.limit], texts[: args.limit]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index, done, skipped = [], 0, 0
    for i, (cid, text) in enumerate(zip(ids, texts)):
        path = out / f"{cid}.pt"
        if path.exists():
            index.append(cid)
            continue
        try:
            with torch.no_grad():
                audio = tts.generate_audio(voice_state, text, copy_state=True)  # [C, samples]
                audio = audio.to(device)
                if audio.dim() == 1:
                    audio = audio[None]
                if audio.dim() == 2:
                    audio = audio[None]  # [1, C, samples]
                if audio.shape[1] > 1:
                    audio = audio.mean(1, keepdim=True)
                latent = mimi.encode_to_latent(audio)  # [1, 32, T]
            latent = latent[0].transpose(0, 1).contiguous().to("cpu", torch.float16)  # [T, 32]
            tokens = torch.tensor(tokenizer.sp.encode(text, out_type=int), dtype=torch.long)
            torch.save({"tokens": tokens, "latents": latent}, path)
            index.append(cid)
            done += 1
        except Exception as exc:  # noqa: BLE001 - skip a bad utterance, keep going
            skipped += 1
            if skipped <= 5:
                print(f"  skip {cid}: {exc}")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ids)} synthesized ({done} ok, {skipped} skipped)")

    (out / "index.txt").write_text("\n".join(index))
    print(f"Done. {len(index)} clips distilled to {out} (skipped {skipped}).")


if __name__ == "__main__":
    main()

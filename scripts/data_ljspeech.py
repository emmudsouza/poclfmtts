"""LJSpeech data pipeline for PocketLFM training.

Three stages:
  1. `download_ljspeech` — fetch + extract the LJSpeech-1.1 corpus (public domain).
  2. `preprocess` — encode every clip to 32-dim Mimi latents @ 12.5 Hz and tokenize the
     text with the PocketTTS SentencePiece tokenizer, caching results to disk. This runs
     the (frozen) Mimi encoder ONCE so training never needs the codec in memory — important
     on a 4 GB GPU.
  3. `LJSpeechLatents` Dataset + `collate` — yield (tokens, latents) and right-pad batches.

Run preprocessing from the CLI:
    python data_ljspeech.py --root data/LJSpeech-1.1 --out data/ljspeech_cache [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import tarfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset

LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
LJSPEECH_SR = 22050


# ----------------------------------------------------------------------------------------
# Stage 1: download
# ----------------------------------------------------------------------------------------


def download_ljspeech(dest: str | Path) -> Path:
    """Download + extract LJSpeech-1.1 into `dest` (parent dir). Returns the dataset root."""
    dest = Path(dest)
    root = dest / "LJSpeech-1.1"
    n_meta = len(read_metadata(root)) if (root / "metadata.csv").exists() else 0
    n_wav = len(list((root / "wavs").glob("*.wav"))) if (root / "wavs").exists() else 0
    if n_meta and n_wav >= n_meta:
        print(f"LJSpeech already present at {root} ({n_wav} wav files)")
        return root
    if n_meta:
        print(f"LJSpeech incomplete ({n_wav}/{n_meta} wavs present) — re-extracting")
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "LJSpeech-1.1.tar.bz2"
    if not tar_path.exists():
        print(f"Downloading LJSpeech (~2.6 GB) from {LJSPEECH_URL} ...")
        urllib.request.urlretrieve(LJSPEECH_URL, tar_path)
    print("Extracting ...")
    try:
        with tarfile.open(tar_path, "r:bz2") as tf:
            tf.extractall(dest)
    except (tarfile.TarError, EOFError):
        print("Archive is corrupt — re-downloading ...")
        tar_path.unlink(missing_ok=True)
        urllib.request.urlretrieve(LJSPEECH_URL, tar_path)
        with tarfile.open(tar_path, "r:bz2") as tf:
            tf.extractall(dest)
    return root


def read_metadata(root: str | Path) -> list[tuple[str, str]]:
    """Return [(clip_id, normalized_text), ...] from metadata.csv."""
    root = Path(root)
    rows: list[tuple[str, str]] = []
    with open(root / "metadata.csv", encoding="utf-8") as f:
        for parts in csv.reader(f, delimiter="|"):
            if len(parts) >= 3:
                clip_id, _raw, norm = parts[0], parts[1], parts[2]
                rows.append((clip_id, norm))
    return rows


# ----------------------------------------------------------------------------------------
# Stage 2: preprocess to cached latents + token ids
# ----------------------------------------------------------------------------------------


def preprocess(
    root: str | Path,
    out: str | Path,
    language: str = "english",
    device: str | None = None,
    limit: int | None = None,
) -> None:
    """Encode each clip to [T, 32] latents (fp16) + token ids, cached as <out>/<id>.pt."""
    import torchaudio  # local import: only needed for preprocessing

    from pocketlfm import load_pretrained_codec

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    mimi, tokenizer, sample_rate, _frame_rate = load_pretrained_codec(language)
    mimi = mimi.to(device)
    resample = torchaudio.transforms.Resample(LJSPEECH_SR, sample_rate).to(device)

    rows = read_metadata(root)
    if limit is not None:
        rows = rows[:limit]

    index: list[str] = []
    skipped = 0
    for i, (clip_id, text) in enumerate(rows):
        cache_path = out / f"{clip_id}.pt"
        if cache_path.exists():
            index.append(clip_id)
            continue
        wav_path = root / "wavs" / f"{clip_id}.wav"
        if not wav_path.exists():
            skipped += 1
            continue
        wav, sr = torchaudio.load(str(wav_path))  # [C, T]
        wav = wav.mean(0, keepdim=True).to(device)  # mono [1, T]
        if sr != LJSPEECH_SR:
            wav = torchaudio.functional.resample(wav, sr, sample_rate).to(device)
        else:
            wav = resample(wav)
        with torch.no_grad():
            latent = mimi.encode_to_latent(wav[None])  # [1, 32, T]
        latent = latent[0].transpose(0, 1).contiguous().to("cpu", torch.float16)  # [T, 32]
        tokens = torch.tensor(tokenizer.sp.encode(text, out_type=int), dtype=torch.long)
        torch.save({"tokens": tokens, "latents": latent}, cache_path)
        index.append(clip_id)
        if (i + 1) % 200 == 0:
            print(f"  preprocessed {i + 1}/{len(rows)} (skipped {skipped})")

    (out / "index.txt").write_text("\n".join(index))
    print(f"Done. {len(index)} clips cached to {out} (skipped {skipped}).")


# ----------------------------------------------------------------------------------------
# Stage 3: Dataset + collate
# ----------------------------------------------------------------------------------------


class LJSpeechLatents(Dataset):
    """Yields (tokens[Lt] long, latents[T, 32] float32) from the preprocessed cache."""

    def __init__(self, cache_dir: str | Path, max_frames: int | None = None):
        self.cache_dir = Path(cache_dir)
        index_file = self.cache_dir / "index.txt"
        if not index_file.exists():
            raise FileNotFoundError(
                f"No index.txt in {cache_dir}. Run preprocessing first (python data_ljspeech.py ...)."
            )
        self.ids = [line for line in index_file.read_text().splitlines() if line]
        self.max_frames = max_frames

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = torch.load(self.cache_dir / f"{self.ids[idx]}.pt")
        tokens = rec["tokens"].long()
        latents = rec["latents"].float()
        if self.max_frames is not None and latents.shape[0] > self.max_frames:
            latents = latents[: self.max_frames]
        return tokens, latents


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Right-pad a batch. Returns text_tokens, text_lens, latents, lat_lens."""
    tokens, latents = zip(*batch)
    text_lens = torch.tensor([t.shape[0] for t in tokens], dtype=torch.long)
    lat_lens = torch.tensor([x.shape[0] for x in latents], dtype=torch.long)
    lt_max = int(text_lens.max())
    t_max = int(lat_lens.max())
    latent_dim = latents[0].shape[1]

    text_tokens = torch.zeros(len(batch), lt_max, dtype=torch.long)
    lat_padded = torch.zeros(len(batch), t_max, latent_dim)
    for i, (tok, lat) in enumerate(zip(tokens, latents)):
        text_tokens[i, : tok.shape[0]] = tok
        lat_padded[i, : lat.shape[0]] = lat
    return {
        "text_tokens": text_tokens,
        "text_lens": text_lens,
        "latents": lat_padded,
        "lat_lens": lat_lens,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocess LJSpeech into PocketLFM latents.")
    ap.add_argument("--root", default="data/LJSpeech-1.1", help="LJSpeech root (with metadata.csv)")
    ap.add_argument("--out", default="data/ljspeech_cache", help="output cache dir")
    ap.add_argument("--download", action="store_true", help="download LJSpeech first")
    ap.add_argument("--download-dir", default="data", help="where to download/extract")
    ap.add_argument("--language", default="english")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--limit", type=int, default=None, help="only process first N clips")
    args = ap.parse_args()

    root = download_ljspeech(args.download_dir) if args.download else args.root
    preprocess(root, args.out, args.language, args.device, args.limit)

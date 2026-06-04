from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from data_ljspeech import LJSpeechLatents, collate

from pocketlfm.pocketlfm2 import LFM2Config, PocketLFM2


def latent_stats(dataset, n: int = 500):
    cat = torch.cat([dataset[i][1] for i in range(min(len(dataset), n))], dim=0)
    return cat.mean(0), cat.std(0)


def evaluate(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            out = model.loss(batch["text_tokens"], batch["text_lens"],
                             batch["latents"].to(device), batch["lat_lens"])
            total += float(out["total"])
            count += 1
    return total / max(1, count)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PocketLFM2 (LFM2 backbone) on a latent cache.")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="runs/lfm2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--val-split", type=float, default=0.02)
    ap.add_argument("--input-noise", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = args.device
    dataset = LJSpeechLatents(args.cache, preload=True)
    if args.limit:
        dataset = Subset(dataset, range(min(len(dataset), args.limit)))

    latent_dim = dataset[0][1].shape[-1]
    cfg = LFM2Config(latent_dim=latent_dim, input_noise=args.input_noise)
    model = PocketLFM2(cfg).to(device)
    mean, std = latent_stats(dataset)
    model.set_latent_stats(mean, std)
    for key, val in model.param_breakdown().items():
        print(f"  {key:10s}: {val:6.1f}M")
    if args.compile:
        model = torch.compile(model)

    n_val = max(1, int(len(dataset) * args.val_split))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = args.bf16 and device == "cuda"
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"train={len(train_ds)} val={len(val_ds)} | device={device} | bf16={amp}")

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running, steps, t0 = 0.0, 0, time.monotonic()
        for batch in train_dl:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out_losses = model.loss(batch["text_tokens"], batch["text_lens"],
                                        batch["latents"].to(device), batch["lat_lens"])
            loss = out_losses["total"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach())
            steps += 1
        train_loss = running / max(1, steps)
        val_loss = evaluate(model, val_dl, device)
        dt = time.monotonic() - t0
        print(f"epoch {epoch + 1}/{args.epochs} ({steps} steps, {dt:.0f}s) | "
              f"train {train_loss:.4f} | val {val_loss:.4f}")

        ckpt = {"cfg": asdict(cfg), "model": model.state_dict()}
        torch.save(ckpt, out / "lfm2_last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, out / "lfm2_best.pt")


if __name__ == "__main__":
    main()

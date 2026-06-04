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
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=0,
                    help="also checkpoint every N optimizer steps (0 = only per epoch)")
    ap.add_argument("--resume", default=None, help="resume same run (model+optimizer+epoch+best)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start weights from a checkpoint (fresh optimizer/epochs); for fine-tuning")
    args = ap.parse_args()

    device = args.device
    print(f"loading cache from {args.cache} ...", flush=True)
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
    core = model
    if args.compile:
        model = torch.compile(model)

    n_val = max(1, int(len(dataset) * args.val_split))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=args.workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = args.bf16 and device == "cuda"
    accum = max(1, args.grad_accum)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_batches = len(train_dl)

    best = float("inf")
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        core.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        best = ck.get("best", float("inf"))
        start_epoch = ck.get("epoch", 0)
        print(f"resumed from {args.resume} @ epoch {start_epoch} (best val {best:.4f})", flush=True)
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device, mmap=True)
        core.load_state_dict(ck["model"])
        print(f"warm-started weights from {args.init_from} "
              f"(fresh optimizer/epochs; keeping its latent stats)", flush=True)

    print(f"train={len(train_ds)} val={len(val_ds)} | device={device} | bf16={amp} | "
          f"batch={args.batch_size} x accum {accum} = eff {args.batch_size * accum} | "
          f"{n_batches} batches/epoch", flush=True)

    def save_state(val_loss, next_epoch):
        nonlocal best
        ckpt = {"cfg": asdict(cfg), "model": core.state_dict(), "opt": opt.state_dict(),
                "epoch": next_epoch, "best": min(best, val_loss)}
        torch.save(ckpt, out / "lfm2_last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, out / "lfm2_best.pt")
            print(f"  saved best (val {val_loss:.4f}) -> {out / 'lfm2_best.pt'}", flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, window, t0, last = 0.0, 0.0, time.monotonic(), time.monotonic()
        opt.zero_grad()
        opt_steps = 0
        for i, batch in enumerate(train_dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out_losses = model.loss(batch["text_tokens"], batch["text_lens"],
                                        batch["latents"].to(device), batch["lat_lens"])
            (out_losses["total"] / accum).backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                opt_steps += 1
                if args.save_every and opt_steps % args.save_every == 0:
                    val_loss = evaluate(model, val_dl, device)
                    print(f"  [mid-epoch] opt-step {opt_steps} | val {val_loss:.4f}", flush=True)
                    save_state(val_loss, epoch)
                    model.train()
            running += float(out_losses["total"].detach())
            window += float(out_losses["total"].detach())

            if (i + 1) % args.log_every == 0:
                now = time.monotonic()
                ms = (now - last) / args.log_every * 1000
                last = now
                eta = (n_batches - i - 1) * ms / 1000
                print(f"  ep {epoch + 1} [{i + 1:>5}/{n_batches}] "
                      f"loss {window / args.log_every:.4f} | {ms:.0f} ms/it | "
                      f"eta {eta / 60:.1f} min", flush=True)
                window = 0.0

        train_loss = running / max(1, n_batches)
        val_loss = evaluate(model, val_dl, device)
        dt = time.monotonic() - t0
        print(f"epoch {epoch + 1}/{args.epochs} done ({dt:.0f}s) | "
              f"train {train_loss:.4f} | val {val_loss:.4f}", flush=True)
        save_state(val_loss, epoch + 1)


if __name__ == "__main__":
    main()

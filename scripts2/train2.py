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


class EMA:
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def weights(self, ref_state):
        return {k: self.shadow[k].to(ref_state[k].dtype) for k in ref_state}

    def state_dict(self):
        return self.shadow

    def load(self, sd):
        for k in self.shadow:
            if k in sd:
                self.shadow[k].copy_(sd[k])


def make_scheduler(opt, warmup: int, total: int, final_ratio: float = 0.1):
    import math

    def fn(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train PocketLFM2 (LFM2 backbone) on a latent cache.")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="runs/lfm2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (Ampere+; safe everywhere, no speedup on T4/P100)")
    ap.add_argument("--fp16", action="store_true", help="float16 autocast + GradScaler (uses T4 tensor cores; ~2x on Turing)")
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
    ap.add_argument("--warmup", type=int, default=500, help="LR warmup steps, then cosine decay to 10%")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA decay (0 disables); EMA weights are used for val + saved as best (better audio)")
    ap.add_argument("--eos-weight", type=float, default=1.0,
                    help="weight on the stop/EOS loss; raise to ~3-5 for a cleaner automatic cutoff")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stop after this many validations with no improvement (0 = never)")
    args = ap.parse_args()

    device = args.device
    print(f"loading cache from {args.cache} ...", flush=True)
    dataset = LJSpeechLatents(args.cache, preload=True)
    if args.limit:
        dataset = Subset(dataset, range(min(len(dataset), args.limit)))

    latent_dim = dataset[0][1].shape[-1]
    cfg = LFM2Config(latent_dim=latent_dim, input_noise=args.input_noise, eos_weight=args.eos_weight)
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
    use_amp = (args.bf16 or args.fp16) and device == "cuda"
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device == "cuda"))
    accum = max(1, args.grad_accum)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_batches = len(train_dl)
    total_steps = max(1, (n_batches // accum) * args.epochs)
    scheduler = make_scheduler(opt, args.warmup, total_steps)
    ema = EMA(core, args.ema_decay) if args.ema_decay > 0 else None
    global_step = 0
    stale = 0

    best = float("inf")
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        core.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "scheduler" in ck:
            scheduler.load_state_dict(ck["scheduler"])
        if ema is not None and "ema" in ck:
            ema.load(ck["ema"])
        global_step = ck.get("step", 0)
        best = ck.get("best", float("inf"))
        start_epoch = ck.get("epoch", 0)
        print(f"resumed from {args.resume} @ epoch {start_epoch} (best val {best:.4f})", flush=True)
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device, mmap=True)
        core.load_state_dict(ck["model"])
        print(f"warm-started weights from {args.init_from} "
              f"(fresh optimizer/epochs; keeping its latent stats)", flush=True)

    precision = "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32")
    print(f"train={len(train_ds)} val={len(val_ds)} | device={device} | precision={precision} | "
          f"batch={args.batch_size} x accum {accum} = eff {args.batch_size * accum} | "
          f"{n_batches} batches/epoch", flush=True)

    def eval_ema():
        if ema is None:
            return evaluate(model, val_dl, device)
        backup = {k: v.detach().clone() for k, v in core.state_dict().items()}
        core.load_state_dict(ema.weights(core.state_dict()))
        v = evaluate(model, val_dl, device)
        core.load_state_dict(backup)
        return v

    def save_state(val_loss, next_epoch):
        nonlocal best, stale
        last_ckpt = {"cfg": asdict(cfg), "model": core.state_dict(), "opt": opt.state_dict(),
                     "scheduler": scheduler.state_dict(), "epoch": next_epoch,
                     "best": min(best, val_loss), "step": global_step}
        if ema is not None:
            last_ckpt["ema"] = ema.state_dict()
        torch.save(last_ckpt, out / "lfm2_last.pt")
        if val_loss < best:
            best, stale = val_loss, 0
            infer_model = ema.weights(core.state_dict()) if ema is not None else core.state_dict()
            torch.save({"cfg": asdict(cfg), "model": infer_model}, out / "lfm2_best.pt")
            print(f"  saved best (val {val_loss:.4f}{' EMA' if ema else ''}) -> {out / 'lfm2_best.pt'}",
                  flush=True)
        else:
            stale += 1
    stop = False
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        model.train()
        running, window, t0, last = 0.0, 0.0, time.monotonic(), time.monotonic()
        opt.zero_grad()
        opt_steps = 0
        for i, batch in enumerate(train_dl):
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out_losses = model.loss(batch["text_tokens"], batch["text_lens"],
                                        batch["latents"].to(device), batch["lat_lens"])
            scaler.scale(out_losses["total"] / accum).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(core.parameters(), 1.0)
                prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= prev_scale:  # optimizer actually stepped (no fp16 overflow)
                    scheduler.step()
                    if ema is not None:
                        ema.update(core)
                opt.zero_grad()
                opt_steps += 1
                global_step += 1
                if args.save_every and opt_steps % args.save_every == 0:
                    val_loss = eval_ema()
                    print(f"  [mid-epoch] step {global_step} | val {val_loss:.4f} | "
                          f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
                    save_state(val_loss, epoch)
                    model.train()
                    if args.patience and stale >= args.patience:
                        stop = True
                        break
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

        if stop:
            print(f"early stop: no val improvement in {args.patience} checks "
                  f"(best {best:.4f})", flush=True)
            continue
        train_loss = running / max(1, n_batches)
        val_loss = eval_ema()
        dt = time.monotonic() - t0
        print(f"epoch {epoch + 1}/{args.epochs} done ({dt:.0f}s) | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
        save_state(val_loss, epoch + 1)
        if args.patience and stale >= args.patience:
            stop = True
            print(f"early stop: no val improvement in {args.patience} checks "
                  f"(best {best:.4f})", flush=True)


if __name__ == "__main__":
    main()

"""Train PocketLFM on LJSpeech latents.

Tuned for a small (4 GB) GPU:
  * Mixed precision (fp16 autocast + GradScaler) — halves activation memory.
  * Small per-step batch with gradient accumulation for a larger effective batch.
  * The frozen Mimi codec is NOT loaded here — training consumes cached latents produced
    by `data_ljspeech.py`, so only the ~100M model + optimizer live on the GPU.
    (100M params fp32 ~0.4 GB, AdamW state ~0.8 GB, grads ~0.4 GB ≈ 1.6 GB + activations.)

Usage:
    python data_ljspeech.py --download --out data/ljspeech_cache    # one-time
    python train.py --cache data/ljspeech_cache --out runs/exp1

Quick self-check without any data:
    python train.py --smoke
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from pocketlfm import PocketLFM, PocketLFMConfig


@torch.no_grad()
def evaluate(model, loader, device, use_amp, amp_dtype) -> float:
    """Mean total loss over a validation loader."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        tt = batch["text_tokens"].to(device, non_blocking=True)
        tl = batch["text_lens"].to(device, non_blocking=True)
        lat = batch["latents"].to(device, non_blocking=True)
        ll = batch["lat_lens"].to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            losses = model.compute_losses(tt, tl, lat, ll)
        total += float(losses["total"])
        n += 1
    model.train()
    return total / max(1, n)


def build_optimizer(model: torch.nn.Module, lr: float, wd: float,
                    fused: bool = False) -> torch.optim.Optimizer:
    """AdamW with weight decay only on 2D+ tensors (not norms/biases/embeddings scale)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=fused)


def lr_at(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_ckpt(path, model, optim, scaler, cfg, step, epoch, best_loss) -> None:
    """Save full training state so a run can resume seamlessly."""
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": cfg.__dict__,
            "step": step,
            "epoch": epoch,
            "best_loss": best_loss,
        },
        path,
    )


class _SmokeDataset(Dataset):
    """Tiny synthetic dataset (random tokens + latents) to exercise the training loop."""

    def __init__(self, n: int, cfg: PocketLFMConfig):
        self.n, self.cfg = n, cfg

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        lt = int(torch.randint(5, 20, (1,)))
        t = int(torch.randint(8, 40, (1,)))
        tokens = torch.randint(0, self.cfg.text_vocab, (lt,))
        latents = torch.randn(t, self.cfg.latent_dim)
        return tokens, latents


def train(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device == "cuda") and not args.no_amp
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = PocketLFMConfig()
    model = PocketLFM(cfg).to(device)
    print(f"PocketLFM: {model.num_parameters() / 1e6:.2f}M params | device={device} | "
          f"amp={use_amp}({amp_dtype if use_amp else '-'})")
    if args.compile:
        model.compute_losses = torch.compile(model.compute_losses, dynamic=True)

    # --- data ---
    preload = not args.no_preload
    if args.smoke:
        from data_ljspeech import collate

        dataset: Dataset = _SmokeDataset(args.smoke_size, cfg)
        preload = False
    else:
        from data_ljspeech import LJSpeechLatents, collate

        dataset = LJSpeechLatents(args.cache, max_frames=args.max_frames, preload=preload)

    # Deterministic train/val split (seeded so the held-out set is stable across runs).
    val_n = int(round(len(dataset) * args.val_split))
    if val_n > 0:
        train_ds, val_ds = random_split(
            dataset, [len(dataset) - val_n, val_n],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_ds, val_ds = dataset, None
    print(f"dataset: {len(dataset)} clips ({len(train_ds)} train / "
          f"{len(val_ds) if val_ds else 0} val)")

    # Preloaded data lives in RAM, so workers (and their IPC) only add overhead.
    workers = 0 if preload else args.workers
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=(workers > 0),
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate,
            pin_memory=(device == "cuda"),
            drop_last=False,
            persistent_workers=(workers > 0),
        )

    optim = build_optimizer(model, args.lr, args.weight_decay, fused=(device == "cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the training length: --epochs (a full pass over the data) takes precedence
    # over --steps. The LR cosine schedule spans the resolved total step count.
    steps_per_epoch = max(1, len(loader))
    if args.epochs is not None:
        total_steps = args.epochs * steps_per_epoch
        n_epochs = args.epochs
    else:
        total_steps = args.steps
        n_epochs = math.ceil(total_steps / steps_per_epoch)
    print(f"training for {total_steps} steps "
          f"(~{n_epochs} epochs x {steps_per_epoch} batches/epoch)")

    # Resume from a checkpoint if requested. "auto" picks pocketlfm_last.pt in --out.
    step = 0
    start_epoch = 0
    best_loss = float("inf")
    resume_path = None
    if args.resume == "auto":
        cand = out_dir / "pocketlfm_last.pt"
        resume_path = cand if cand.exists() else None
    elif args.resume:
        resume_path = Path(args.resume)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optim" in ckpt:
            optim.load_state_dict(ckpt["optim"])
        if "scaler" in ckpt and ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0)  # epoch field is the count already completed
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"resumed from {resume_path} at epoch {start_epoch} / step {step} "
              f"(best_loss {best_loss:.4f})")

    model.train()
    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    done = False

    for epoch in range(start_epoch, n_epochs):
        epoch_loss, epoch_n = 0.0, 0
        for batch in loader:
            tt = batch["text_tokens"].to(device, non_blocking=True)
            tl = batch["text_lens"].to(device, non_blocking=True)
            lat = batch["latents"].to(device, non_blocking=True)
            ll = batch["lat_lens"].to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                losses = model.compute_losses(tt, tl, lat, ll)
                loss = losses["total"] / args.grad_accum

            scaler.scale(loss).backward()
            epoch_loss += float(losses["total"].detach())
            epoch_n += 1

            if (step + 1) % args.grad_accum == 0:
                for g in optim.param_groups:
                    g["lr"] = lr_at(step // args.grad_accum, args.warmup, total_steps,
                                    args.lr, args.min_lr)
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            if step % args.log_every == 0:
                lr_now = optim.param_groups[0]["lr"]
                sps = (step + 1) / (time.time() - t0)
                d = {k: float(v.detach()) for k, v in losses.items()}
                print(f"step {step:6d} | total {d['total']:.4f} "
                      f"| mtp {d['mtp']:.4f} | eos {d['eos']:.4f} "
                      f"| lr {lr_now:.2e} | {sps:.1f} it/s")

            step += 1
            if step % args.ckpt_every == 0 and not args.smoke:
                ckpt = out_dir / f"pocketlfm_step{step}.pt"
                save_ckpt(ckpt, model, optim, scaler, cfg, step, epoch, best_loss)
                print(f"saved {ckpt}")
            if step >= total_steps:
                done = True
                break

        train_mean = epoch_loss / max(1, epoch_n)

        # Validate (every --val-every epochs, and always on the final epoch). The best
        # checkpoint is selected on validation loss when a val split exists, else on the
        # mean training loss.
        run_val = val_loader is not None and (
            (epoch + 1) % args.val_every == 0 or epoch + 1 == n_epochs or done
        )
        val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype) if run_val else None
        metric = val_loss if val_loss is not None else train_mean
        metric_name = "val" if val_loss is not None else "train"

        msg = f"epoch {epoch + 1}/{n_epochs} complete ({step} steps) | train {train_mean:.4f}"
        if val_loss is not None:
            msg += f" | val {val_loss:.4f}"
        print(msg)

        if not args.smoke:
            # Update best first so "last" records the current best_loss for clean resume.
            if metric < best_loss:
                best_loss = metric
                save_ckpt(out_dir / "pocketlfm_best.pt", model, optim, scaler, cfg,
                          step, epoch + 1, best_loss)
                print(f"  new best {metric_name} loss {metric:.4f} -> saved pocketlfm_best.pt")
            # "last" carries full state for seamless resume.
            save_ckpt(out_dir / "pocketlfm_last.pt", model, optim, scaler, cfg,
                      step, epoch + 1, best_loss)
        if done:
            break

    if not args.smoke:
        save_ckpt(out_dir / "pocketlfm_final.pt", model, optim, scaler, cfg, step, n_epochs,
                  best_loss)
        print(f"saved {out_dir / 'pocketlfm_final.pt'}")
    print(f"done: {step} steps in {time.time() - t0:.1f}s")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train PocketLFM on LJSpeech.")
    ap.add_argument("--cache", default="data/ljspeech_cache", help="preprocessed latent cache dir")
    ap.add_argument("--out", default="runs/exp1", help="checkpoint output dir")
    ap.add_argument("--resume", default=None,
                    help="resume from a checkpoint path, or 'auto' for <out>/pocketlfm_last.pt")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    # 4 GB-friendly defaults
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=600, help="truncate long clips (memory cap)")
    ap.add_argument("--val-split", type=float, default=0.02,
                    help="fraction of clips held out for validation (0 disables)")
    ap.add_argument("--val-every", type=int, default=1, help="validate every N epochs")
    ap.add_argument("--epochs", type=int, default=None,
                    help="number of full passes over the dataset (overrides --steps)")
    ap.add_argument("--steps", type=int, default=20000,
                    help="total training steps; ignored if --epochs is set")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--bf16", action="store_true", help="use bf16 autocast (if GPU supports it)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-preload", action="store_true",
                    help="stream the cache from disk instead of preloading it into RAM")
    ap.add_argument("--compile", action="store_true", help="torch.compile the loss (dynamic shapes)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    # smoke test
    ap.add_argument("--smoke", action="store_true", help="run a few steps on synthetic data")
    ap.add_argument("--smoke-size", type=int, default=64)
    return ap


if __name__ == "__main__":
    parser = build_argparser()
    cli = parser.parse_args()
    if cli.smoke:
        cli.steps = min(cli.steps, 12)
        cli.warmup = 2
        cli.log_every = 1
    train(cli)

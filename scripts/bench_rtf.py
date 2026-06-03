from __future__ import annotations

import argparse
import statistics
import time

import torch

TEXTS = {
    "short": "Hello there.",
    "medium": "The quick brown fox jumps over the lazy dog and then trots away.",
    "long": (
        "In a distant valley where the rivers run clear and the mountains touch the sky, "
        "an old storyteller gathered the village children to recount the legends of their "
        "ancestors, of brave journeys, of kindness rewarded, and of the quiet courage it "
        "takes to do what is right when no one is watching."
    ),
}


def stats(values):
    return statistics.median(values), min(values), max(values)


class OriginalRunner:
    name = "PocketTTS (orig)"

    def __init__(self, device, threads):
        from pocket_tts.default_parameters import get_default_voice_for_language
        from pocket_tts.models.tts_model import TTSModel
        from pocket_tts.modules.stateful_module import init_states

        self.init_states = init_states
        self.tts = TTSModel.load_model(language="english").to(device)
        torch.set_num_threads(threads)
        self.voice = self.tts.get_state_for_audio_prompt(
            get_default_voice_for_language("english")
        )
        self.sr = self.tts.sample_rate
        self.device = device

    def ar_params(self):
        return sum(p.numel() for p in self.tts.flow_lm.parameters())

    def backbone_params(self):
        return sum(p.numel() for p in self.tts.flow_lm.transformer.parameters())

    def generate_seconds(self, text):
        audio = self.tts.generate_audio(self.voice, text, copy_state=True)
        return audio.shape[-1] / self.sr

    def time_to_first_audio(self, text):
        t0 = time.time()
        for _ in self.tts.generate_audio_stream(
            model_state=self.voice, text_to_generate=text, copy_state=True
        ):
            return time.time() - t0
        return time.time() - t0

    def backbone_seconds_per_64(self, threads):
        torch.set_num_threads(threads)
        bb = self.tts.flow_lm.transformer
        st = self.init_states(self.tts.flow_lm, batch_size=1, sequence_length=4096)
        bb(torch.randn(1, 8, 1024, device=self.device), st)
        for _ in range(3):
            bb(torch.randn(1, 1, 1024, device=self.device), st)
        samples = []
        for _ in range(3):
            t0 = time.time()
            for _ in range(64):
                bb(torch.randn(1, 1, 1024, device=self.device), st)
            samples.append(time.time() - t0)
        return statistics.median(samples)


class PocketLFMRunner:
    def __init__(self, device, threads, cfg):
        from pocketlfm import PocketLFMPipeline

        self.pipe = PocketLFMPipeline.from_pretrained(cfg=cfg).to(device)
        torch.set_num_threads(threads)
        self.k = cfg.mtp_horizon
        self.name = f"PocketLFM (K={self.k})"
        self.sr = self.pipe.sample_rate
        self.fr = self.pipe.frame_rate
        self.device = device

    def ar_params(self):
        return self.pipe.model.num_parameters()

    def backbone_params(self):
        return sum(p.numel() for p in self.pipe.model.backbone.parameters())

    def generate_for_frames(self, n_frames):
        secs = n_frames / self.fr
        wav = self.pipe.generate(TEXTS["medium"], max_seconds=secs, flow_steps=1,
                                 eos_threshold=float("inf"))
        if self.device == "cuda":
            torch.cuda.synchronize()
        return wav.shape[-1] / self.sr

    def generate_text_to(self, text, n_frames):
        secs = n_frames / self.fr
        wav = self.pipe.generate(text, max_seconds=secs, flow_steps=1, eos_threshold=float("inf"))
        if self.device == "cuda":
            torch.cuda.synchronize()
        return wav.shape[-1] / self.sr

    def time_to_first_audio(self, text):
        t0 = time.time()
        for _ in self.pipe.generate_stream(text, max_seconds=5.0, flow_steps=1,
                                           eos_threshold=float("inf"), decode_every=1):
            return time.time() - t0
        return time.time() - t0

    def backbone_seconds_per_64(self, threads):
        torch.set_num_threads(threads)
        bb = self.pipe.model.backbone
        st = bb.init_state()
        bb(torch.randn(1, 8, 1024, device=self.device), st)
        for _ in range(3):
            bb(torch.randn(1, self.k, 1024, device=self.device), st)
        samples = []
        for _ in range(3):
            t0 = time.time()
            for _ in range(64 // self.k):
                bb(torch.randn(1, self.k, 1024, device=self.device), st)
            samples.append(time.time() - t0)
        return statistics.median(samples)


def timed_rtf(seconds_fn, arg, trials):
    seconds_fn(arg)
    rtfs, audios = [], []
    for _ in range(trials):
        t0 = time.time()
        secs = seconds_fn(arg)
        rtfs.append(secs / (time.time() - t0))
        audios.append(secs)
    return rtfs, statistics.median(audios)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    print(f"device={args.device} threads={args.threads} trials={args.trials}\n")

    from pocketlfm import PocketLFMConfig

    orig = OriginalRunner(args.device, args.threads)
    lfm = PocketLFMRunner(args.device, args.threads, PocketLFMConfig())

    print("=== parameters ===")
    print(f"{orig.name:18s} backbone {orig.backbone_params()/1e6:6.1f}M  "
          f"AR {orig.ar_params()/1e6:6.1f}M")
    print(f"{lfm.name:18s} backbone {lfm.backbone_params()/1e6:6.1f}M  "
          f"AR {lfm.ar_params()/1e6:6.1f}M\n")

    targets = {}
    print("=== end-to-end RTF by utterance length (median [min-max]) ===")
    print(f"{'text':8s} {'audio_s':>8s} {'orig RTF':>20s} {'PocketLFM RTF':>22s} {'speedup':>8s}")
    summary = []
    for key, text in TEXTS.items():
        o_rtfs, o_audio = timed_rtf(orig.generate_seconds, text, args.trials)
        n_frames = round(o_audio * lfm.fr)
        targets[key] = n_frames
        l_rtfs, _ = timed_rtf(lambda _t: lfm.generate_text_to(text, n_frames), text, args.trials)
        om, olo, ohi = stats(o_rtfs)
        lm, llo, lhi = stats(l_rtfs)
        summary.append((key, om, lm))
        print(f"{key:8s} {o_audio:8.2f} {om:7.2f}x [{olo:.1f}-{ohi:.1f}]   "
              f"{lm:8.2f}x [{llo:.1f}-{lhi:.1f}]   {lm/om:6.2f}x")

    lfm.pipe.quantize()
    print(f"\n=== PocketLFM int8 (dynamic quant) ===")
    int8_summary = []
    for key, text in TEXTS.items():
        rtfs, _ = timed_rtf(lambda _t: lfm.generate_text_to(text, targets[key]), text, args.trials)
        m, lo, hi = stats(rtfs)
        om = dict((k, o) for k, o, _ in summary)[key]
        int8_summary.append((key, om, m))
        print(f"{key:8s} {m:8.2f}x [{lo:.1f}-{hi:.1f}]   {m/om:6.2f}x vs orig")

    print("\n=== time-to-first-audio (streaming latency, ms) ===")
    for runner in (orig, lfm):
        runner.time_to_first_audio(TEXTS["medium"])
        ttfa = statistics.median(
            [runner.time_to_first_audio(TEXTS["medium"]) * 1000 for _ in range(args.trials)]
        )
        label = runner.name + (" int8" if runner is lfm else "")
        print(f"{label:18s} {ttfa:7.1f} ms")

    print("\n=== backbone-only cost for 64 frames (~5.1s audio) ===")
    ob = orig.backbone_seconds_per_64(args.threads)
    lb = lfm.backbone_seconds_per_64(args.threads)
    print(f"{orig.name:18s} {ob:6.3f}s  (64 passes x 1 frame)")
    print(f"{lfm.name:18s} {lb:6.3f}s  ({64//lfm.k} passes x {lfm.k} frames, MTP)  "
          f"-> {ob/lb:.2f}x faster")

    avg = statistics.mean(lm / om for _, om, lm in summary)
    print(f"\nPocketLFM averages {avg:.2f}x faster end-to-end than PocketTTS "
          f"across {len(summary)} lengths ({args.device}, {args.threads} threads).")
    print("PocketLFM weights are random: RTF measures architecture+MTP, not quality.")


if __name__ == "__main__":
    main()

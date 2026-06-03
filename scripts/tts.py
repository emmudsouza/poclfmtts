"""Text-to-speech with LFM2-Audio.

Synthesizes speech for a piece of text using the LiquidAI/LFM2.5-Audio-1.5B
model in its sequential TTS mode (the model reads the provided text aloud,
rather than holding a conversation).

Examples:
    uv run python tts.py "Hello, world."
    uv run python tts.py "Good evening." --voice "US female" --out hello.wav
    uv run python tts.py --text-file script.txt --device cpu

Notes:
    * On GPU this uses the higher-quality LFM2 audio detokenizer
      (`processor.decode`). That detokenizer is hardcoded to CUDA in the
      package, so on CPU we fall back to the Mimi codec (`mimi.decode`).
    * The 1.5B model on CPU works but is slow (well below real time); fine for
      offline synthesis, just not interactive.
"""

import argparse
import sys

import torch
import torchaudio

from liquid_audio import ChatState, LFM2AudioModel, LFM2AudioProcessor

HF_DIR = "LiquidAI/LFM2.5-Audio-1.5B"
VOICES = ["UK male", "US male", "UK female", "US female"]
SAMPLE_RATE = 24_000


def build_models(device: str, dtype: torch.dtype):
    print(f"Loading processor on {device} ...", file=sys.stderr)
    proc = LFM2AudioProcessor.from_pretrained(HF_DIR, device=device).eval()
    print(f"Loading model on {device} ...", file=sys.stderr)
    model = LFM2AudioModel.from_pretrained(
        HF_DIR, device=device, dtype=dtype).eval()
    return proc, model


def synthesize(
    text: str,
    *,
    proc: LFM2AudioProcessor,
    model: LFM2AudioModel,
    voice: str,
    device: str,
    max_new_tokens: int,
    audio_temperature: float,
    audio_top_k: int,
) -> torch.Tensor:
    """Run TTS and return a (1, T) 24 kHz waveform tensor."""
    chat = ChatState(proc)

    chat.new_turn("system")
    chat.add_text(f"Perform TTS. Use the {voice} voice.")
    chat.end_turn()

    chat.new_turn("user")
    chat.add_text(text)
    chat.end_turn()

    chat.new_turn("assistant")

    audio_out: list[torch.Tensor] = []
    with torch.no_grad():
        for t in model.generate_sequential(
            **chat,
            max_new_tokens=max_new_tokens,
            audio_temperature=audio_temperature,
            audio_top_k=audio_top_k,
        ):
            if t.numel() == 1:  # a text token the model emits alongside speech
                print(proc.text.decode(t), end="", flush=True, file=sys.stderr)
            else:  # an 8-codebook audio frame
                audio_out.append(t)

    print(file=sys.stderr)
    if not audio_out:
        raise RuntimeError("Model produced no audio frames.")

    # Drop the trailing end-of-audio (2048) frame, shape -> (1, 8, T)
    audio_codes = torch.stack(audio_out[:-1], 1).unsqueeze(0)

    if device == "cuda":
        # Higher-quality LFM2 detokenizer (CUDA-only in this package).
        waveform = proc.decode(audio_codes)
    else:
        # Mimi codec respects the active device, so it works on CPU.
        with proc.mimi.streaming(1):
            waveform = proc.mimi.decode(audio_codes)[0]

    return waveform.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="LFM2-Audio text-to-speech")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("text", nargs="?", help="Text to speak")
    src.add_argument("--text-file", help="Read the text to speak from a file")
    parser.add_argument("--out", default="tts.wav",
                        help="Output WAV path (default: tts.wav)")
    parser.add_argument("--voice", default="UK female",
                        choices=VOICES, help="Speaker voice")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda if available, else cpu)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--audio-temperature", type=float, default=0.8)
    parser.add_argument("--audio-top-k", type=int, default=64)
    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file) as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        text = "What is this obsession people have with books? They put them in their houses like they're trophies."
        print(
            f"No text given; using sample text:\n  {text}\n", file=sys.stderr)

    # bfloat16 is fine on GPU; use float32 on CPU (faster and broadly supported).
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    proc, model = build_models(args.device, dtype)

    waveform = synthesize(
        text,
        proc=proc,
        model=model,
        voice=args.voice,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        audio_temperature=args.audio_temperature,
        audio_top_k=args.audio_top_k,
    )

    torchaudio.save(args.out, waveform, SAMPLE_RATE)
    print(
        f"Wrote {args.out} ({waveform.shape[-1] / SAMPLE_RATE:.1f}s @ {SAMPLE_RATE} Hz)", file=sys.stderr)


if __name__ == "__main__":
    main()

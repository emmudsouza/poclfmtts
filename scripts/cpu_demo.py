"""Run the LFM2-Audio speech-to-speech chat demo on CPU.

The packaged `liquid-audio-demo` command hardcodes CUDA and loads the model to
GPU at import time, which OOMs on small / shared GPUs. This launcher rebuilds
the exact same Gradio app but pins every tensor to CPU.

Run with:  uv run python cpu_demo.py

NOTE: this is a ~1.5B speech-to-speech model. On CPU it runs, but generation is
far slower than real time, so the live voice chat will feel very laggy. It's
useful for smoke-testing / development, not for a smooth conversation.
"""

from queue import Queue
from threading import Thread

import gradio as gr
import numpy as np
import torch
from fastrtc import AdditionalOutputs, ReplyOnPause, WebRTC

from liquid_audio import (
    ChatState,
    LFM2AudioModel,
    LFM2AudioProcessor,
    LFMModality,
)

DEVICE = "cpu"
HF_DIR = "LiquidAI/LFM2.5-Audio-1.5B"

# float32 is generally faster than bfloat16 on CPU (uses ~2x RAM, ~6GB for 1.5B
# params). Switch to torch.bfloat16 if you are RAM-constrained instead.
DTYPE = torch.float32

print(f"Loading processor on {DEVICE}")
proc = LFM2AudioProcessor.from_pretrained(HF_DIR, device=DEVICE).eval()
print(f"Loading model on {DEVICE}")
lfm2_audio = LFM2AudioModel.from_pretrained(HF_DIR, device=DEVICE, dtype=DTYPE).eval()
print("Loading tokenizer")
mimi = proc.mimi.eval()


def chat_producer(
    q: "Queue[torch.Tensor | None]",
    chat: ChatState,
    temp: float | None,
    topk: int | None,
):
    print(f"Starting generation with state {chat}.")
    with torch.no_grad(), mimi.streaming(1):
        for t in lfm2_audio.generate_interleaved(
            **chat,
            max_new_tokens=1024,
            audio_temperature=temp,
            audio_top_k=topk,
        ):
            q.put(t)

            if t.numel() > 1:
                if (t == 2048).any():
                    continue

                wav_chunk = mimi.decode(t[None, :, None])[0]
                q.put(wav_chunk)

    q.put(None)


def chat_response(
    audio: tuple[int, np.ndarray],
    _id: str,
    chat: ChatState,
    temp: float | None = 1.0,
    topk: int | None = 4,
):
    if temp == 0:
        temp = None
    if topk == 0:
        topk = None

    if temp is not None:
        temp = float(temp)
    if topk is not None:
        topk = int(topk)

    if len(chat.text) == 1:
        chat.new_turn("system")
        chat.add_text("Respond with interleaved text and audio.")
        chat.end_turn()

        chat.new_turn("user")

    rate, wav = audio
    chat.add_audio(torch.tensor(wav / 32_768, dtype=torch.float), rate)
    chat.end_turn()

    chat.new_turn("assistant")

    q: "Queue[torch.Tensor | None]" = Queue()
    chat_thread = Thread(target=chat_producer, args=(q, chat, temp, topk))
    chat_thread.start()

    out_text: list[torch.Tensor] = []
    out_audio: list[torch.Tensor] = []
    out_modality: list[LFMModality] = []

    while True:
        t = q.get()
        if t is None:
            break
        elif t.numel() == 1:  # text
            out_text.append(t)
            out_modality.append(LFMModality.TEXT)
            print(proc.text.decode(t), end="")
            cur_string = proc.text.decode(torch.cat(out_text)).removesuffix("<|text_end|>")
            yield AdditionalOutputs(cur_string)
        elif t.numel() == 8:
            out_audio.append(t)
            out_modality.append(LFMModality.AUDIO_OUT)
        elif t.numel() == 1920:
            np_chunk = (t.cpu().numpy() * 32_767).astype(np.int16)
            yield (24_000, np_chunk)
        else:
            raise RuntimeError(f"unexpected shape: {t.shape}")

    chat.append(
        text=torch.stack(out_text, 1),
        audio_out=torch.stack(out_audio, 1),
        modality_flag=torch.tensor(out_modality, device=DEVICE),
    )

    chat.end_turn()
    chat.new_turn("user")


def clear():
    gr.Info("Cleared chat history", duration=3)
    return ChatState(proc), None


with gr.Blocks() as demo:
    gr.Markdown("# LFM2-Audio speech-to-speech chat (CPU)")

    chat_state = gr.State(ChatState(proc))
    webrtc = WebRTC(
        modality="audio",
        mode="send-receive",
        full_screen=False,
    )
    text_out = gr.Textbox(
        lines=4,
        label="Output",
    )
    clear_btn = gr.Button("Reset chat")

    webrtc.stream(
        ReplyOnPause(
            chat_response,  # type: ignore[arg-type]
            input_sample_rate=24_000,
            output_sample_rate=24_000,
            can_interrupt=False,
        ),
        inputs=[webrtc, chat_state],
        outputs=[webrtc],
    )
    webrtc.on_additional_outputs(
        lambda s: s,
        outputs=[text_out],
    )
    clear_btn.click(clear, outputs=[chat_state, text_out])


def main():
    demo.launch()


if __name__ == "__main__":
    main()

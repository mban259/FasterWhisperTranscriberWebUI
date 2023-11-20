from faster_whisper import WhisperModel
import gradio as gr
import numpy as np
from typing import Tuple, List, Final
from scipy import signal
from scipy.io import wavfile
import argparse
import copy
import json
import os


MODEL_SIZE: Final[str] = "large-v2"
WHISPER_SAMPLING_RATE: Final[int] = 16000

model: WhisperModel = WhisperModel(
    MODEL_SIZE, device="cuda", compute_type="int8_float32")


class StateClass:
    textList: List[str]
    buffer: np.ndarray[np.float32]

    def __init__(self) -> None:
        self.textList = []
        self.buffer = np.array((0,), dtype=np.float32)
        self.is_speech = False
        self.last_transcribe = 0
        pass

    def to_str(self) -> str:
        if self.textList:
            return "\n".join(self.textList)
        else:
            return ""


def is_speech(wave: np.ndarray[np.float32], threshold: np.float32):
    return np.max(np.abs(wave)) > threshold


def transcribe(state: StateClass, mic: Tuple[int, np.ndarray], volume: float, threshold: float, language: str) -> Tuple[StateClass, str, bool]:
    sampleRate, rawWave = mic

    # ステレオならモノラルに変換
    if len(rawWave.shape) == 2:
        wave = rawWave.mean(axis=1)
    else:
        wave = rawWave.astype(np.float64)

    # 16khzにダウンサンプル
    wave: np.ndarray[np.float64] = signal.resample(
        wave, int(wave.shape[0] * WHISPER_SAMPLING_RATE / sampleRate))

    # 音量調節
    wave *= volume / 32768

    wave = wave.astype(np.float32)

    flag_speech = is_speech(wave, threshold)

    if language == "ja":
        lang = "ja"
    else:
        lang = None

    if not state.is_speech:
        if flag_speech:
            state.is_speech = True
            state.buffer = wave
        res = state.to_str()
    else:

        if flag_speech:
            state.buffer = np.concatenate([state.buffer, wave])
            if state.buffer.shape[0] - state.last_transcribe > 2 * WHISPER_SAMPLING_RATE:
                segments, info = model.transcribe(
                    state.buffer, word_timestamps=True, language=lang)
                copy_text_list = copy.copy(state.textList)
                for seg in segments:
                    copy_text_list.append("{}".format(seg.text))
                res = "" if not copy_text_list else "\n".join(copy_text_list)
                state.last_transcribe = state.buffer.shape[0]
            else:
                res = state.to_str()
        else:
            segments, info = model.transcribe(
                state.buffer, word_timestamps=True, language=lang)
            state.is_speech = False
            for seg in segments:
                state.textList.append("{}".format(seg.text))
            res = state.to_str()

    return state, res, flag_speech


def reset(state: StateClass) -> Tuple[StateClass, str]:
    state.textList = []
    return state, ""


def save(threshold: float, volume: float, language: str):
    js = {}
    js["threshold"] = threshold
    js["volume"] = volume
    js["language"] = language

    with open("setting.json", mode="wt") as f:
        json.dump(js, f)
    pass


if os.path.isfile("setting.json"):
    with open("setting.json") as f:
        js = json.load(f)
        val_threshold = js["threshold"]
        val_volume = js["volume"]
        val_language = js["language"] 
else:
    val_threshold = 0.1
    val_volume = 0.8
    val_language = "None"

with gr.Blocks() as demo:
    state = gr.State(value=StateClass())
    with gr.Tab("main"):
        with gr.Row():
            with gr.Column():
                mic = gr.Audio(sources="microphone")
                speaking = gr.Checkbox(label="speaking")
                reset_button = gr.Button("reset")
            output = gr.Textbox(value="", label="output")
    with gr.Tab("settings"):
        threshold = gr.Slider(minimum=0.01, maximum=1.0,
                              value=val_threshold, label="threshold")
        volume = gr.Slider(minimum=0.01, maximum=1.0,
                           value=val_volume, label="volume")
        language = gr.Dropdown(
            choices=["None", "ja"], label="language", value=val_language, filterable=False)
        save_button = gr.Button("save")

    reset_button.click(
        fn=reset,
        inputs=[state],
        outputs=[state, output]
    )

    mic.stream(
        fn=transcribe,
        inputs=[state, mic, volume, threshold, language],
        outputs=[state, output, speaking],
        show_progress=False
    )

    mic.stop_recording(
        cancels=[transcribe]
    )

    save_button.click(
        save,
        inputs=[threshold, volume, language]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--cert", type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    if args.listen:
        if (args.cert == None) or (args.key == None):
            print("cert:{}, key:{}".format(args.cert, args.key))
            exit()
        demo.launch(server_name="0.0.0.0",
                    ssl_certfile=args.cert, ssl_keyfile=args.key,
                    ssl_verify=False, share=args.share, server_port=args.port)
    else:
        demo.launch(share=args.share, server_port=args.port)

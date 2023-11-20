from faster_whisper import WhisperModel
import gradio as gr
import numpy as np
from typing import Tuple, List, Final
from scipy import signal
from scipy.io import wavfile
import argparse
import copy


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


def transcribe(state: StateClass, mic: Tuple[int, np.ndarray], volume: float, threshold: float) -> Tuple[StateClass, str, bool]:
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

    if not state.is_speech:
        if flag_speech:
            state.is_speech = True
            state.buffer = wave
        res = state.to_str()
    else:
        
        if flag_speech:
            state.buffer = np.concatenate([state.buffer, wave])
            if state.buffer.shape[0] - state.last_transcribe > 2 * WHISPER_SAMPLING_RATE:
                segments, info = model.transcribe(state.buffer, word_timestamps=True)
                copy_text_list = copy.copy(state.textList)
                for seg in segments:
                    copy_text_list.append("{}".format(seg.text))
                res = "" if not copy_text_list else "\n".join(copy_text_list)
                state.last_transcribe = state.buffer.shape[0]
            else:
                res = state.to_str()
        else:
            segments, info = model.transcribe(state.buffer, word_timestamps=True)
            state.is_speech = False
            for seg in segments:
                state.textList.append("{}".format(seg.text))
            res = state.to_str()

    return state, res, flag_speech


def reset(state: StateClass) -> Tuple[StateClass, str]:
    state.textList = []
    return state, ""


with gr.Blocks() as demo:
    state = gr.State(value=StateClass())
    mic = gr.Audio(sources="microphone")
    output = gr.Textbox(value="", label="output")

    with gr.Row():
        # inference_len = gr.Number(value=20)
        threshold = gr.Slider(minimum=0.01, maximum=1.0,
                              value=0.1, label="threshold")
        volume = gr.Slider(minimum=0.01, maximum=1.0,
                           value=0.8, label="volume")

    speaking = gr.Checkbox(label="speaking")
    reset_button = gr.Button("reset")

    reset_button.click(
        fn=reset,
        inputs=[state],
        outputs=[state, output]
    )

    mic.stream(
        fn=transcribe,
        inputs=[state, mic, volume, threshold],
        outputs=[state, output, speaking],
        show_progress=False
    )

    mic.stop_recording(
        cancels=[transcribe]
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

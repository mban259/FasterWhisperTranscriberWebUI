from faster_whisper import WhisperModel
import numpy as np
from typing import Tuple, List, Final
from scipy import signal
import copy
import state_class


MODEL_SIZE: Final[str] = "large-v2"
WHISPER_SAMPLING_RATE: Final[int] = 16000

model: WhisperModel = WhisperModel(
    MODEL_SIZE, device="cuda", compute_type="int8_float32")


def is_speech(wave: np.ndarray[np.float32], threshold: np.float32) -> bool:
    return np.max(np.abs(wave)) > threshold


def transcribe(state: state_class.StateClass, mic: Tuple[int, np.ndarray], volume: float, threshold: float, language: str) -> Tuple[state_class.StateClass, str, bool]:
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
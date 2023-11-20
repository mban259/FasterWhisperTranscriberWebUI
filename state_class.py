import numpy as np
from typing import Tuple, List, Final


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


def reset(state: StateClass) -> Tuple[StateClass, str]:
    state.textList = []
    return state, ""

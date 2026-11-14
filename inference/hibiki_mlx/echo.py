"""The echo canceller: in mic mode the speakers play the translation and the mic hears it, so the
model would translate its own voice back, and that translation again, without end; the voice gate
passes it, being speech. WebRTC's echo canceller (AEC3, in livekit's audio processing module)
removes from what the mic hears what the speakers were given, before the level stage raises it.
main.py's mic mode uses it; file mode plays nothing while it listens.

The canceller finds the echo's delay itself (~280 ms from the MacBook's speakers to its mic in
main.py's duplex stream), so it takes no delay hint.
"""

from __future__ import annotations

import numpy as np
from livekit import rtc

FRAME = 1920  # samples at 24 kHz: one Mimi frame, 80 ms
RATE = 24000
CHUNK = RATE // 100  # the module's frame: 10 ms


def _int16(pcm: np.ndarray) -> rtc.AudioFrame:
    data = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    return rtc.AudioFrame(data.tobytes(), RATE, 1, CHUNK)


class EchoCanceller:
    """One session's canceller: call it with each 80 ms frame the mic heard and the frame the
    speakers were given in the same callback, in order; it returns the heard frame without the
    speakers' echo."""

    def __init__(self) -> None:
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True)

    def __call__(self, heard: np.ndarray, played: np.ndarray) -> np.ndarray:
        if heard.shape != (FRAME,) or played.shape != (FRAME,):
            raise ValueError(f"the echo canceller takes {FRAME}-sample frames, not {heard.shape}, {played.shape}")
        out = np.empty(FRAME, np.float32)
        for i in range(0, FRAME, CHUNK):
            self.apm.process_reverse_stream(_int16(played[i : i + CHUNK]))
            frame = _int16(heard[i : i + CHUNK])
            self.apm.process_stream(frame)
            out[i : i + CHUNK] = np.frombuffer(frame.data, np.int16) / 32767
        return out

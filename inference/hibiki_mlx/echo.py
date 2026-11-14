"""The echo canceller: in mic mode the speakers play the translation and the mic hears it, so the
model would translate its own voice back, and that translation again, without end; the voice gate
passes it, being speech. WebRTC's echo canceller (AEC3, in livekit's audio processing module)
removes from what the mic hears what the speakers were given, before the level stage raises it.
main.py's mic mode uses it; file mode plays nothing while it listens.

The canceller finds the echo's delay itself, so it takes no hint, but only up to ~450 ms: the
MacBook Pro's speakers reach its mic 132 ms late in main.py's duplex stream; a Bluetooth or AirPlay
speaker can add enough to put its echo out of reach.
"""

from __future__ import annotations

import numpy as np
from livekit import rtc

RATE = 24000
CHUNK = RATE // 100  # the module's frame: 10 ms


def _int16(pcm: np.ndarray) -> rtc.AudioFrame:
    data = (np.clip(pcm, -1, 1) * 32767).astype(np.int16)
    return rtc.AudioFrame(data.tobytes(), RATE, 1, CHUNK)


class EchoCanceller:
    """One session's canceller: call it with each block the mic heard (a whole number of 10 ms
    chunks) and the block the speakers were given in the same callback, in order; it returns the
    heard block without the speakers' echo."""

    def __init__(self) -> None:
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True)

    def __call__(self, heard: np.ndarray, played: np.ndarray) -> np.ndarray:
        if heard.shape != played.shape or heard.ndim != 1 or heard.shape[0] % CHUNK:
            raise ValueError(f"the echo canceller takes blocks of whole {CHUNK}-sample chunks, "
                             f"not {heard.shape} heard, {played.shape} played")
        out = np.empty(heard.shape[0], np.float32)
        for i in range(0, heard.shape[0], CHUNK):
            self.apm.process_reverse_stream(_int16(played[i : i + CHUNK]))
            frame = _int16(heard[i : i + CHUNK])
            self.apm.process_stream(frame)
            out[i : i + CHUNK] = np.frombuffer(frame.data, np.int16) / 32767
        return out

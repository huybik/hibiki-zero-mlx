"""The input level stage: raises quiet source audio toward the level Hibiki trained on, before
Mimi encodes it. Mimi codes quiet speech differently from the same speech loud (a clip peaking at
−46 dBFS keeps 4% of the semantic codes it has at −6 dBFS), so quiet input was translated worse,
or not at all (docs/agent-memory/sft-evaluation.md#input-level).

Each 80 ms frame is scaled by one gain, set from a peak envelope that rises at once to a louder
frame and falls 6 dB a second after it:

    envelope = max(frame peak, previous envelope × RELEASE)
    gain     = TARGET_PEAK / envelope, between 1 and MAX_GAIN

Audio peaking above TARGET_PEAK passes unchanged: only quieter input is raised, and a raised
frame never peaks above TARGET_PEAK, so nothing clips. Silence and pauses longer than about
3 s are raised too, up to MAX_GAIN, as a stream's start is: the gain cannot tell quiet speech
from a quiet room. It is causal and frame by frame, so a stream gives the same samples in any
whole-frame chunks. The MLX runtime and the PyTorch evaluation use this module; the Swift
runtime mirrors it (`Leveller` in Audio.swift).
"""

from __future__ import annotations

import numpy as np

FRAME = 1920  # samples at 24 kHz: one Mimi frame, 80 ms
FRAME_RATE = 12.5
# Training sources peak at a median −2 to −5 dBFS, their noisy copies up to 15 dB lower
# (augment.GAIN_DB). The two-way model translated FLEURS clips peaking above −20 dBFS as well
# as louder ones, and those under −30 worse; under −35 some came out empty.
TARGET_PEAK = 10 ** (-20 / 20)
MAX_GAIN = 10 ** (30 / 20)  # speech peaking at −50 dBFS reaches the target
RELEASE = 10 ** (-6 / 20 / FRAME_RATE)  # per frame: the envelope falls 6 dB a second


class Leveller:
    """One stream's level stage: call it on the stream's samples (1-D, 24 kHz) in order, in
    chunks of whole frames (the last chunk may end mid-frame); a new stream takes a new one."""

    def __init__(self) -> None:
        self.envelope = 0.0

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        out = np.empty_like(pcm)
        for start in range(0, pcm.shape[-1], FRAME):
            frame = pcm[start : start + FRAME]
            self.envelope = max(float(np.abs(frame).max()), self.envelope * RELEASE)
            if self.envelope * MAX_GAIN <= TARGET_PEAK:
                gain = MAX_GAIN
            else:
                gain = max(1.0, TARGET_PEAK / self.envelope)
            out[start : start + FRAME] = frame * np.float32(gain) if gain != 1.0 else frame
        return out

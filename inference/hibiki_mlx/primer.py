"""Priming the direction: a stream that starts with a short sentence in the language spoken, its
output dropped, keeps that direction (main.py's --speaking, in both modes)."""

import mlx.core as mx
import numpy as np
import rustymimi
import sphn

from hibiki_mlx.level import FRAME
from hibiki_mlx.lm.generate import LmGen
from hibiki_mlx.model import ASSETS, Model

# One short sentence in each source language (FLEURS dev, CC BY 4.0), fed first and unheard when
# the user says which language they speak: the two-way model guesses the direction from the voice
# and gets some voices wrong, but a training stream never switches direction, so it keeps the one
# the primer sets (docs/agent-memory/two-way.md#next). A stopgap until the direction is a trained
# input. main.py seeds the sampler before every run, so priming comes out the same each time: each
# primer was chosen to settle at that seed with its translation complete. One that does not, or
# ends in "he said", runs into every clip's translation.
PRIMERS = {"vi": ASSETS / "primer_vi.wav", "en": ASSETS / "primer_en.wav"}
# After the primer, silence until the model has finished translating it: this long without a new
# word, at most PRIMER_MAX_GAP_S. A fixed 2 s left primer words in the speech's translation.
PRIMER_SETTLE_S = 1.5
PRIMER_MAX_GAP_S = 4.0


def prime(model: Model, gen: LmGen, mimi_enc: rustymimi.Tokenizer, language: str) -> None:
    """Steps the primer of `language` (a PRIMERS key) through `gen` and `mimi_enc` at the start of
    a stream, then silence until the model has written no word for PRIMER_SETTLE_S (at most
    PRIMER_MAX_GAP_S). Their output is dropped. The primers are heard as recorded, without the
    level stage: they peak at its target already, and it would raise all that follows their
    loudest frame."""
    pcm, _ = sphn.read(str(PRIMERS[language]), sample_rate=24000)
    pcm = np.pad(pcm[0], (0, -pcm.shape[-1] % FRAME))
    pad_token = model.tokenizer.pad_id()

    def step(frame: np.ndarray) -> int:
        codes = mimi_enc.encode_step(frame[None, None, :])
        oat = np.transpose(codes, (0, 2, 1))[0, :, : model.config.other_codebooks]
        return gen.step(mx.array(oat)).item()

    for frame in pcm.reshape(-1, FRAME):
        step(frame)
    silence = np.zeros(FRAME, np.float32)
    settle, gap, quiet = round(PRIMER_SETTLE_S * 12.5), 0, 0
    while quiet < settle and gap < round(PRIMER_MAX_GAP_S * 12.5):
        quiet = quiet + 1 if step(silence) <= pad_token else 0
        gap += 1


def primer_steps(language: str | None) -> int:
    """The most steps `prime` can run for `language`, none without one."""
    if language is None:
        return 0
    pcm, _ = sphn.read(str(PRIMERS[language]), sample_rate=24000)
    return -(-pcm.shape[-1] // FRAME) + round(PRIMER_MAX_GAP_S * 12.5)

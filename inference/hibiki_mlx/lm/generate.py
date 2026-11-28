# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in this directory.

from typing import Optional

import mlx.core as mx

from . import sampling
from .lm import Lm


class LmGen:
    """One stream's delayed tokens: step t feeds the model the previous text token and each
    codebook's token of frame t - 1 - delay, and stores the codebooks it samples at their frame."""

    # Fills the frames not generated yet.
    ungenerated_token = -2

    def __init__(
        self,
        model: Lm,
        max_steps: int,
        text_sampler: sampling.Sampler,
        audio_sampler: sampling.Sampler,
    ):
        self.model: Lm = model
        self.text_sampler = text_sampler
        self.audio_sampler = audio_sampler
        self.max_steps = max_steps
        self.num_codebooks = 1 + model.cfg.audio_codebooks
        self.gen_sequence = mx.full(
            shape=(1, self.num_codebooks, max_steps),
            vals=self.ungenerated_token,
            dtype=mx.int32,
        )
        self.step_idx = 0
        self.audio_padding_token = self.model.cfg.audio_padding_token
        self.audio_delays = self.model.cfg.audio_delays
        self.max_delay = max(self.audio_delays)
        self.main_codebooks = self.model.cfg.generated_codebooks

    def step(self, other_audio_tokens: mx.array) -> mx.array:
        """Feeds the source codebooks of this frame and returns the sampled text token [1, 1]."""
        if self.step_idx >= self.max_steps:
            raise ValueError(f"reached max-steps {self.max_steps}")
        if self.step_idx == 0:
            text_tokens = mx.full(
                shape=(1, 1),
                vals=self.model.cfg.text_out_vocab_size,
                dtype=mx.int32,
            )
        else:
            text_tokens = self.gen_sequence[:, 0, self.step_idx - 1 : self.step_idx]
        self.gen_sequence[:, 1 + self.main_codebooks :, self.step_idx] = other_audio_tokens
        audio_tokens = []
        for cb_idx, delay in enumerate(self.audio_delays):
            gen_idx = self.step_idx - 1 - delay
            if gen_idx >= 0:
                audio_token = self.gen_sequence[:, cb_idx + 1, gen_idx][:, None]
            else:
                audio_token = mx.full((1, 1), self.audio_padding_token, dtype=mx.int32)
            audio_tokens.append(audio_token)
        text_tokens, audio_tokens = self.model.sample(
            text_tokens,
            audio_tokens,
            self.text_sampler,
            self.audio_sampler,
        )

        assert audio_tokens.shape[-2] == self.model.cfg.generated_codebooks, (
            "invalid output audio-token shape"
        )
        self.gen_sequence[:, 0, self.step_idx] = text_tokens.squeeze(-1)
        for cb_idx, delay in enumerate(self.audio_delays[: self.main_codebooks]):
            gen_idx = self.step_idx - delay
            if gen_idx >= 0:
                self.gen_sequence[:, cb_idx + 1, gen_idx] = audio_tokens[:, cb_idx, 0]
        self.step_idx += 1
        return text_tokens

    def last_audio_tokens(self) -> Optional[mx.array]:
        gen_idx = self.step_idx - 1 - self.max_delay
        if gen_idx < 0:
            return None
        tokens = self.gen_sequence[:, 1 : 1 + self.main_codebooks, gen_idx]

        if (tokens == self.audio_padding_token).any():  # type: ignore
            return None
        if (tokens == self.ungenerated_token).any():  # type: ignore
            raise ValueError(f"ungenerated value in last-audio tokens {self.step_idx}")
        return tokens

"""Parallel codebook heads for a phone student ([phone-student](../../docs/agent-memory/phone-student.md#parallel-heads),
plan stage 5.4): k linear heads on the main transformer's output sample a step's k codebooks in one
pass, where the depformer runs k sequential steps. Codebook j is delayed j frames (MusicGen's
pattern, carried by moshi's own `delays`), so a head predicting codebook j at a step has read codebook
j-1 of the same frame in the step's input. The heads also read the step's text token, as the
depformer's first step does: the model's own text embedding of it, added to the main transformer's
output (text is sampled before audio at each step, so this costs no latency and no parameters).
The heads sit in the model's `depformer` slot, so moshi's
training forward and its generation (`LMGen`, which samples audio only when a depformer exists) run
them through two replaced methods; the depformer's own tensors are dropped.

A config marks such a model with `"audio_heads": "parallel"`, which `common.load_checkpoint_info`
takes out before moshi reads the config (moshi hands every key to the model).
"""

from __future__ import annotations

import types
from typing import Any

import torch
from moshi.models.lm import LMGen, LMModel
from moshi.utils.sampling import sample_token

CONFIG_KEY = "audio_heads"
KIND = "parallel"
# The depformer's tensors, which a heads model drops (its `depformer` slot then holds the heads).
DEPFORMER_PARTS = ("depformer_in", "depformer_emb", "depformer_text_emb", "depformer_norms", "linears")


def delays(codebooks: int) -> list[int]:
    """A heads model's target delays: codebook j delayed j frames."""
    return list(range(codebooks))


class ParallelHeads(torch.nn.Module):
    """One linear head a codebook on the main transformer's (normed) output, from scratch."""

    def __init__(self, dim: int, card: int, codebooks: int):
        super().__init__()
        self.heads = torch.nn.ModuleList(torch.nn.Linear(dim, card, bias=False) for _ in range(codebooks))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """[B, T, dim] (the main transformer's output plus its text token's embedding) -> logits
        [B, codebooks, T, card]."""
        return torch.stack([head(context) for head in self.heads], 1)


def is_heads(lm: Any) -> bool:
    return isinstance(getattr(lm, "depformer", None), ParallelHeads)


def install(lm: LMModel) -> LMModel:
    """`lm` (built by moshi from the config) with ParallelHeads in its depformer slot, initialized
    afresh on its device and dtype, the depformer's tensors dropped."""
    weight = lm.text_linear.weight
    lm.depformer = ParallelHeads(lm.dim, lm.card, lm.dep_q).to(device=weight.device, dtype=weight.dtype)
    lm.depformer.train(lm.training)
    for name in DEPFORMER_PARTS:
        setattr(lm, name, None)
    lm.forward_depformer_training = types.MethodType(_forward_training, lm)
    return lm


def _forward_training(self: LMModel, sequence: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
    """LMModel.forward_depformer_training for heads: logits [B, dep_q, T, card] on the delayed
    timeline, as the depformer's are (moshi's forward undelays them). `sequence[:, 0]` is each step's
    text token, teacher-forced; the audio codes reached the heads through the main transformer."""
    return self.depformer(transformer_out + self.text_emb(sequence[:, 0]))


_depformer_step = LMGen.depformer_step


def depformer_step(self: LMGen, text_token: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
    """LMGen.depformer_step: a heads model samples its dep_q codebooks at once ([B, dep_q]);
    any other model runs moshi's depformer steps."""
    if not is_heads(self.lm_model):
        return _depformer_step(self, text_token, transformer_out)
    if self.cfg_coef != 1.0:
        raise NotImplementedError("classifier-free guidance with parallel heads")
    lm = self.lm_model
    logits = lm.depformer(transformer_out + lm.text_emb(text_token[:, None]))[:, :, 0]  # [B, dep_q, card]
    return sample_token(logits.float(), self.use_sampling, self.temp, self.top_k)


LMGen.depformer_step = depformer_step

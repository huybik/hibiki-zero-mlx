"""The teacher-forced losses: Kyutai's moshi-finetune weighting (`LossWeights`) as weighted sums
of one kernel, each token's cross-entropy (`token_ce`) or, distilling, its KL from a teacher
(`token_kl`), for training, teacher-forced validation and GRPO's logprobs. The chunked
distillation that skips the full logits is `distill.py`'s.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from moshi.run_inference import get_condition_tensors
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class LossWeights:
    """Kyutai's moshi-finetune weighting: audio codebook 0 (semantic) counts `first_codebook`
    times an acoustic codebook, AR acoustic codebooks 1..ar_end-1 count `acoustic_ar` times
    each parallel acoustic codebook ar_end..dep_q-1, and a text PAD `text_pad` times a word
    token, each within its own weighted mean; the audio and text means are then added with
    weights `audio` and `text`."""

    audio: float = 1.0
    text: float = 1.0
    first_codebook: float = 100.0
    text_pad: float = 0.5
    acoustic_ar: float = 1.0


# Each component's loss sum and the token count or weight it is divided by.
LOSS_SUMS = ("audio", "semantic", "acoustic", "acoustic_ar", "acoustic_par", "text", "content", "pad")


def audio_ar_end(lm: Any) -> int:
    """Exclusive upper codebook index of autoregressive acoustic codebooks (e.g. 8 for 16-cb
    slices with parallel_codebooks=8, meaning cb 1..7 are AR and cb 8..15 are parallel). Other
    models split their logged acoustic loss at codebook 8; `--acoustic-ar-weight` is slices-only."""
    dep_q = int(lm.dep_q)
    parallel = getattr(lm, "parallel_codebooks", None)
    if parallel is not None:
        return max(1, dep_q - int(parallel))
    return min(dep_q, 8)


def audio_codebook_weights(lm: Any, weights: LossWeights, device: torch.device) -> torch.Tensor:
    """Per-codebook audio loss weights `[1, dep_q, 1]`."""
    dep_q = int(lm.dep_q)
    ar_end = audio_ar_end(lm)
    cb_w = [1.0] * dep_q
    cb_w[0] = float(weights.first_codebook)
    for cb in range(1, ar_end):
        cb_w[cb] = float(weights.acoustic_ar)
    return torch.tensor(cb_w, device=device, dtype=torch.float32).view(1, dep_q, 1)


def batch_condition_tensors(lm: Any, model_type: str, batch_size: int) -> Any | None:
    if lm.fuser is None:
        return None
    return get_condition_tensors(model_type, lm, batch_size=batch_size, cfg_coef=1.0)


def loss_means(sums: dict[str, Any], weights: LossWeights) -> dict[str, Any]:
    """Losses from `<name>_sum` / `<name>_n`, as tensors in training or floats in evaluation."""

    def mean(name: str) -> Any:
        total, count = sums[f"{name}_sum"], sums[f"{name}_n"]
        if isinstance(count, torch.Tensor):
            return total / count.clamp(min=1)
        return total / count if count else 0.0

    audio_loss = mean("audio")
    text_loss = mean("text")
    return {
        "loss": weights.audio * audio_loss + weights.text * text_loss,
        "audio_loss": audio_loss,
        "semantic_loss": mean("semantic"),
        "acoustic_loss": mean("acoustic"),
        "acoustic_ar_loss": mean("acoustic_ar"),
        "acoustic_par_loss": mean("acoustic_par"),
        "text_loss": text_loss,
        "content_text_loss": mean("content"),
        "pad_text_loss": mean("pad"),
    }


@torch.compile(dynamic=True)
def token_ce(logits: Any, targets: Any, valid: Any, temperature: float = 1.0) -> Any:
    """Each target token's cross-entropy under softmax(logits / temperature), float32 in the
    shape of `targets`, and 0 where not `valid` (moshi fills the logits there with NaN, or the
    target is -1). The one loss kernel: SFT's losses are its weighted sums, and GRPO's token
    logprobs its negative at the sampling temperature.

    Dense on purpose. The fp32 scores live only inside the fused kernel, so no full-vocabulary
    fp32 copy is kept for the backward pass, and no boolean index gathers the valid positions
    (a host sync, and 4.5% of an SFT step's kernels on the H200). No gradient reaches the model
    from a masked position: `torch.where` hands its CE a 0, and where the logits are moshi's NaN
    fill the resulting 0 x NaN stops at the fill (`_undelay_sequence` overwrites those slices in
    place, whose backward zeroes them). The trainers still check every gradient norm."""
    scores = logits.float() if temperature == 1.0 else logits.float() / temperature
    ce = torch.nn.functional.cross_entropy(
        scores.flatten(0, -2), torch.where(valid, targets, 0).flatten(), reduction="none"
    ).view(targets.shape)
    return torch.where(valid, ce, 0.0)


@torch.compile(dynamic=True)
def token_kl(logits: Any, teacher_logits: Any, valid: Any) -> Any:
    """KL(teacher || model) of each valid position's next-token distribution, float32 in the
    shape of `valid`, and 0 elsewhere: distillation's per-token loss in place of `token_ce`'s.
    Both logits are read only where `valid` (moshi's NaN fill elsewhere), so no NaN reaches the
    loss or the gradient through either. With the arguments swapped, `token_kl(teacher, model)`
    is the reverse KL(model || teacher), on-policy distillation's
    (`grpo_objective.distill_loss`)."""
    keep = valid.unsqueeze(-1)
    teacher_log = torch.log_softmax(torch.where(keep, teacher_logits.float(), 0.0), dim=-1)
    log = torch.log_softmax(torch.where(keep, logits.float(), 0.0), dim=-1)
    kl = (teacher_log.exp() * (teacher_log - log)).sum(-1)
    return torch.where(valid, kl, 0.0)


def compute_batch_losses(
    lm: Any,
    codes: Any,
    condition_tensors: Any | None,
    weights: LossWeights,
    accuracy: bool = False,
    teacher: Any | None = None,
) -> dict[str, Any]:
    """Teacher-forced losses of one batch, as tensors on the GPU (no host sync), with every
    component's sum and count beside them so evaluation can aggregate exactly. Positions holding
    -1 are outside the masks. `accuracy` adds the counts of text tokens the argmax gets right
    (`content_correct`, `pad_correct`). With `teacher` (a teacher's output on the same frames,
    for the same codebooks), each token's loss is its KL from the teacher's distribution
    (`token_kl`) instead of its cross-entropy, with the same weights, at the positions both score:
    moshi returns each model's logits on the stream's frames, and a model with other delays (a
    heads student's) has other last frames to score."""
    output = lm(codes, condition_tensors=condition_tensors)
    audio_valid = output.mask  # [B, dep_q, T]
    text_valid = output.text_mask  # [B, 1, T]
    if teacher is not None:
        audio_valid = audio_valid & teacher.mask
        text_valid = text_valid & teacher.text_mask
    audio_targets = codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q]
    if teacher is None:
        audio_token_loss = token_ce(output.logits, audio_targets, audio_valid)
    else:
        audio_token_loss = token_kl(output.logits, teacher.logits, audio_valid)
    ar_end = audio_ar_end(lm)
    semantic = torch.zeros_like(audio_valid)
    semantic[:, 0] = audio_valid[:, 0]
    acoustic = audio_valid & ~semantic
    acoustic_ar = torch.zeros_like(audio_valid)
    if ar_end > 1:
        acoustic_ar[:, 1:ar_end] = audio_valid[:, 1:ar_end]
    acoustic_par = acoustic & ~acoustic_ar
    audio_weight = audio_codebook_weights(lm, weights, audio_valid.device) * audio_valid
    text_targets = codes[:, :1]
    if teacher is None:
        text_token_loss = token_ce(output.text_logits, text_targets, text_valid)
    else:
        text_token_loss = token_kl(output.text_logits, teacher.text_logits, text_valid)
    pad = (text_targets == lm.text_padding_token_id) & text_valid
    content = text_valid & ~pad
    text_weight = torch.where(pad, weights.text_pad, 1.0) * text_valid
    sums = {
        "audio_sum": (audio_token_loss * audio_weight).sum(),
        "audio_n": audio_weight.sum(),
        "semantic_sum": (audio_token_loss * semantic).sum(),
        "semantic_n": semantic.sum(),
        "acoustic_sum": (audio_token_loss * acoustic).sum(),
        "acoustic_n": acoustic.sum(),
        "acoustic_ar_sum": (audio_token_loss * acoustic_ar).sum(),
        "acoustic_ar_n": acoustic_ar.sum(),
        "acoustic_par_sum": (audio_token_loss * acoustic_par).sum(),
        "acoustic_par_n": acoustic_par.sum(),
        "text_sum": (text_token_loss * text_weight).sum(),
        "text_n": text_weight.sum(),
        "content_sum": (text_token_loss * content).sum(),
        "content_n": content.sum(),
        "pad_sum": (text_token_loss * pad).sum(),
        "pad_n": pad.sum(),
    }
    result = {**sums, **loss_means(sums, weights)}
    if accuracy:
        correct = output.text_logits.argmax(-1) == text_targets
        result["content_correct"] = (correct & content).sum()
        result["pad_correct"] = (correct & pad).sum()
    return result


def evaluate_teacher_forced(
    lm: Any,
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    weights: LossWeights,
    max_batches: int = 0,
) -> dict[str, float | int]:
    """Corpus-level teacher-forced losses and text accuracies over `dataloader`, under the bf16
    autocast that training computes its losses in (1.7x faster than fp32 weights without it)."""
    was_training = bool(lm.training)
    lm.eval()
    condition_cache: dict[int, Any | None] = {}
    keys = [f"{name}_{part}" for name in LOSS_SUMS for part in ("sum", "n")]
    keys += ["content_correct", "pad_correct"]
    totals = dict.fromkeys(keys, 0.0)
    batches = streams_seen = max_frames = 0
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        for batch_index, batch in enumerate(dataloader):
            if max_batches and batch_index >= max_batches:
                break
            codes = batch["codes"].to(device=device, dtype=torch.long)
            batch_size = int(codes.shape[0])
            if batch_size not in condition_cache:
                condition_cache[batch_size] = batch_condition_tensors(lm, model_type, batch_size)
            result = compute_batch_losses(
                lm, codes, condition_cache[batch_size], weights, accuracy=True
            )
            for key in keys:
                totals[key] += float(result[key])
            batches += 1
            streams_seen += batch_size
            max_frames = max(max_frames, int(batch["frames"].max()))
    if was_training:
        lm.train()
    if not batches:
        raise RuntimeError("Validation dataloader produced no batches.")
    content_tokens = int(totals["content_n"])
    pad_tokens = int(totals["pad_n"])
    return {
        **loss_means(totals, weights),
        "audio_tokens": int(totals["semantic_n"] + totals["acoustic_n"]),
        "content_tokens": content_tokens,
        "content_acc": totals["content_correct"] / content_tokens if content_tokens else 0.0,
        "pad_tokens": pad_tokens,
        "pad_acc": totals["pad_correct"] / pad_tokens if pad_tokens else 0.0,
        "batches": batches,
        "streams": streams_seen,
        "max_frames": max_frames,
    }

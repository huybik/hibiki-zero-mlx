"""GRPO's update (`grpo.py`, step 3): the policy's logprobs on its rollouts, the clipped
objective and, with a teacher, the on-policy distillation term, the tokens the loss keeps and the
forward length of each micro-batch; the loss terms' options, which the profiler shares.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from training.finetune import streams
from training.finetune.losses import batch_condition_tensors, token_ce, token_kl
from training.finetune.model import teacher_checkpoint_info


def token_logprobs(
    lm: Any, codes: torch.Tensor, condition_tensors: Any | None, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`output_logprobs` of a forward of `lm` on `codes`."""
    return output_logprobs(lm, lm(codes, condition_tensors=condition_tensors), codes, temperature)


def output_logprobs(
    lm: Any, output: Any, codes: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logprobs at `temperature` of the text and target audio tokens of `codes` (the cache
    layout [B, 1 + 2 * dep_q, T]) under `output`, `lm`'s forward on them: [B, 1 + dep_q, T]
    float32 (0 where undefined), and where they are defined. They are SFT's loss kernel negated
    (`losses.token_ce`)."""
    audio = codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q]
    text_ce = token_ce(output.text_logits, codes[:, :1], output.text_mask, temperature)
    audio_ce = token_ce(output.logits, audio, output.mask, temperature)
    return -torch.cat([text_ce, audio_ce], dim=1), torch.cat([output.text_mask, output.mask], dim=1)


def grpo_loss(
    lm: Any,
    output: Any,
    codes: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    keep: torch.Tensor,
    stream_weights: torch.Tensor,
    epsilon: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The negated clipped objective summed over a micro-batch's kept tokens (Eq. 7-8), with the
    sums behind its ratio statistics. `output` is `lm`'s forward on `codes`, `advantages`
    [B, T], `keep` [B, 1 + dep_q, T] (from `kept_tokens`), `stream_weights` [1 + dep_q]."""
    logprobs, mask = output_logprobs(lm, output, codes, temperature)
    mask = mask & keep
    ratio = torch.exp(logprobs - old_logprobs)
    advantage = advantages[:, None, :]
    objective = torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)
    objective = (objective * stream_weights[None, :, None] * mask).sum()
    kept = ratio.detach() * mask  # masked reductions: boolean indexing would sync the host
    deviation = (kept - mask.float()).abs()
    stats = {
        "ratio_sum": kept.sum(),
        "clipped": (deviation > epsilon).sum(),
        "tokens": mask.sum(),
        "max_deviation": deviation.max(),
    }
    return -objective, stats


def distill_loss(
    output: Any, teacher: Any, keep: torch.Tensor, stream_weights: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The reverse KL(policy || teacher) of each kept token's next-token distribution (at
    temperature 1, as SFT's distillation), weighted by `stream_weights` [1 + dep_q] and summed
    over a micro-batch, where both forwards (`output` the policy's, `teacher` the teacher's on the
    same rollouts) score the token: their delays may differ, as a heads student's do. On the
    policy's own samples it is mode-seeking: the policy is pulled off whatever the teacher would
    not say, where SFT's KL(teacher || policy) spreads it over all the teacher might."""
    text_valid = output.text_mask & teacher.text_mask & keep[:, :1]
    audio_valid = output.mask & teacher.mask & keep[:, 1:]
    text_kl = token_kl(teacher.text_logits, output.text_logits, text_valid)
    audio_kl = token_kl(teacher.logits, output.logits, audio_valid)
    kl = torch.cat([text_kl, audio_kl], dim=1) * stream_weights[None, :, None]
    stats = {
        "kl_text_sum": text_kl.sum().detach(),
        "kl_text_tokens": text_valid.sum(),
        "kl_audio_sum": audio_kl.sum().detach(),
        "kl_audio_tokens": audio_valid.sum(),
    }
    return kl.sum(), stats


@dataclass(frozen=True)
class Objective:
    """An update's loss terms: GRPO's clipped objective times `reward_coef` and, with a teacher,
    the reverse KL to it times `kl_coef`, each token weighted by `stream_weights`."""

    stream_weights: torch.Tensor
    epsilon: float
    temperature: float
    reward_coef: float
    kl_coef: float


def update_loss(
    lm: Any,
    codes: torch.Tensor,
    condition_tensors: Any | None,
    keep: torch.Tensor,
    objective: Objective,
    old_logprobs: torch.Tensor | None,
    advantages: torch.Tensor,
    teacher_output: Any | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """A micro-batch's loss and statistics: one forward of the policy serves both terms. The
    GRPO term needs π_old's logprobs (`old_logprobs`) when `objective.reward_coef` is set, the
    distillation term the teacher's forward on the same `codes` (`teacher_output`)."""
    output = lm(codes, condition_tensors=condition_tensors)
    loss = torch.zeros((), device=codes.device)
    stats: dict[str, torch.Tensor] = {}
    if objective.reward_coef:
        policy, stats = grpo_loss(lm, output, codes, old_logprobs, advantages, keep,
                                  objective.stream_weights, objective.epsilon, objective.temperature)
        loss = loss + objective.reward_coef * policy
    if teacher_output is not None:
        kl, kl_stats = distill_loss(output, teacher_output, keep, objective.stream_weights)
        loss = loss + objective.kl_coef * kl
        stats.update(kl_stats)
    return loss, stats


def kept_tokens(
    rollouts: torch.Tensor, lengths: torch.Tensor, delays: torch.Tensor, eos_id: int
) -> torch.Tensor:
    """Which tokens of rollouts [N, 1 + dep_q, T] the loss keeps.
    - Text through each rollout's first EOS and audio before it, as SFT streams end. No reward
      reads the rest. The sampled tokens stay in place as inputs: set to -1, they would change
      what the delayed codebooks of the two frames before the EOS were sampled from.
    - Only what a forward of the rollout's own length scores (`lengths` [N], `delays` [1 + dep_q]
      of the text and target codebooks): a codebook delayed d frames stops d frames early. So a
      rollout's tokens do not depend on the longer rollouts it shares a micro-batch with."""
    frames = rollouts.shape[-1]
    is_eos = rollouts[:, 0] == eos_id
    first = torch.where(is_eos.any(dim=1), is_eos.int().argmax(dim=1), frames)
    index = torch.arange(frames, device=rollouts.device)
    text = index[None] <= first[:, None]
    audio = (index[None] < first[:, None])[:, None].expand(-1, rollouts.shape[1] - 1, -1)
    scored = index[None, None] < (lengths[:, None] - delays[None])[:, :, None]
    return torch.cat([text[:, None], audio], dim=1) & scored


def forward_frames(keep: torch.Tensor, delays: torch.Tensor) -> torch.Tensor:
    """How many frames a training forward needs to score each rollout's kept tokens ([N], from
    `kept_tokens`' [N, 1 + dep_q, T]): a token of a codebook delayed d sits d positions after
    its frame. The model is causal, so a shorter forward scores the same tokens the same way."""
    index = torch.arange(1, keep.shape[-1] + 1, dtype=torch.int32, device=keep.device)
    last = (keep * index).amax(dim=-1)  # 1 + each stream's last kept frame, 0 for none
    return (last + delays[None] * (last > 0)).amax(dim=1).clamp(min=1)


def micro_frames(spans: torch.Tensor, start: int, end: int, limit: int) -> int:
    """The forward length of the micro-batch of rows `start`..`end`: its longest span
    (`forward_frames`) up to a whole FRAME_BUCKET, as SFT pads its batches, at most `limit`."""
    bucket = streams.FRAME_BUCKET
    return min(math.ceil(int(spans[start:end].max()) / bucket) * bucket, limit)


def micro_batches(
    window: int, per_update: int, size: int, update: int | None = None
) -> list[tuple[int, int]]:
    """Row ranges of the micro-batches of one update (`update`) or of every update in the
    window, cut within each update's rows."""
    updates = range(window) if update is None else [update]
    return [
        (start, min(start + size, (index + 1) * per_update))
        for index in updates
        for start in range(index * per_update, (index + 1) * per_update, size)
    ]


def add_objective_args(parser: argparse.ArgumentParser) -> None:
    """The loss terms' options, the profiler's too."""
    parser.add_argument(
        "--teacher-weight", type=Path,
        help="Also distil this frozen model on the policy's rollouts (reverse KL); it must read the policy's codebooks.",
    )
    parser.add_argument("--teacher-config", type=Path, help="The teacher's config; default the 3B's.")
    parser.add_argument("--kl-coef", type=float, help="The distillation term's weight; default 1 with a teacher.")
    parser.add_argument(
        "--reward-coef", type=float, default=1.0,
        help="The GRPO term's weight; 0, with a teacher, is on-policy distillation alone.",
    )


def validate_objective_args(args: argparse.Namespace) -> None:
    """The loss terms' options (`add_objective_args`), the profiler's too; --kl-coef becomes 1
    with a teacher when not given."""
    if args.teacher_weight is None:
        if args.teacher_config is not None or args.kl_coef is not None or not args.reward_coef:
            raise ValueError("--teacher-config, --kl-coef and --reward-coef 0 need --teacher-weight")
    elif args.kl_coef is None:
        args.kl_coef = 1.0
    for name in ("reward_coef", "kl_coef"):
        value = getattr(args, name)
        if value is not None and not (math.isfinite(value) and value >= 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and not negative")
    if args.teacher_weight is not None and not args.kl_coef:
        raise ValueError("--teacher-weight needs a positive --kl-coef")


def teacher_info(args: argparse.Namespace, codebooks: int) -> Any | None:
    """The checkpoint info of --teacher-weight, or None. The teacher scores the policy's own
    rollouts, which hold only the policy's `codebooks`, so it must read as many."""
    if args.teacher_weight is None:
        return None
    info = teacher_checkpoint_info(args)
    if int(info.lm_config["dep_q"]) != codebooks:
        raise ValueError(f"the teacher must read the policy's {codebooks} codebooks a stream")
    return info


def teacher_forward(teacher: Any, info: Any, codes: torch.Tensor) -> Any:
    """The teacher's training forward on a micro-batch's rollouts, without gradients."""
    condition = batch_condition_tensors(teacher, info.model_type, len(codes))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        return teacher(codes, condition_tensors=condition)


def scored_delays(lm: Any, teacher: Any | None) -> torch.Tensor:
    """The text and target codebooks' delays `kept_tokens` and `forward_frames` count by: the
    policy's, or with a teacher each the larger of the two, so a token is kept only where both
    score it and a micro-batch's longer rollouts change no kept token of either forward."""
    delays = torch.tensor(lm.delays[: 1 + lm.dep_q])
    if teacher is not None:
        delays = torch.maximum(delays, torch.tensor(teacher.delays[: 1 + lm.dep_q]))
    return delays

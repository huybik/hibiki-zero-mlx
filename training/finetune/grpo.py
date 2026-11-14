#!/usr/bin/env python
"""GRPO for the full model (Hibiki-Zero, arXiv 2602.11072v2, §3.3; plan in docs/grpo_plan.md).

Every --tau updates, π_old becomes the current policy:
1. a bf16 copy of it samples --group rollouts for each stream of the window's updates (the
   planned streams in a seeded order), each as long as its stream plus --tail-seconds, the tail
   evaluation generates past a source, and at most --frames. The window's streams are generated
   longest first in batches of --gen-frames (rollouts times the batch's longest), so a batch
   stops at its own longest rollout and shorter ones batch more rows. Meanwhile the fp32 policy
   and AdamW's states wait on the CPU, leaving the GPU's memory to the batches;
2. the rollouts get BLEU process rewards and group-normalized advantages (`rewards.py`) in CPU
   workers, while the GPU keeps generating;
3. the policy itself scores them, giving π_old's logprobs. It runs the update's own forward on
   the update's own micro-batches, gradients enabled and the result detached: a no-grad forward,
   or another batch shape, picks other bf16 kernels.
   The first update's ratio is then exactly 1 (MPS: 1.000000 against a no-grad pass's 1.0012,
   with 6 tokens clipped).
Each update then takes --streams streams' rollouts, in micro-batches, each forward as long as
its last kept token needs. Its loss is the clipped ratio π_θ/π_old of every text and target-audio
token times its frame's advantage, weighted --text-weight for text against 1 for each audio
codebook, summed over frames and averaged over rollouts.

Every --eval-every updates, and at the start, the run translates --eval-pairs free-running, as SFT
validation does. The first set of each direction selects (`common.selecting_sets`: FLEURS dev
both ways by default): the run promotes the checkpoint with the best mean chrF over them among
those that pass every set's health gates and end no more than LATENCY_SLACK_S later than the start
on each (its median end offset); val_log.jsonl records each evaluation's verdict.

Rollouts replay a cached stream's Vietnamese through `LMGen` and keep every sampled text and
audio token in the training frame layout, so the training forward scores them in exactly the
contexts they were sampled in. In that layout, `LMGen` reads the input given at step s as source
frame s - 1, samples text frame s at step s, and samples audio codebook k's frame s - delay_k.

Pairs, resume and logs are `checkpoints.py`'s, so `h100.sh grpo` and `hf_sync.py` run it as they
run SFT. Saves land on window boundaries: a resume starts a window and draws its rollouts again.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.finetune import common, rewards, streams
from training.finetune.checkpoints import (
    BEST_METRIC,
    CheckpointWriter,
    append_log,
    atomic_write_text,
    claim_run_config,
    jsonable,
    keep_log_lines,
    load_newest_pair,
    open_run,
    promote_best,
)
from training.finetune.hibiki_helpers import get_lmgen
from training.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    repo_display_path,
    require_dir,
    require_file,
    resolve_repo_path,
)

# A resumed run must match these run_config.json entries.
RESUME_KEYS = (
    "cache_dir", "config_path", "model_weight", "mimi_weight", "tokenizer", "max_frames",
    "stream_seconds", "frames", "tail_seconds", "streams", "group", "updates", "tau", "lr", "adam_beta1",
    "adam_beta2", "weight_decay", "grad_clip", "epsilon", "alpha", "text_weight", "temperature",
    "top_k", "eval_pairs", "eval_batch_size", "eval_every", "save_every", "keep_checkpoints",
    "seed", "train_rows", "train_streams", "eval_rows",
)
EVAL_PAIRS = [  # the two-way model's (user, 2026-11-10: one GRPO, on both directions)
    Path("data/eval/fleurs-vi/dev.jsonl"),
    Path("data/eval/fleurs-en/dev.jsonl"),
    Path("data/eval/en-vi-val/cv27.jsonl"),
    Path("data/eval/en-vi-val/libriheavy.jsonl"),
    Path("data/eval/en-vi-val-noisy/cv27.jsonl"),
    Path("data/eval/en-vi-val-noisy/libriheavy.jsonl"),
    Path("data/eval/en-vi-val-en/cv27.jsonl"),
    Path("data/eval/en-vi-val-en/libriheavy.jsonl"),
]
LATENCY_SLACK_S = 0.2  # how far past the start's median end offset, on each selecting set, a checkpoint may promote
# The defaults the profiler measures with too.
TEMPERATURE = 0.8  # rollouts sample, and logprobs are taken, at evaluation's temperature
TOP_K = 250
EPSILON = 0.2  # the ratio clip
TEXT_WEIGHT = 100.0  # c_0: the text stream against 1 for each audio codebook


def stream_order(count: int, seed: int) -> np.ndarray:
    """The order updates take the planned streams in (`plan_streams` keeps a document's rows in
    consecutive streams, so in plan order a long recording would fill several updates)."""
    return np.random.default_rng(seed).permutation(count)


def rollout_frames(stream_frames: int, tail: int, limit: int) -> int:
    """A rollout's length: its stream and `tail` frames more, up to a whole FRAME_BUCKET, at most
    `limit`. Its last reward sits there, so text emitted later scores nothing, as evaluation stops
    generating a tail past the source. A stream ends at its last source or teacher target, so
    after a late-starting target the tail runs from that target's end."""
    bucket = streams.FRAME_BUCKET
    return min(math.ceil((stream_frames + tail) / bucket) * bucket, limit)


def rollout(lm_gen: Any, sources: torch.Tensor, silence: torch.Tensor) -> torch.Tensor:
    """Sample one rollout per row of `sources` (source codes [B, Q, T], training frames):
    text and audio tokens [B, 1 + dep_q, T] in the same frames."""
    lm = lm_gen.lm_model
    batch, _, frames = sources.shape
    audio_delays = lm.delays[1 : 1 + lm.dep_q]
    steps = frames + max(audio_delays)
    # Step s reads source frame s - 1 (step 0's input is never read); past the stream, silence.
    tail = silence[None, :, None].expand(batch, -1, steps - frames - 1)
    inputs = torch.cat([sources[:, :, :1], sources, tail], dim=-1)
    texts: list[torch.Tensor] = []
    audios: list[torch.Tensor] = []
    # CUDA graphs reuse their output buffers, so each step's tokens are copied out.
    lm_gen.on_text_hook = lambda tokens: texts.append(tokens.clone())
    lm_gen.on_audio_hook = lambda tokens: audios.append(tokens.clone())
    with torch.no_grad(), lm_gen.streaming(batch):
        for step in range(steps):
            lm_gen.step(inputs[:, :, step : step + 1])
    text = torch.stack(texts[:frames], dim=-1)  # [B, T]: text frame s is sampled at step s
    audio = torch.stack(audios, dim=-1)  # [B, dep_q, steps]: codebook k's frame s - delay_k
    audio = torch.stack(
        [audio[:, k, delay : delay + frames] for k, delay in enumerate(audio_delays)], dim=1
    )
    return torch.cat([text[:, None], audio], dim=1)


def token_logprobs(
    lm: Any, codes: torch.Tensor, condition_tensors: Any | None, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logprobs at `temperature` of the text and target audio tokens of `codes` (the cache
    layout [B, 1 + 2 * dep_q, T]), [B, 1 + dep_q, T] float32 (0 where undefined), and where they
    are defined. They are SFT's loss kernel negated (`common.token_ce`)."""
    output = lm(codes, condition_tensors=condition_tensors)
    audio = codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q]
    text_ce = common.token_ce(output.text_logits, codes[:, :1], output.text_mask, temperature)
    audio_ce = common.token_ce(output.logits, audio, output.mask, temperature)
    return -torch.cat([text_ce, audio_ce], dim=1), torch.cat([output.text_mask, output.mask], dim=1)


def grpo_loss(
    lm: Any,
    codes: torch.Tensor,
    condition_tensors: Any | None,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    keep: torch.Tensor,
    stream_weights: torch.Tensor,
    epsilon: float,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The negated clipped objective summed over a micro-batch's kept tokens (Eq. 7-8), with the
    sums behind its ratio statistics. `advantages` is [B, T], `keep` [B, 1 + dep_q, T] (from
    `kept_tokens`), `stream_weights` [1 + dep_q]."""
    logprobs, mask = token_logprobs(lm, codes, condition_tensors, temperature)
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


def stream_sentences(
    stream: streams.Stream, rows: list[streams.Row], tokenizer: Any
) -> list[rewards.Sentence]:
    """Each row of `stream` as a sentence: its Vietnamese span and its English reference, the
    words of its cache row's text stream (tokens above PAD, as evaluation decodes)."""
    pad_id = int(tokenizer.pad_id())
    sentences = []
    for piece in stream.pieces:
        row = rows[piece.row]
        tokens = row.codes[0, row.delay_frames : row.target_frames].tolist()
        words = [token for token in tokens if token > pad_id]
        sentences.append(
            rewards.Sentence(piece.source_start, piece.source_start + row.source_frames, tokenizer.decode(words))
        )
    return sentences


def stream_source(
    dataset: streams.StreamDataset, index: int, silence: torch.Tensor
) -> torch.Tensor:
    """A stream's Vietnamese codes [Q, T], silence past its end."""
    codes = dataset[index]["codes"]
    source = silence.long().clone()
    source[:, : codes.shape[1]] = codes[1 + dataset.layout.dep_q :]
    return source


_TOKENIZER: Any = None


def _start_reward_worker(tokenizer_path: str, trainer: int) -> None:
    global _TOKENIZER
    import ctypes
    import signal

    import sentencepiece

    # A forked worker holds the launcher's run lock too, and a pool never notices its parent
    # dying: without this, a killed trainer leaves workers that keep the lock and block every
    # relaunch. PR_SET_PDEATHSIG (1) kills the worker with the trainer.
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
    if os.getppid() != trainer:  # the trainer died before the call
        os._exit(1)
    _TOKENIZER = sentencepiece.SentencePieceProcessor(tokenizer_path)


def _group_rewards(
    tokens: np.ndarray, sentences: list[rewards.Sentence], alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return rewards.group_advantages(tokens, sentences, int(_TOKENIZER.eos_id()),
                                    int(_TOKENIZER.pad_id()), _TOKENIZER.decode, alpha)


def reward_pool(workers: int, tokenizer: Path) -> ProcessPoolExecutor:
    """Reward workers, forked now, before CUDA or any thread starts: from Python 3.11 a fork pool
    starts all its workers at the first submit (3.10 spawned them on demand, later)."""
    if sys.version_info < (3, 11):
        raise RuntimeError("GRPO's reward workers need Python 3.11 or newer")
    pool = ProcessPoolExecutor(
        workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=_start_reward_worker,
        initargs=(str(tokenizer), os.getpid()),
    )
    pool.submit(int).result()
    return pool


def generation_batches(lengths: torch.Tensor, gen_frames: int, group: int) -> list[torch.Tensor]:
    """The rows of each generation batch of rollouts `lengths` [N] (groups of `group` adjacent
    rows, equally long), longest first: whole groups while the batch's rows times its first,
    longest rollout stay within `gen_frames`, the size of its KV caches. Shorter rollouts thus
    batch more rows, and the last, partial batch runs the fewest steps. The sort is stable, so a
    group stays adjacent and in one batch."""
    order = torch.argsort(lengths, stable=True, descending=True)
    batches = []
    start = 0
    while start < len(order):
        rows = max(group, gen_frames // int(lengths[order[start]]) // group * group)
        batches.append(order[start : start + rows])
        start += rows
    return batches


def sample_window(
    generator: Any,
    info: Any,
    sources: torch.Tensor,
    lengths: torch.Tensor,
    batches: list[torch.Tensor],
    silence_frame: torch.Tensor,
    seed_key: tuple[int, ...],
    temperature: float,
    top_k: int,
    on_batch: Any,
) -> torch.Tensor:
    """Rollouts [N, 1 + dep_q, T] int32 on the CPU for `sources` [N, Q, T]: row i holds
    `lengths[i]` sampled frames and -1 after them. Each of `batches` (row indices, from
    `generation_batches`) runs to its longest row and is seeded from (`seed_key`, its index) as a
    32-bit seed (NumPy's limit); `on_batch(rows, rollouts)` follows it with its rows' indices."""
    count, _, frames = sources.shape
    outputs = torch.full((count, 1 + generator.dep_q, frames), -1, dtype=torch.int32)
    for index, batch_rows in enumerate(batches):
        batch_lengths = lengths[batch_rows]
        batch_frames = int(batch_lengths.max())
        steps = batch_frames + max(generator.delays)
        common.seed_all(int(np.random.SeedSequence([*seed_key, index]).generate_state(1)[0]))
        batch = sources[batch_rows, :, :batch_frames].to(silence_frame.device)
        lm_gen = get_lmgen(
            generator, info, batch_size=len(batch), temp=temperature, temp_text=temperature,
            top_k=top_k, top_k_text=top_k,
        )
        common.set_attention_context(generator.transformer, min(steps, generator.context))
        try:
            sampled = rollout(lm_gen, batch, silence_frame)
        finally:
            common.set_attention_context(generator.transformer, generator.context)
        sampled = sampled.to(device="cpu", dtype=torch.int32)
        past = torch.arange(batch_frames)[None] >= batch_lengths[:, None]
        sampled.masked_fill_(past[:, None], -1)
        outputs[batch_rows, :, :batch_frames] = sampled
        on_batch(batch_rows, sampled)
        # Leaving streaming freed the batch's KV caches and CUDA graphs; the dropped LMGen's own
        # cycle holds a few MB, collected with the generator at the window's end (grpo.md#running).
        del lm_gen, sampled
    return outputs


class HostOffload:
    """GPU storages parked in pinned host memory while a phase that never reads them runs. A
    window generates on a bf16 copy, so the fp32 policy and AdamW's states (37.5 GB on the 3.1B
    model) wait on the host and the generation batches get the GPU's memory (grpo.md#running).
    `to_host` copies each storage out and shrinks it to nothing, so nothing may read those tensors
    until `to_device` restores them exactly; the tensors, the optimizer's and the compiled
    forward's alike, stay the same objects. The pinned buffers are allocated once per storage and
    reused (AdamW's states appear at the first update)."""

    def __init__(self) -> None:
        self.host: dict[int, torch.Tensor] = {}
        self.out: list[tuple[torch.UntypedStorage, torch.Tensor]] = []

    def to_host(self, tensors: list[torch.Tensor]) -> None:
        if self.out:
            raise RuntimeError("HostOffload.to_host while storages are already out")
        storages = {}
        for tensor in tensors:
            storage = tensor.untyped_storage()
            if storage.nbytes():
                storages.setdefault(storage._cdata, storage)  # its StorageImpl: shared views once
        for key, storage in storages.items():
            if key not in self.host:
                self.host[key] = torch.empty(storage.nbytes(), dtype=torch.uint8, pin_memory=True)
            self.host[key].untyped_storage().copy_(storage, non_blocking=True)
            self.out.append((storage, self.host[key]))
        torch.cuda.synchronize()  # every copy has landed before its storage goes
        for storage, _ in self.out:
            storage.resize_(0)
        torch.cuda.empty_cache()  # whole segments for the generation batches

    def to_device(self) -> None:
        for storage, host in self.out:
            storage.resize_(host.numel())
            storage.copy_(host.untyped_storage(), non_blocking=True)
        torch.cuda.synchronize()
        self.out = []


def trainer_tensors(lm: Any, optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    """The fp32 policy's parameters and AdamW's state tensors (none before its first step)."""
    return [*lm.parameters(), *(value for state in optimizer.state.values()
                                for value in state.values() if torch.is_tensor(value))]


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


def parse_args() -> argparse.Namespace:
    # No abbreviations: h100.sh must be able to reject --out-dir.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cache-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-weight", type=Path, required=True, help="The SFT checkpoint.")
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, help="Default: the one the model weight is on (its text_tokenizer metadata), in weights/.")
    parser.add_argument("--hf-repo", default="kyutai/hibiki-zero-3b-pytorch-bf16")
    parser.add_argument("--max-frames", type=int, default=280)
    parser.add_argument(
        "--stream-seconds", type=float, nargs=2, default=[30.0, 75.0], metavar=("MIN", "MAX"),
        help="The paper's RL inputs are 30-75 s utterances.",
    )
    parser.add_argument("--frames", type=int, default=1200, help="The longest rollout.")
    parser.add_argument(
        "--tail-seconds", type=float, default=common.GenerationConfig.tail_s,
        help="A rollout runs this long past its stream, as evaluation generates past a source.",
    )
    parser.add_argument("--streams", type=int, default=32, help="Streams per update.")
    parser.add_argument("--group", type=int, default=4, help="Rollouts per stream (G).")
    parser.add_argument(
        "--updates", type=int, default=2000,
        help="The paper's 2,000; every evaluation promotes, so a plateaued run can just be stopped.",
    )
    parser.add_argument("--tau", type=int, default=20, help="Updates per π_old.")
    parser.add_argument("--lr", type=float, default=2e-7)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=EPSILON, help="Ratio clip.")
    parser.add_argument("--alpha", type=float, default=rewards.ALPHA, help="Final BLEU's share.")
    parser.add_argument("--text-weight", type=float, default=TEXT_WEIGHT, help="c_0 against 1 per codebook.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--gen-frames", type=int, default=230_400,
        help="A generation batch's rollouts times its longest rollout's frames (profiled: grpo.md).",
    )
    parser.add_argument("--micro-batch", type=int, default=2, help="Rollouts per forward; 3 runs out of memory at T 1,200.")
    parser.add_argument("--reward-workers", type=int, default=8)
    parser.add_argument("--eval-pairs", type=Path, nargs="+", default=EVAL_PAIRS)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("frames", "streams", "group", "updates", "tau", "gen_frames", "micro_batch",
                 "reward_workers", "eval_batch_size", "eval_every", "save_every"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.group < 2:
        raise ValueError("--group needs at least 2 rollouts to normalize")
    if args.save_every % args.tau or args.eval_every % args.save_every or args.updates % args.tau:
        raise ValueError(
            "--save-every and --updates must be multiples of --tau, --eval-every of --save-every"
        )
    if args.gen_frames < args.group * args.frames:
        raise ValueError("--gen-frames must hold a group of the longest rollouts")
    longest = round(args.stream_seconds[1] * streams.FRAME_RATE)
    if longest + round(args.tail_seconds * streams.FRAME_RATE) > args.frames:
        raise ValueError("--frames must hold the longest stream and the tail after it")
    if args.tail_seconds < 0:
        raise ValueError("--tail-seconds must not be negative")
    common.require_torch_compile()


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    validate_args(args)
    args.model_weight = require_file(args.model_weight, "start checkpoint")
    args.tokenizer = common.run_tokenizer(args.tokenizer, args.model_weight)  # before the workers load it
    pool = reward_pool(args.reward_workers, args.tokenizer)
    device = common.check_device("cuda")
    torch.set_float32_matmul_precision("high")

    cache_dirs = [require_dir(path, "code cache directory") for path in args.cache_dir]
    args.config_path = require_file(args.config_path, "config")
    args.mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    args.eval_pairs = [require_file(path, "evaluation pairs") for path in args.eval_pairs]
    out_dir = resolve_repo_path(args.out_dir)
    resuming, best_state = open_run(out_dir)

    info = common.load_checkpoint_info(args)
    tokenizer = info.get_text_tokenizer()
    silence = streams.silence_track(info.get_mimi(device="cpu"), args.frames)
    rows, layout = streams.load_rows(cache_dirs, args.max_frames, args.tokenizer)
    dataset = streams.StreamDataset(
        rows, layout, silence, streams.frame_range(args.stream_seconds), args.seed
    )
    if len(dataset) < args.updates * args.streams:
        raise ValueError(f"{len(dataset)} streams, fewer than {args.updates} updates need")
    print(f"Planned {len(dataset)} streams from {len(rows)} rows; {args.updates} updates use "
          f"{args.updates * args.streams}")
    # Planned streams follow their documents (a long recording fills ~75 in a row), so updates
    # take them in a seeded order: the same on a resume.
    order = stream_order(len(dataset), args.seed)
    eval_sets, eval_rows = common.load_eval_rows(args.eval_pairs)
    selecting = common.selecting_sets(eval_sets, eval_rows)
    print(f"Evaluation translates {len(eval_rows)} rows of {', '.join(eval_sets)}; {', '.join(selecting)} select")

    run_config = {key: jsonable(value) for key, value in vars(args).items()}
    run_config.update(
        train_rows=len(rows), train_streams=len(dataset), eval_rows=len(eval_rows),
        total_steps=args.updates, val_every=args.eval_every, best_metric=BEST_METRIC,
    )
    # Before the model loads: a resume with other arguments or data stops here, in seconds.
    claim_run_config(out_dir, run_config, RESUME_KEYS, resuming)
    config_path = out_dir / "run_config.json"

    lm = common.load_policy(info, device)
    params = list(lm.parameters())
    mimi = info.get_mimi(device=device)
    optimizer = common.fused_adamw(params, args)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    stream_weights = torch.ones(1 + lm.dep_q, device=device)
    stream_weights[0] = args.text_weight
    silence_frame = silence[:, -1].long().to(device)
    tail = round(args.tail_seconds * streams.FRAME_RATE)
    delays = torch.tensor(lm.delays[: 1 + lm.dep_q])  # the text and target codebooks'

    log_path = out_dir / "train_log.jsonl"
    val_log_path = out_dir / "val_log.jsonl"
    step = load_newest_pair(lm, optimizer, out_dir, device) if resuming else 0
    validated = {int(item["step"]) for item in keep_log_lines(val_log_path, step)}
    keep_log_lines(log_path, step)
    if step % args.tau:
        raise RuntimeError(f"Resumed step {step} is not on a --tau boundary")
    writer = CheckpointWriter(lm, optimizer, args, out_dir)
    # The rows and streams (~4 million objects for the tranche) never become garbage: frozen, they
    # stay out of the collection after every generation batch (0.6 s each otherwise).
    gc.collect()
    gc.freeze()
    # The start's end offset on each selecting set, which a promoted checkpoint may exceed by at
    # most LATENCY_SLACK_S. It is kept in run_config.json, which hf_sync uploads as it changes,
    # unlike the logs.
    start_offset = json.loads(config_path.read_text(encoding="utf-8")).get("start_end_offset_s")
    if step and start_offset is None:
        raise RuntimeError("run_config.json lacks the start's start_end_offset_s")

    def evaluate(at: int) -> None:
        nonlocal best_state, start_offset
        started = time.time()
        optimizer.zero_grad(set_to_none=True)
        free_running = common.validate_free_running(
            lm, info, mimi, tokenizer, eval_rows, eval_sets, args.eval_batch_size,
            out_dir / "eval" / f"step{at:06d}", args.seed,
        )
        healthy = all(metrics["health_eligible"] for metrics in free_running.values())
        for name, metrics in free_running.items():
            print(f"val step={at} {name} {common.score_summary(metrics)}")
        if at == 0:
            start_offset = {name: free_running[name]["median_end_offset_s"] for name in selecting}
            if silent := [name for name, offset in start_offset.items() if not math.isfinite(offset)]:
                raise RuntimeError(f"The start emitted no word on {', '.join(silent)}")
            stored = json.loads(config_path.read_text(encoding="utf-8"))
            stored["start_end_offset_s"] = start_offset
            atomic_write_text(json.dumps(stored, indent=2, sort_keys=True) + "\n", config_path)
        # The gate allows LATENCY_SLACK_S: two updates at 2e-7 (in effect the start) measured
        # 0.1 s later than the start, and a blocked checkpoint is not kept.
        on_time = all(free_running[name]["median_end_offset_s"] <= start_offset[name] + LATENCY_SLACK_S
                      for name in selecting)
        if at > 0 and healthy and on_time:
            writer.wait()
            chrf = sum(float(free_running[name]["chrf"]) for name in selecting) / len(selecting)
            best_state = promote_best(at, chrf, out_dir, best_state)
        promoted = best_state is not None and best_state["step"] == at
        seconds = time.time() - started
        item = {"step": at, "seconds": seconds, "free_running": free_running, "healthy": healthy,
                "on_time": on_time, "promoted": promoted}
        append_log(val_log_path, item)
        validated.add(at)
        print(f"val step={at} took {seconds:.0f} s")

    if step % args.eval_every == 0 and step not in validated:
        evaluate(step)

    per_update = args.streams * args.group
    offload = HostOffload()
    while step < args.updates:
        window = min(args.tau, args.updates - step)
        window_started = time.time()
        # 1. π_old's rollouts for the window, rewards in the workers as each batch lands.
        optimizer.zero_grad(set_to_none=True)
        stream_ids = [int(order[index]) for update in range(step, step + window)
                      for index in range(update * args.streams, (update + 1) * args.streams)]
        sources = torch.stack([stream_source(dataset, index, silence) for index in stream_ids])
        sources = sources.repeat_interleave(args.group, dim=0)
        lengths = torch.tensor(
            [rollout_frames(dataset.streams[index].frames, tail, args.frames)
             for index in stream_ids]
        ).repeat_interleave(args.group)
        jobs: dict[int, Future] = {}

        def submit(batch_rows: torch.Tensor, rollouts: torch.Tensor) -> None:
            for offset in range(0, len(batch_rows), args.group):
                first_row = int(batch_rows[offset])
                group = first_row // args.group
                stream = dataset.streams[stream_ids[group]]
                sentences = stream_sentences(stream, rows, tokenizer)
                tokens = rollouts[offset : offset + args.group, 0, : int(lengths[first_row])]
                jobs[group] = pool.submit(_group_rewards, tokens.numpy().copy(), sentences,
                                          args.alpha)

        generator = common.bf16_copy(info, lm, device)
        moving = time.time()
        offload.to_host(trainer_tensors(lm, optimizer))
        moved = time.time() - moving
        batches = generation_batches(lengths, args.gen_frames, args.group)
        rollouts = sample_window(generator, info, sources, lengths, batches, silence_frame,
                                 (args.seed, step), args.temperature, args.top_k, submit)
        del generator
        gc.collect()
        torch.cuda.empty_cache()
        moving = time.time()
        offload.to_device()
        moved += time.time() - moving
        generated = time.time() - window_started
        results = [jobs[group].result() for group in range(len(stream_ids))]
        advantages = torch.zeros((len(rollouts), args.frames))
        for group, result in enumerate(results):  # [G, the group's length]; 0 past it
            rows_of = slice(group * args.group, (group + 1) * args.group)
            advantages[rows_of, : result[2].shape[1]] = torch.from_numpy(result[2]).float()
        final_bleu = np.concatenate([result[1][:, -1] for result in results])
        eos_rate = (rollouts[:, 0] == tokenizer.eos_id()).any(dim=1).float()
        keep = kept_tokens(rollouts, lengths, delays, int(tokenizer.eos_id()))
        spans = forward_frames(keep, delays)
        codes = torch.cat([rollouts, sources.to(torch.int32)], dim=1)
        rewarded = time.time() - window_started

        # 2. π_old's logprobs: the policy before this window's first update, through the update's
        # own forward (gradients enabled, never backpropagated).
        # Micro-batches cut as each update cuts them: the same rows in the same batch shapes.
        old_logprobs = torch.zeros((len(codes), 1 + lm.dep_q, args.frames), dtype=torch.float32)
        with autocast:
            for start, end in micro_batches(window, per_update, args.micro_batch):
                frames = micro_frames(spans, start, end, args.frames)
                batch = codes[start:end, :, :frames].to(device=device, dtype=torch.long)
                condition = common.batch_condition_tensors(lm, info.model_type, end - start)
                logprobs, _ = token_logprobs(lm, batch, condition, args.temperature)
                old_logprobs[start:end, :, :frames] = logprobs.detach().cpu()
                del logprobs
        scored = time.time() - window_started

        # 3. The window's updates.
        for update in range(window):
            update_started = time.time()
            optimizer.zero_grad(set_to_none=True)
            first = update * per_update
            totals = {"loss": 0.0, "ratio_sum": 0, "clipped": 0, "tokens": 0, "max_deviation": 0}
            forward_total = 0
            for start, end in micro_batches(window, per_update, args.micro_batch, update):
                frames = micro_frames(spans, start, end, args.frames)
                forward_total += frames * (end - start)
                batch = codes[start:end, :, :frames].to(device=device, dtype=torch.long)
                condition = common.batch_condition_tensors(lm, info.model_type, end - start)
                with autocast:
                    loss, stats = grpo_loss(
                        lm, batch, condition, old_logprobs[start:end, :, :frames].to(device),
                        advantages[start:end, :frames].to(device),
                        keep[start:end, :, :frames].to(device),
                        stream_weights, args.epsilon, args.temperature)
                (loss / per_update).backward()
                totals["loss"] += loss.detach()
                totals["max_deviation"] = torch.maximum(
                    torch.as_tensor(totals["max_deviation"], device=device), stats.pop("max_deviation"))
                for key, value in stats.items():
                    totals[key] += value
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip or math.inf)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError(f"Non-finite gradient norm at update {step + 1}")
            optimizer.step()
            step += 1
            tokens = float(totals["tokens"])
            item = {
                "step": step,
                "loss": float(totals["loss"]) / per_update,
                "ratio_mean": float(totals["ratio_sum"]) / tokens,
                "clip_fraction": float(totals["clipped"]) / tokens,
                "ratio_max_deviation": float(totals["max_deviation"]),
                "grad_norm": float(grad_norm),
                "final_bleu": float(final_bleu[first : first + per_update].mean()),
                "eos_rate": float(eos_rate[first : first + per_update].mean()),
                "rollout_frames": float(lengths[first : first + per_update].float().mean()),
                "forward_frames": forward_total / per_update,
                "lr": args.lr,
                "sec_per_update": time.time() - update_started,
                "window_generate_s": generated,
                "window_gen_batches": len(batches),
                "window_move_s": moved,
                "window_reward_wait_s": rewarded - generated,
                "window_old_logprob_s": scored - rewarded,
            }
            append_log(log_path, item)
            print(f"update={step} loss={item['loss']:.1f} bleu={item['final_bleu']:.1f} "
                  f"ratio={item['ratio_mean']:.4f} clip={item['clip_fraction']:.4f} "
                  f"grad_norm={item['grad_norm']:.2f} s={item['sec_per_update']:.1f}", flush=True)
            if step % args.save_every == 0:
                writer.save(step)
            if step % args.eval_every == 0:
                evaluate(step)

    writer.save(step)
    if step not in validated:
        evaluate(step)
    writer.wait()
    pool.shutdown()
    print(f"Saved the final GRPO checkpoint at update {step} in {repo_display_path(out_dir)}")


if __name__ == "__main__":
    main()

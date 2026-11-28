"""GRPO's rollouts (`grpo.py`, steps 1 and 2): the run's streams and the order its updates take
them in, a window's rollouts sampled in generation batches on a bf16 copy of the policy, and the
reward workers that score them (`rewards.py`) while the GPU keeps generating.
"""
from __future__ import annotations

import argparse
import gc
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from moshi.modules.transformer import set_attention_context

from training.finetune import rewards, streams
from training.finetune.generation import get_lmgen
from training.finetune.model import seed_all
from training.finetune.packs import Layout
from training.finetune.rooms import run_rooms
from training.finetune.rows import Row

# Rooms where nobody speaks (rooms.run_rooms), drawn once (a window's 640 streams draw from them).
# Half the streams hold one: the other half keep digital silence, what a voice gate passes.
ROOMS = 256
ROOM_SHARE = 0.5


def stream_order(count: int, seed: int) -> np.ndarray:
    """The order updates take the planned streams in (`streams.plan_streams` keeps a document's
    rows in consecutive streams, so in plan order a long recording would fill several updates)."""
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
    # Step s reads source frame s - 1 (step 0's input is never read); past `sources` (the batch's
    # longest rollout), silence, which no kept token reads.
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


def stream_sentences(
    stream: streams.Stream, rows: list[Row], tokenizer: Any
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


def stream_dataset(
    args: argparse.Namespace, info: Any, rows: list[Row], layout: Layout,
    codebooks: int, mimi: Any,
) -> streams.StreamDataset:
    """The run's streams, planned from `rows` with --seed: assembled at the cache's codebooks and
    cut to the model's first `codebooks`, as the SFT trainer's are, each as long as --frames at
    most; a --room-share of them hold a room in their source where nobody speaks, encoded by
    `mimi` (the run's, on its GPU)."""
    silence = streams.silence_track(info.get_mimi(device="cpu"), args.frames, layout.dep_q)
    return streams.StreamDataset(
        rows, layout, silence, streams.frame_range(args.stream_seconds), args.seed, codebooks,
        run_rooms(args, mimi, silence, ROOMS), args.room_share,
    )


def stream_source(dataset: streams.StreamDataset, index: int, codebooks: int) -> torch.Tensor:
    """A stream's source codes [Q, T] at the model's Q `codebooks`, T the dataset's silence
    frames: the stream, then what its source holds wherever nobody speaks (a room or silence,
    `source_quiet`). The dataset's items hold text, then Q target and Q source codebooks."""
    codes = dataset[index]["codes"]
    if codes.shape[0] != 1 + 2 * codebooks:
        raise ValueError(f"A stream of {codes.shape[0]} rows is not text and twice {codebooks} codebooks")
    source = dataset.source_quiet(index)[:codebooks].long().clone()
    source[:, : codes.shape[1]] = codes[1 + codebooks :]
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
        seed_all(int(np.random.SeedSequence([*seed_key, index]).generate_state(1)[0]))
        batch = sources[batch_rows, :, :batch_frames].to(silence_frame.device)
        lm_gen = get_lmgen(
            generator, info, batch_size=len(batch), temp=temperature, temp_text=temperature,
            top_k=top_k, top_k_text=top_k,
        )
        set_attention_context(generator.transformer, min(steps, generator.context))
        try:
            sampled = rollout(lm_gen, batch, silence_frame)
        finally:
            set_attention_context(generator.transformer, generator.context)
        sampled = sampled.to(device="cpu", dtype=torch.int32)
        past = torch.arange(batch_frames)[None] >= batch_lengths[:, None]
        sampled.masked_fill_(past[:, None], -1)
        outputs[batch_rows, :, :batch_frames] = sampled
        on_batch(batch_rows, sampled)
        del lm_gen, sampled
        gc.collect()  # moshi's streaming state holds the KV cache in reference cycles
    return outputs

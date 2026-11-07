#!/usr/bin/env python
"""The tranche's training caches, built on CUDA: each segments shard's rows as grounded-v2 cache
rows ([tranche-cache](../../docs/agent-memory/tranche-cache.md)), in the format of the
Mac's MLX builder `training.prep.cache_codes` ([sft-data-caches](../../docs/agent-memory/sft-data-caches.md)).

  build <sources...>  every shard whose segments, text_en, en_audio and asr_vi are on the Hub
                      becomes <root>/cache/<source>/shard_<shard stem>.pt:
                      the rows the filters below keep, the English text timed by wav2vec2-base-960h
                      CTC (the `text_timing` Viterbi) and both clips Mimi-encoded in float32, each
                      sample with its `doc` and `seq` so the trainer joins a document's rows in
                      order. Pushes each source's folder every 30 min and at its end; a rerun skips
                      the shards written or pushed, and first pushes any written but not yet pushed.

A row is dropped, with the first reason that applies (the shard's `rejected`), when
  - its English stayed broken after the TTS's retries (`en_broken`);
  - the ASR of its English writes Vietnamese letters its text_en lacks (a Vietnamese lead-in);
  - Qwen3-ASR heard its clip as another language than Vietnamese, or none;
  - its corpus transcript and that ASR differ by more than --max-cer (CER over `filters.spoken`);
  - the ASR hears more nghìn/triệu/tỷ than the transcript holds (GigaSpeech 2's transcripts drop them);
  - its transcript overlaps an evaluation sentence (`filters.overlaps`; needs data/eval);
  - a clip decodes more than 0.02 s off its stored duration, or the row exceeds --max-frames;
  - its English text fails CTC alignment or token timing.

Decode workers (PyAV, soxr) feed the GPU, which runs the CTC forward over padded batches (the
padding masked, so each clip's emissions are its own) and Mimi over padded batches; the CTC
Viterbi runs in the workers while Mimi encodes. Env: the training .venv (training/h100.sh setup).
From the repo root on the box, with HF_TOKEN_AN (pushes) in the environment:
  .venv/bin/python -m training.tranche.cache build pho vivoice giga
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import re
import time
import unicodedata
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.paths import (
    CACHE_FORMAT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_TOKENIZER,
    cpu_budget,
    durable,
    exclusive,
    read_json,
    require_file,
)
from training.prep.audio import Rejected
from training.prep.cache_rows import CONFIG_KEYS, FRAME_RATE, assemble_codes, target_delay_s
from training.prep.text_timing import (
    ALIGNER_REPO,
    ALIGNER_REVISION,
    ALIGNMENT_NAME,
    DEFAULT_MIN_ALIGNMENT_SCORE,
    AlignmentError,
    CTCWordAligner,
    aligner_files,
    sentencepiece_groups,
    timed_sentencepiece_tokens,
)
from training.tranche.filters import eval_grams, overlaps, spoken
from training.tranche.ingest import CACHE_REPO, REPO, ROOT, SOURCES, part_file, push_part

SAMPLE_RATE = 24_000  # Mimi
CTC_RATE = 16_000
FRAME_SAMPLES = 1920
SPLIT = "train"  # the delay draw's key, as cache_codes seeds it
SEED = 1234  # cache_codes' defaults: the delay seed, ratio and the alignment floor
DELAY_RATIO = 0.5
DURATION_TOLERANCE_S = 0.02
# Above 0.15 the corpus transcript and the clip differ by phrases (PhoAudiobook's repeated clauses);
# from 0.10 to 0.15 mostly the ASR's misses on names and loanwords, numbers and percentages spelled
# otherwise, a word at a cut edge (tranche-cache#gates).
MAX_CER = 0.15
# GigaSpeech 2's transcripts drop these from numbers ("hơn một chuyến" where the speaker says "hơn
# một nghìn chuyến"), a wrong fact the CER barely sees (tranche-cache#gates).
MAGNITUDES = ("nghìn", "triệu", "tỷ")
VI_LETTERS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
MIMI_BATCH_S = 640.0  # padded audio per Mimi batch, as mimi_cache_cuda
CTC_BATCH_S = 640.0
PUSH_S = 1800
PARTS = ("segments", "text_en", "en_audio", "asr_vi")
DEVICE = "cuda"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("build")
    p.add_argument("sources", nargs="+", choices=sorted(SOURCES))
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--limit", type=int, help="build only this many shards (a staged start)")
    p.add_argument("--max-cer", type=float, default=MAX_CER)
    p.add_argument("--max-frames", type=int, default=280, help="The trainer's frame cap.")
    p.add_argument("--workers", type=int, help="decode and alignment processes (default: the CPU budget less one)")
    p.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    return parser.parse_args()


# --------------------------------------------------------------------------- workers


def prepare(vi_audio: bytes, en_audio: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Worker: both clips at 24 kHz for Mimi, and the English at 16 kHz for CTC, normalized to zero
    mean and unit variance over the clip as wav2vec2's feature extractor does."""
    import soxr

    from training.prep.audio import decode

    vi, vi_rate = decode(vi_audio)
    en, en_rate = decode(en_audio)
    en16 = soxr.resample(en, en_rate, CTC_RATE).astype(np.float32)
    en16 = (en16 - en16.mean()) / np.sqrt(en16.var() + 1e-7)
    return soxr.resample(vi, vi_rate, SAMPLE_RATE), soxr.resample(en, en_rate, SAMPLE_RATE), en16


_aligner: CTCWordAligner | None = None


def align(
    files: Path,
    log_probs: np.ndarray,
    lead: list[int],
    groups: list[tuple[list[int], str]],
    raw_target_frames: int,
    delay_frames: int,
    eos_id: int,
) -> tuple[list[int], list[int], float]:
    """Worker: the row's text tokens and their frames, and the alignment score; raises
    AlignmentError to reject the row. `files`: the aligner snapshot, resolved by the parent."""
    global _aligner
    if _aligner is None:
        _aligner = CTCWordAligner(files, DEFAULT_MIN_ALIGNMENT_SCORE)
    alignment = _aligner.align(log_probs, groups)
    tokens, frames = timed_sentencepiece_tokens(lead, groups, alignment, raw_target_frames, delay_frames, eos_id)
    return tokens, frames, alignment.score


# --------------------------------------------------------------------------- GPU


def batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Indices shortest first, cut where the batch padded to its longest would pass `budget`."""
    out: list[list[int]] = []
    for index in sorted(range(len(lengths)), key=lengths.__getitem__):
        if out and (len(out[-1]) + 1) * lengths[index] <= budget:
            out[-1].append(index)
        else:
            out.append([index])
    return out


class Models:
    """Mimi and wav2vec2-base-960h CTC on CUDA, both in full float32."""

    def __init__(self, args: argparse.Namespace, cfg: dict[str, Any]):
        from moshi.models import loaders
        from transformers import Wav2Vec2ForCTC

        # TF32 changes Mimi's residual codes as bf16 does (mimi_cache_cuda), and moves CTC near-ties.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(DEVICE)
        num_codebooks = max(int(cfg["dep_q"]), int(cfg["n_q"]) - int(cfg["dep_q"]))
        self.mimi = loaders.get_mimi(require_file(args.mimi_weight, "Mimi weight"), device=self.device,
                                     num_codebooks=num_codebooks).eval()
        if self.mimi.cardinality != int(cfg["card"]):
            raise RuntimeError(f"config card {cfg['card']} != Mimi cardinality {self.mimi.cardinality}")
        self.ctc = Wav2Vec2ForCTC.from_pretrained(ALIGNER_REPO, revision=ALIGNER_REVISION,
                                                  dtype=torch.float32).to(self.device).eval()
        if self.ctc.config.feat_extract_norm != "group":
            raise RuntimeError("the CTC forward masks the first GroupNorm of a group-norm wav2vec2")

    @torch.no_grad()
    def mimi_codes(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        """Each clip's codes [codebooks, ceil(samples / 1920)]. Right padding changes nothing
        before it: Mimi's convolutions and attention are causal."""
        out: list[np.ndarray] = [np.empty(0)] * len(clips)
        for group in batches([len(c) for c in clips], int(MIMI_BATCH_S * SAMPLE_RATE)):
            frames = [math.ceil(len(clips[i]) / FRAME_SAMPLES) for i in group]
            batch = torch.zeros(len(group), 1, max(frames) * FRAME_SAMPLES, device=self.device)
            for row, i in enumerate(group):
                batch[row, 0, : len(clips[i])] = torch.from_numpy(clips[i])
            codes = self.mimi.encode(batch).to("cpu", torch.int32).numpy()
            for row, (i, count) in enumerate(zip(group, frames, strict=True)):
                out[i] = codes[row, :, :count]
        return out

    @torch.no_grad()
    def ctc_log_probs(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        """Each normalized 16 kHz clip's CTC log-probabilities [frames, vocab], as computed alone:
        the base model normalizes its first convolution over the whole input and has no attention
        mask, so the padding is masked out of those statistics, the positional convolution's input
        and the attention keys (as wav2vec2_mlx does)."""
        w2v = self.ctc.wav2vec2
        first, rest = w2v.feature_extractor.conv_layers[0], w2v.feature_extractor.conv_layers[1:]
        out: list[np.ndarray] = [np.empty(0)] * len(clips)
        for group in batches([len(c) for c in clips], int(CTC_BATCH_S * CTC_RATE)):
            lengths = torch.tensor([len(clips[i]) for i in group])
            batch = torch.zeros(len(group), 1, int(lengths.max()), device=self.device)
            for row, i in enumerate(group):
                batch[row, 0, : len(clips[i])] = torch.from_numpy(clips[i])
            hs = first.conv(batch)
            valid = (lengths - first.conv.kernel_size[0]) // first.conv.stride[0] + 1
            own = (torch.arange(hs.shape[-1])[None, :] < valid[:, None])[:, None, :].to(self.device, torch.float32)
            count = own.sum(-1, keepdim=True)
            mean = (hs * own).sum(-1, keepdim=True) / count
            var = ((hs - mean).square() * own).sum(-1, keepdim=True) / count
            norm = first.layer_norm
            hs = (hs - mean) / torch.sqrt(var + norm.eps) * norm.weight[:, None] + norm.bias[:, None]
            hs = first.activation(hs)
            for layer in rest:
                hs = layer(hs)
            hs, _ = w2v.feature_projection(hs.transpose(1, 2))
            frames = self.ctc._get_feat_extract_output_lengths(lengths)
            mask = (torch.arange(hs.shape[1]) < frames[:, None]).to(self.device)
            hs = w2v.encoder(hs, attention_mask=mask).last_hidden_state
            log_probs = torch.log_softmax(self.ctc.lm_head(hs), dim=-1).cpu().numpy()
            for row, (i, count) in enumerate(zip(group, frames.tolist(), strict=True)):
                out[i] = log_probs[row, :count]
        return out


# --------------------------------------------------------------------------- shards


def vietnamese_letters(en_asr: str, text_en: str) -> set[str]:
    """Vietnamese letters the English ASR wrote that the English text lacks: a clause the TTS spoke
    in Vietnamese (tranche-tts), or a Vietnamese name the ASR spelled otherwise."""
    have = set(unicodedata.normalize("NFC", text_en).lower())
    return {c for c in unicodedata.normalize("NFC", en_asr).lower() if c in VI_LETTERS} - have


def fetch(root: Path, source: str, name: str) -> dict[str, Path]:
    """The shard's input files, the box's own copies first (`ingest.part_file`)."""
    return {part: part_file(root, source, part, name) for part in PARTS}


def magnitudes(text: str) -> list[int]:
    words = spoken(text)
    return [words.count(m) for m in MAGNITUDES]


def read_parts(source: str, name: str, files: dict[str, Path]) -> list[dict[str, Any]]:
    """The shard's rows with every part's columns, in the segments' order."""
    import pyarrow.parquet as pq

    get = lambda part, columns: pq.read_table(files[part], columns=columns).to_pylist()  # noqa: E731
    rows = get("segments", ["id", "doc", "seq", "text", "duration_s", "audio"])
    for part, columns in (("text_en", ["id", "text_en"]),
                          ("en_audio", ["id", "en_audio", "en_duration_s", "en_asr", "en_broken"]),
                          ("asr_vi", ["id", "asr_vi", "language", "cer"])):
        other = get(part, columns)
        if [r["id"] for r in other] != [r["id"] for r in rows]:
            raise RuntimeError(f"{source}/{part}/{name} does not hold the segments' rows in their order")
        for row, extra in zip(rows, other, strict=True):
            row.update(extra)
    return rows


def gate(row: dict[str, Any], max_cer: float, grams) -> str | None:
    """The first filter that drops the row, before any audio is decoded."""
    if row["en_broken"]:
        return "en_broken"
    if letters := vietnamese_letters(row["en_asr"], row["text_en"]):
        return f"Vietnamese letters in en_asr: {''.join(sorted(letters))}"
    if row["language"] != "Vietnamese":
        return f"language {row['language'] or 'none'}"
    if round(row["cer"], 6) > max_cer:  # float32: 0.1 is stored as 0.10000000149
        return f"CER {round(row['cer'], 6)} over {max_cer}"
    # Word by word: a swapped magnitude (the ASR's tỷ for the transcript's triệu) is a wrong fact.
    if any(heard > written for heard, written in zip(magnitudes(row["asr_vi"]), magnitudes(row["text"]))):
        return "the ASR hears a nghìn/triệu/tỷ the transcript lacks"
    if overlaps(row["text"], grams):
        return "overlaps an evaluation sentence"
    return None


@dataclass
class Loaded:
    """A shard's rows, gated, with the kept rows' clips queued for decoding."""

    source: str
    name: str
    rows: list[dict[str, Any]]
    rejected: list[dict[str, str]]
    clips: dict[int, Future] = field(default_factory=dict)  # row index -> (vi24, en24, en16)


def read_shard(root: Path, source: str, name: str, max_cer: float, grams) -> tuple[Loaded, dict[int, tuple[bytes, bytes]]]:
    """The shard fetched, read and gated (in the read thread), with the kept rows' audio."""
    rows = read_parts(source, name, fetch(root, source, name))
    shard, audio = Loaded(source, name, rows, []), {}
    for index, row in enumerate(rows):
        if reason := gate(row, max_cer, grams):
            shard.rejected.append({"id": row["id"], "reason": reason})
        else:
            audio[index] = (row["audio"]["bytes"], row["en_audio"]["bytes"])
        row["audio"] = row["en_audio"] = None
    return shard, audio


def queue(read: tuple[Loaded, dict[int, tuple[bytes, bytes]]], pool: ProcessPoolExecutor) -> Loaded:
    """The read shard with its kept rows' clips submitted to the decode workers."""
    shard, audio = read
    shard.clips = {index: pool.submit(prepare, *clips) for index, clips in audio.items()}
    return shard


@dataclass
class Pending:
    """A row past the decode gates, waiting for its codes and text timing."""

    index: int
    row: dict[str, Any]
    vi24: np.ndarray
    en24: np.ndarray  # with the delay's silence in front
    en16: np.ndarray
    lead: list[int]
    groups: list[tuple[list[int], str]]
    delay_s: float
    delay_frames: int
    vi_frames: int
    en_frames: int


class Builder:
    def __init__(self, args: argparse.Namespace):
        import sentencepiece

        self.args = args
        self.cfg = read_json(args.config_path)
        self.tokenizer = sentencepiece.SentencePieceProcessor(str(require_file(args.tokenizer, "text tokenizer")))
        self.eos_id = int(self.tokenizer.eos_id())
        self.aligner_files = aligner_files()
        self.models = Models(args, self.cfg)

    def pending(self, index: int, row: dict[str, Any], clips: tuple[np.ndarray, ...]) -> Pending:
        """The decode gates, the delay draw and the text grouping; raises Rejected or AlignmentError."""
        vi24, en24, en16 = clips
        for side, pcm, stored in (("vi", vi24, row["duration_s"]), ("en", en24, row["en_duration_s"])):
            seconds = len(pcm) / SAMPLE_RATE
            if abs(seconds - stored) > DURATION_TOLERANCE_S:
                raise Rejected(f"{side} audio decodes to {seconds:.2f} s, the row says {stored:.2f} s")
        vi_frames = math.ceil(len(vi24) / FRAME_SAMPLES)
        frames = max(math.ceil(len(en24) / FRAME_SAMPLES), vi_frames) + 1  # text EOS, codec EOS
        if frames > self.args.max_frames:
            raise Rejected(f"{frames} frames exceed --max-frames {self.args.max_frames}")
        # As cache_codes: the delay is drawn within the room the English leaves under the cap.
        room_s = ((self.args.max_frames - 1) * FRAME_SAMPLES - len(en24)) / SAMPLE_RATE
        delay_s = target_delay_s(row["id"], SPLIT, min(DELAY_RATIO * round(row["duration_s"], 2), room_s), SEED)
        pad = int(round(delay_s * SAMPLE_RATE))
        if int(self.models.ctc._get_feat_extract_output_lengths(torch.tensor(len(en16)))) < 1:
            raise Rejected("the English clip is shorter than one CTC frame")
        lead, groups = sentencepiece_groups(row["text_en"], self.tokenizer)
        return Pending(index, row, vi24, np.concatenate([np.zeros(pad, np.float32), en24]), en16, lead, groups,
                       delay_s, int(round(delay_s * FRAME_RATE)), vi_frames, math.ceil((pad + len(en24)) / FRAME_SAMPLES))

    def build(self, shard: Loaded, pool: ProcessPoolExecutor, after_ctc) -> dict[str, Any]:
        """The shard's payload. `after_ctc` runs once the CTC forward is done (the next shard's
        load), so its decodes queue behind this shard's alignments."""
        rejected = shard.rejected
        todo: list[Pending] = []
        for index, future in shard.clips.items():
            row = shard.rows[index]
            try:
                todo.append(self.pending(index, row, future.result()))
            except (Rejected, AlignmentError) as exc:
                rejected.append({"id": row["id"], "reason": f"{type(exc).__name__}: {exc}"})
        shard.clips.clear()
        log_probs = self.models.ctc_log_probs([p.en16 for p in todo])
        aligned = [pool.submit(align, self.aligner_files, lp, p.lead, p.groups, p.en_frames - p.delay_frames, p.delay_frames, self.eos_id)
                   for lp, p in zip(log_probs, todo, strict=True)]
        after_ctc()
        vi_codes = self.models.mimi_codes([p.vi24 for p in todo])
        en_codes = self.models.mimi_codes([p.en24 for p in todo])
        samples = []
        for p, future, vi, en in zip(todo, aligned, vi_codes, en_codes, strict=True):
            try:
                tokens, text_frames, score = future.result()
            except AlignmentError as exc:
                rejected.append({"id": p.row["id"], "reason": f"AlignmentError: {exc}"})
                continue
            codes = assemble_codes(p.row["id"], vi, en, tokens, text_frames, self.cfg)
            samples.append({
                "id": p.row["id"], "split": SPLIT, "codes": torch.from_numpy(codes),
                "frames": int(codes.shape[1]), "vi_frames": p.vi_frames, "en_frames": p.en_frames,
                "text_tokens": len(tokens), "target_delay_s": p.delay_s, "target_delay_frames": p.delay_frames,
                "text_en": p.row["text_en"], "text_vi": p.row["text"], "doc": p.row["doc"], "seq": int(p.row["seq"]),
                "text_timing": ALIGNMENT_NAME, "alignment_score": score,
                "alignment_text": " ".join(spoken for _, spoken in p.groups),
                "source_row": f"{shard.source}/segments/{shard.name}#r{p.index}",
            })
        return {
            "format": CACHE_FORMAT, "sample_rate": SAMPLE_RATE, "frame_rate": FRAME_RATE,
            "alignment_min_score": DEFAULT_MIN_ALIGNMENT_SCORE, "max_frames": self.args.max_frames,
            "config": {key: int(self.cfg[key]) for key in CONFIG_KEYS},
            "source": {"file": f"{shard.source}/segments/{shard.name}", "rows": len(shard.rows)},
            "max_cer": self.args.max_cer, "samples": samples, "rejected": rejected,
        }


def cache_dir(root: Path, source: str) -> Path:
    return root / "cache" / source


def shard_file(name: str) -> str:
    return f"shard_{name.removesuffix('.parquet')}.pt"


def write(payload: dict[str, Any], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(f".{dst.name}.tmp")
    torch.save(payload, part)
    durable(part)
    part.replace(dst)
    durable(dst.parent)


def todo_shards(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The shards with every input on the Hub and no cache here or there, pushing first the caches
    an earlier run wrote but did not push."""
    from huggingface_hub import HfApi

    api, todo = HfApi(), []
    files, cached = (api.list_repo_files(r, repo_type="dataset") for r in (REPO, CACHE_REPO))
    for source in args.sources:
        parts = {part: {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/{part}/")}
                 for part in ("segments", "text_en", "en_audio", "asr_vi")}
        pushed = {f.rsplit("/", 1)[1] for f in cached if f.startswith(f"{source}/")}
        local = {p.name for p in cache_dir(args.root, source).glob("shard_*.pt")}
        ready = set.intersection(*parts.values())
        waiting = len(parts["segments"] - ready)
        todo += [(source, n) for n in sorted(ready) if shard_file(n) not in pushed | local]
        log(f"{source}: {len(parts['segments'])} shards, {len(pushed)} cached on the Hub, {len(local - pushed)} "
            f"only here, {waiting} waiting for an input")
        if local - pushed:
            push_part(args.root, source, "cache")
    return todo[: args.limit]


def build(args: argparse.Namespace) -> None:
    from training.tranche.ingest import an_token

    an_token()
    # Taken first: a second run must not push or load models onto a GPU the first one uses.
    args.root.mkdir(parents=True, exist_ok=True)
    with exclusive(args.root / ".cache.lock", f"another cache build uses {args.root}"):
        todo = todo_shards(args)
        log(f"{len(todo)} shards to build: " + ", ".join(f"{s} {sum(t[0] == s for t in todo)}" for s in args.sources))
        if todo:
            build_shards(args, todo)


def build_shards(args: argparse.Namespace, todo: list[tuple[str, str]]) -> None:
    grams = eval_grams()
    builder = Builder(args)
    workers = args.workers or max(1, math.floor(cpu_budget()) - 1)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # the spawned workers' thread pools, under the box's thread limit (voice.py)
    pushed = {s: time.monotonic() for s in args.sources}
    with (ThreadPoolExecutor(1) as reads, ThreadPoolExecutor(1) as writes,
          ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool):
        try:
            ahead = queue(read_shard(args.root, *todo[0], args.max_cer, grams), pool)
            written: Future | None = None  # the last shard's save, which runs while the next one builds
            began, last = time.monotonic(), time.perf_counter()
            for k, (source, name) in enumerate(todo):
                shard = ahead
                # The next shard downloads, reads and is gated in a thread while this one builds.
                read = reads.submit(read_shard, args.root, *todo[k + 1], args.max_cer, grams) if k + 1 < len(todo) else None

                def load_next() -> None:
                    nonlocal ahead
                    if read is not None:
                        ahead = queue(read.result(), pool)

                payload = builder.build(shard, pool, load_next)
                if written is not None:
                    written.result()
                written = writes.submit(write, payload, cache_dir(args.root, source) / shard_file(name))
                now = time.perf_counter()
                wall, last = now - last, now
                kept = payload["samples"]
                hours = sum(s["vi_frames"] + s["en_frames"] for s in kept) / FRAME_RATE / 3600
                reasons = collections.Counter(re.sub(r"[\d.]+", "N", r["reason"]).split(":")[0] for r in payload["rejected"])
                log(f"{source}/{name}: {len(kept)} of {len(shard.rows)} rows kept, rejected {dict(reasons)}; "
                    f"{hours:.2f} h of Mimi audio in {wall:.0f} s ({hours * 3600 / wall:.0f}x RT) [{k + 1}/{len(todo)}]")
                last_of_source = k + 1 == len(todo) or todo[k + 1][0] != source
                if last_of_source or time.monotonic() - pushed[source] > PUSH_S:
                    written.result()
                    push_part(args.root, source, "cache")
                    pushed[source] = time.monotonic()
            log(f"done: {len(todo)} shards in {(time.monotonic() - began) / 3600:.2f} h")
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)  # drop the queued decodes rather than run them
            raise


def main() -> None:
    args = parse_args()
    {"build": build}[args.step](args)


if __name__ == "__main__":
    main()

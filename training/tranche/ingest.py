#!/usr/bin/env python
"""Ingest: the tranche's real Vietnamese segments, one parquet per source shard, in the corpus's
own order ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#pipeline)), and the tranche
repos' layout every stage reads and pushes by. PhoAudiobook (its 224 shards of original chunks),
viVoice (354) and GigaSpeech 2 (240 archives) are in the tranche repo whole, each row with its asr_vi.

  segments yodas     download each YODAS3 tar at its pinned commit, cut its rows from whole videos
                     (training/tranche/yodas.py), hear them on the ASR server and write
                     <root>/yodas/segments/<tar>.parquet and its asr_vi; the download is deleted
  push <source>      upload <root>/<source>/segments, or with --part another stage's folder
                     (text_en, en_audio, asr_vi; cache: <root>/cache/<source>, to CACHE_REPO), to
                     the tranche repo (a rerun skips files already committed), only while it is gated

Every source goes to the public repos of the anquachdev account (`REPO`, `CACHE_REPO`), gated by the
sources' terms: GigaSpeech 2's own terms of access, which let a researcher share it with those who first
agree to them; PhoAudiobook with its author's permission (the user, 2026-11-06). Rows, sorted by
seq within a shard:
  id, source
  doc       the recording the row comes from, as far as the corpus tells; a doc never spans
            shards. PhoAudiobook: one reader's run of chunks in rising time (it names no book);
            GigaSpeech 2: the recording; viVoice: the row itself (its rows are shuffled);
            YODAS3: the video.
  seq       the row's order within its shard, and so within its doc
  speaker   the corpus's reader or channel, which can be a platform rather than one voice;
            none for GigaSpeech 2 and YODAS3, whose recordings may hold several voices
  start_s, end_s  the times the corpus names the clip by, within the recording; the trimmed
            audio can be seconds shorter than end - start, so seams come from alignment
  text      the corpus transcript (YODAS3: Qwen3-ASR's); duration_s, source_rate and audio
            {bytes, path} of the clip
A rerun skips shards already written. In the ingest env, beside the ASR server; from the repo root
on the box:
  /workspace/envs/ingest/bin/python -m training.tranche.ingest segments yodas --shards 0:25
"""
from __future__ import annotations

import argparse
import io
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pyarrow as pa

ROOT = Path("/workspace/tranche/repo")
# Public, gated by the sources' terms (GigaSpeech 2's in full); the anquachdev account's, so pushes take
# HF_TOKEN_AN. REPO holds every part but the caches (beside PhoMT's synthetic pairs in data/), as
# <source>/<part>/<shard>; CACHE_REPO the caches, as <source>/shard_<shard>.pt.
REPO = "anquachdev/en-vi-speech-20k"
CACHE_REPO = "anquachdev/mimi-en-vi-speech-20k"
OPUS_BITRATE = 32_000  # real speech, noise included; Mimi keeps far less than this
OPUS_COMPLEXITY = 5
ROW_S = (5.0, 22.4)  # the trainer's row lengths: YODAS's rows are cut to them


@dataclass(frozen=True)
class Source:
    repo: str
    revision: str
    pattern: str  # the source shards, a regex over the repo's file names
    # Shards (by their parquet name) held out of training as an evaluation set of the tranche's own
    # distribution (eval_sets' giga-heldout): voice.py and the cache build skip them.
    heldout: frozenset[str] = frozenset()


SOURCES = {
    "pho": Source("thivux/phoaudiobook", "f19f241", r"data/train-\d+-of-\d+\.parquet"),
    "vivoice": Source("capleaf/viVoice", "693e988", r"data/train-\d+-of-\d+\.parquet"),
    # Archives 86 and 98, drawn by random.Random(0).sample(range(74, 100), 2) from the 26 archives
    # not yet voiced or cached when they were held out (2026-11-09; 185 waits on its translation).
    "giga": Source("speechcolab/gigaspeech2", "8dc0d0e", r"data/vi/train/\d+\.tar\.gz",
                   heldout=frozenset({"86.parquet", "98.parquet"})),
    # YODAS3 (`espnet/yodas3` renamed): ~550 videos a tar, none in YODAS2 (training/tranche/yodas.py).
    "yodas": Source("espnet/yodas3.1", "3247e8cabda1e8e9418a6d8fe582e8c1abe75861", r"data/vi/audio/\d+\.tar"),
}
SCHEMA = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("doc", pa.string()), ("seq", pa.int64()),
    ("speaker", pa.string()), ("start_s", pa.float64()), ("end_s", pa.float64()),
    ("text", pa.string()), ("duration_s", pa.float64()), ("source_rate", pa.int32()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
])


def encode_opus(pcm: np.ndarray, rate: int) -> bytes:
    """Mono float32 -> Ogg Opus; libopus takes 8, 12, 16, 24 or 48 kHz input as it is."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="ogg") as out:
        stream = out.add_stream("libopus", rate=rate, layout="mono")
        stream.codec_context.thread_count = 1  # ffmpeg's default is a thread per host CPU
        stream.bit_rate = OPUS_BITRATE
        stream.options = {"compression_level": str(OPUS_COMPLEXITY)}
        frame = av.AudioFrame.from_ndarray(np.clip(pcm, -1, 1)[None].astype(np.float32), format="flt",
                                           layout="mono")
        frame.sample_rate = rate
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    return buf.getvalue()


def shard_files(name: str) -> list[str]:
    """The source's shards in numeric order, as `--shards` slices them."""
    from huggingface_hub import HfApi

    src = SOURCES[name]
    return sorted((f for f in HfApi().list_repo_files(src.repo, repo_type="dataset", revision=src.revision)
                   if re.fullmatch(src.pattern, f)), key=lambda f: [int(n) for n in re.findall(r"\d+", f)])


def segments(args: argparse.Namespace) -> None:
    from training.tranche import yodas

    yodas.segments(args)


def an_token() -> str:
    """The write token of the anquachdev account, which owns REPO and CACHE_REPO; the stages that
    push read it at their start, so a missing one stops them before any work."""
    return os.environ["HF_TOKEN_AN"]


def push(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=an_token())
    repo, folder, pattern = ((CACHE_REPO, args.root / "cache", f"{args.source}/*.pt") if args.part == "cache"
                             else (REPO, args.root, f"{args.source}/{args.part}/*.parquet"))
    # upload_folder never creates the repo, so a missing one fails instead of appearing ungated.
    if not api.dataset_info(repo).gated:
        raise SystemExit(f"{repo} is not gated: its sources may go only to those who accept their terms")
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=folder, allow_patterns=[pattern],
                      commit_message=f"{args.source} {args.part}")


def part_file(root: Path, source: str, part: str, name: str) -> Path:
    """A shard's part (segments, text_en, en_audio, asr_vi): the box's own copy under `root` where a
    stage wrote one (each writes whole, by a rename, and keeps it after pushing), else the Hub's (the
    HF cache keeps it). The box's downloads through Xet run ~5 MB/s (`HF_HUB_DISABLE_XET=1`: ~140 MB/s), and a second copy in
    the HF cache doubles a shard's disk."""
    local = root / source / part / name
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, f"{source}/{part}/{name}", repo_type="dataset"))


def push_part(root: Path, source: str, part: str) -> None:
    """`push`, tried again after a transient Hub error, for the stages that push as they go."""
    import time

    an_token()  # a missing token is not a Hub error: fail now, not after the retries
    for attempt in range(4):
        try:
            push(argparse.Namespace(root=root, source=source, part=part))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), f"pushed {source} {part}", flush=True)
            return
        except Exception as exc:  # noqa: BLE001 - the Hub's 5xx and timeouts come as many types
            # Printed here too: a push in a thread (`Pushes`) may never have its error read when the
            # stage fails first.
            again = f"again in {60 * 2 ** attempt} s" if attempt < 3 else "giving up"
            print(time.strftime("%Y-%m-%d %H:%M:%S"), f"push {source} {part} failed "
                  f"({type(exc).__name__}: {exc}); {again}", flush=True)
            if attempt == 3:
                raise
            time.sleep(60 * 2 ** attempt)


class Pushes:
    """A stage's pushes (`push_part` of each of its parts, in their order), in order in one thread
    beside its work: in the cache build's loop each held the GPU idle ~13 min at the box's ~5 MB/s
    to the Hub (2026-11-10). A push that failed its retries is raised by the next `push` or `check`,
    and by `wait`, which returns once the last push is in."""

    def __init__(self, root: Path, *parts: str) -> None:
        self.root, self.parts = root, parts
        self._thread = ThreadPoolExecutor(1, thread_name_prefix=f"push-{'-'.join(parts)}")
        self._pending: list[Future] = []

    def push(self, source: str) -> None:
        self.check()
        self._pending.append(self._thread.submit(self._push, source))

    def _push(self, source: str) -> None:
        for part in self.parts:
            push_part(self.root, source, part)

    def check(self) -> None:
        for future in [f for f in self._pending if f.done()]:
            self._pending.remove(future)
            future.result()

    def wait(self) -> None:
        self._thread.shutdown()
        for future in self._pending:
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("segments")
    p.add_argument("source", choices=["yodas"])
    p.add_argument("--shards", type=lambda s: slice(*(int(n) if n else None for n in s.split(":"))),
                   default=slice(None), help="a slice of the source's tars, e.g. 0:23")
    p.add_argument("--workers", type=int, default=None,
                   help="the cutting processes, the CPU quota less 3 by default")
    p.add_argument("--push", action=argparse.BooleanOptionalAction, default=True,
                   help="push each tar's segments and asr_vi as it is written")
    p = steps.add_parser("push")
    p.add_argument("source", choices=sorted(SOURCES))
    p.add_argument("--part", choices=["segments", "text_en", "en_audio", "asr_vi", "cache"], default="segments",
                   help="the folder to upload: the segments, their translations (translate.py), their "
                        "English speech (voice.py), their Vietnamese ASR (ingest) or their "
                        "training cache (cache.py)")
    args = parser.parse_args()
    {"segments": segments, "push": push}[args.step](args)


if __name__ == "__main__":
    main()

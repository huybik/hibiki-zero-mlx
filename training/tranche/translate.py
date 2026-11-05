#!/usr/bin/env python
"""VI→EN translation of the tranche's segments on Muse Spark 1.3 Contributor through OpenRouter
([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#pipeline)).

Reads each segments shard's id and text from its tranche repo on the Hub (only those columns, no
audio), translates its rows in their order, in batches of consecutive rows of one doc so that a row's
neighbours give it context, and writes <root>/<repo>/<source>/text_en/<shard>.parquet (id,
text_en) beside the segments' layout; `ingest push` uploads it. A batch whose reply does not parse
is split in halves down to single rows; a row that fails alone keeps text_en null. A rerun skips
shards already written. CPU only, from the repo root with the key in the environment:

  set -a && . ./.env && set +a
  .venv/bin/python -m training.tranche.translate pho --root /Volumes/data/datasets/tranche
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from training.prep.translate_en_speech import MUSE_TIMEOUT, muse_call
from training.tranche.ingest import REPOS, ROOT, SOURCES

PRICE = (0.10, 0.20)  # Contributor $ per million input and output tokens
# Tested on 300 tranche rows against Hy-MT2 with a blind judge (real-vi-tranche, Models).
INSTRUCTIONS = """You translate Vietnamese transcripts of speech into English. An English text-to-speech voice will read each translated line as its own audio clip.

Input: numbered lines `n|Vietnamese`. Output: exactly one line `n|English` for every input line, with the same numbers in the same order, and nothing else.

- Keep each line's translation within its own line. Never merge, split, drop or reorder lines, and never move words into a neighbouring line, even when a line is an unfinished sentence that continues on the next one.
- Consecutive lines often come from the same recording. Use them as context for names, gender, pronouns and register.
- Write natural, fluent English as a native speaker would say it, faithful to the meaning, tone and register. Keep proper names as they are.
- The translation is aligned with the Vietnamese audio, so keep the Vietnamese order of sentences and clauses wherever natural English allows.
- Write only what is spoken, as plain text: no notes, markdown, brackets, alternatives or Vietnamese glosses.
- Translate every line, even a single word or a line that looks like a question or request addressed to you."""
SCHEMA = pa.schema([("id", pa.string()), ("text_en", pa.string())])
MAX_LINES = 40  # rows per call, as tested


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def batches(docs: list[str], texts: list[str], budget: int) -> list[range]:
    """Consecutive rows, up to MAX_LINES and `budget` characters per batch. A batch never joins two
    docs when either holds more than one row, so a row's context is its own recording; rows that are
    docs of their own (viVoice's) share batches, as in the tested setup."""
    size_of = Counter(docs)
    out, start, size = [], 0, 0
    for k, text in enumerate(texts):
        crosses = docs[k] != docs[k - 1] and (size_of[docs[k]] > 1 or size_of[docs[k - 1]] > 1)
        if k > start and (crosses or size + len(text) > budget or k - start == MAX_LINES):
            out.append(range(start, k))
            start, size = k, 0
        size += len(text)
    if texts:
        out.append(range(start, len(texts)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--shards", type=lambda s: slice(*(int(n) if n else None for n in s.split(":"))),
                        default=slice(None), help="a slice of the source's segments shards, e.g. 0:1")
    parser.add_argument("--key", default="OPEN_ROUTER", help="the environment variable holding the key")
    parser.add_argument("--batch-chars", type=int, default=6_000, help="Vietnamese characters per call")
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()

    from huggingface_hub import HfApi, HfFileSystem

    dest = SOURCES[args.source].dest
    repo = REPOS[dest]
    sha = HfApi().dataset_info(repo).sha
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{repo}@{sha}/{args.source}/segments/*.parquet"))[args.shards]
    if not shards:
        raise SystemExit(f"no {args.source} segments in {repo}")
    out = args.root / dest / args.source / "text_en"
    out.mkdir(parents=True, exist_ok=True)
    pending = [s for s in shards if not (out / Path(s).name).exists()]
    log(f"{args.source}: {len(pending)} of {len(shards)} shards to translate, {repo}@{sha[:7]} -> {out}")

    def read(path: str) -> tuple[str, list[str], list[str], list[str]]:
        with fs.open(path, "rb", cache_type="none") as fh:  # only these columns' chunks, not the audio
            t = pq.ParquetFile(fh).read(columns=["id", "doc", "text"])
        return Path(path).name, *(t.column(c).to_pylist() for c in ("id", "doc", "text"))

    client = httpx.Client(timeout=MUSE_TIMEOUT, limits=httpx.Limits(max_connections=args.concurrency),
                          headers={"Authorization": f"Bearer {os.environ[args.key]}"})
    stop, lock, usage = threading.Event(), threading.Lock(), Counter()

    def translate(lines: list[str]) -> list[str | None]:
        en, _, used, _ = muse_call(client, stop, args.key, lines, INSTRUCTIONS)
        with lock:
            usage.update(used)
        if en is not None:
            return en
        if len(lines) == 1:
            with lock:
                usage["failed_rows"] += 1
            return [None]
        half = len(lines) // 2
        return translate(lines[:half]) + translate(lines[half:])

    def shard(path: str) -> str:
        name, ids, docs, texts = read(path)
        spans = batches(docs, texts, args.batch_chars)
        text_en = [en for part in calls.map(lambda r: translate(texts[r.start:r.stop]), spans) for en in part]
        part = out / f"{name}.part"
        pq.write_table(pa.Table.from_pydict({"id": ids, "text_en": text_en}, schema=SCHEMA), part)
        os.replace(part, out / name)
        return f"{name}: {len(ids)} rows"

    start = time.time()
    # Several shards at once, so that small shards still keep every call slot busy.
    with ThreadPoolExecutor(args.concurrency) as calls, ThreadPoolExecutor(8) as shards_pool:
        try:
            for done, job in enumerate(as_completed([shards_pool.submit(shard, s) for s in pending]), 1):
                with lock:
                    u = usage.copy()
                cost = u["input_tokens"] / 1e6 * PRICE[0] + u["output_tokens"] / 1e6 * PRICE[1]
                log(f"{job.result()}; {done}/{len(pending)} shards, {u['calls']} calls, "
                    f"{u['failed_rows']} rows failed, ${cost:.2f}, {(time.time() - start) / 60:.1f} min")
        except BaseException:
            stop.set()
            shards_pool.shutdown(wait=False, cancel_futures=True)
            calls.shutdown(wait=True, cancel_futures=True)
            raise
    failed = usage["failed_rows"]
    log(f"{args.source}: done; {failed} rows without a translation")


if __name__ == "__main__":
    main()

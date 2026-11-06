#!/usr/bin/env python
"""VI→EN translation of the tranche's segments on the box, by Index-Translate-9B that vLLM serves on
the GPU ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#pipeline)).

Reads each segments shard's id, doc and text from its tranche repo on the Hub (only those columns, no
audio), translates each row in its own call, with up to CONTEXT_ROWS earlier rows of its doc as
context, and writes <root>/<source>/text_en/<shard>.parquet (id, text_en) beside the segments'
layout; `ingest push` uploads it. A transcript all in capitals (GigaSpeech 2's) goes in lowercased:
kept, the model answers in capitals, which the TTS spells out. The first try is greedy; a reply cut
at max_tokens, still in Vietnamese or holding letters of another script is tried again sampled, up
to TRIES times; a row failing every try keeps text_en null, and a rerun translates the null and
bad rows again. The client needs no GPU; from the repo root, with the server up (`serve` prints
its command):

  python -m training.tranche.translate serve
  python -m training.tranche.translate giga --shards 51:
"""
from __future__ import annotations

import argparse
import os
import re
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from training.paths import exclusive
from training.prep.translate_en_speech import Stopped, broken
from training.tranche.ingest import REPO, ROOT, SOURCES

# Chosen on the box against Hy-MT2-30B-A3B and Muse (real-vi-tranche, Models). Apache-2.0.
MODEL = ("IndexTeam/Index-Translate-9B-FP8", "d467e031548df719a665509ce5bfc63c6a62af5c")
URL = "http://localhost:8200/v1/chat/completions"  # the server's model name is `mt`
TIMEOUT = 600  # seconds a call may wait: a full server queues calls behind its batch
MAX_TOKENS = 512  # a row's English takes ~20 tokens, the longest a few hundred
TRIES = 3  # per row and run: greedy, then sampled at TEMPERATURE
TEMPERATURE = 0.7
CONTEXT_ROWS = 2
PROMPT = ("Translate the following Vietnamese text into English. The text is a segment cut from speech and may "
          "start or stop mid-sentence: translate exactly what it says, leaving an unfinished sentence unfinished, "
          "without completing, adding or dropping anything. Write the English with normal capitalization and "
          "punctuation, and Vietnamese names without diacritics (Bắc Giang as Bac Giang). Write only what is "
          "spoken: no brackets, notes or explanations. Output only the translation.")
SCHEMA = pa.schema([("id", pa.string()), ("text_en", pa.string())])
# A run stops once more rows than this share of those it has started fail every try (past the
# first MIN_ROWS): a systematic failure would otherwise write every shard null. Checked at every
# failed row.
MAX_FAILED, MIN_ROWS = 0.01, 5_000


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def serve_command(gpu_memory: float) -> list[str]:
    """The vLLM server translate calls (env /workspace/envs/mt, `envs.sh`). The keep-alive outlasts
    httpx's own 5 s, so the server never closes a connection the client is about to reuse."""
    from huggingface_hub import snapshot_download

    return ["env", "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600", "/workspace/envs/mt/bin/vllm", "serve",
            snapshot_download(MODEL[0], revision=MODEL[1]), "--served-model-name", "mt", "--port", "8200",
            "--gpu-memory-utilization", str(gpu_memory), "--max-model-len", "4096", "--max-num-seqs", "512"]


def vietnamese(text: str) -> bool:
    """Whether most words carry Vietnamese diacritics or đ: a reply left untranslated. A name or two
    with its diacritics stays well below half."""
    words = text.split()
    marked = sum(bool(re.search(r"[̛̣̀́̃̉̂̆]", unicodedata.normalize("NFD", w)))
                 or "đ" in w.lower() for w in words)
    return marked > len(words) / 2


def bad(vi: str, en: str | None) -> bool:
    return en is None or not en.strip() or broken(vi, en) or vietnamese(en)


def call(client: httpx.Client, url: str, prompt: str, attempt: int) -> tuple[str | None, Counter]:
    """One row -> its English, or None when cut at MAX_TOKENS; and the usage. Thinking off; greedy
    on the first attempt, sampled (seeded by the attempt) after. The server is on this box, so any
    HTTP error stops the run rather than being retried."""
    sampling = {"temperature": 0} if attempt == 0 else {"temperature": TEMPERATURE, "seed": attempt}
    response = client.post(url, json={
        "model": "mt", "max_tokens": MAX_TOKENS, **sampling, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}]})
    response.raise_for_status()
    reply = response.json()
    choice, meta = reply["choices"][0], reply["usage"]
    en = (choice["message"]["content"] or "").strip() if choice["finish_reason"] == "stop" else None
    return en, Counter(calls=1, input_tokens=meta["prompt_tokens"], output_tokens=meta["completion_tokens"])


def prompt(texts: list[str], docs: list[str], k: int) -> str:
    """Row k's prompt, with up to CONTEXT_ROWS earlier rows of its doc (only rows of the same doc:
    viVoice's rows are docs of their own)."""
    context = [texts[j] for j in range(max(0, k - CONTEXT_ROWS), k) if docs[j] == docs[k]]
    background = ("\n\nEarlier in the same recording (context only, do not translate):\n" + "\n".join(context)
                  if context else "")
    return f"{PROMPT}{background}\n\nText to translate:\n{texts[k]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=sorted(SOURCES) + ["serve"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--shards", type=lambda s: slice(*(int(n) if n else None for n in s.split(":"))),
                        default=slice(None), help="a slice of the source's segments shards, e.g. 0:1")
    parser.add_argument("--url", default=URL, help="the vLLM server's chat completions")
    parser.add_argument("--concurrency", type=int, default=512, help="calls at once: the server's --max-num-seqs")
    parser.add_argument("--gpu-memory", type=float, default=0.7, help="serve: the server's share of the GPU")
    args = parser.parse_args()
    if args.source == "serve":
        print(" ".join(serve_command(args.gpu_memory)))
        return

    from huggingface_hub import HfApi, HfFileSystem

    repo = REPO
    sha = HfApi().dataset_info(repo).sha
    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{repo}@{sha}/{args.source}/segments/*.parquet"),
                    key=lambda f: [int(n) for n in re.findall(r"\d+", Path(f).name)])[args.shards]
    if not shards:
        raise SystemExit(f"no {args.source} segments in {repo}")
    out = args.root / args.source / "text_en"
    out.mkdir(parents=True, exist_ok=True)
    lock_file = exclusive(out / "translate.lock", f"another translate run is writing {out}")  # noqa: F841
    log(f"{args.source}: {len(shards)} shards, {repo}@{sha[:7]} -> {out}")

    def read(path: str) -> tuple[str, list[str], list[str], list[str]]:
        with fs.open(path, "rb", cache_type="none") as fh:  # only these columns' chunks, not the audio
            t = pq.ParquetFile(fh).read(columns=["id", "doc", "text"])
        ids, docs, texts = (t.column(c).to_pylist() for c in ("id", "doc", "text"))
        return Path(path).name, ids, docs, [text.lower() if text.isupper() else text for text in texts]

    client = httpx.Client(timeout=TIMEOUT, limits=httpx.Limits(max_connections=args.concurrency))
    stop, lock, usage = threading.Event(), threading.Lock(), Counter()

    def translate(vi: str, row_prompt: str) -> str | None:
        """-> the row's English, None when it fails every try."""
        for attempt in range(TRIES):
            if stop.is_set():
                raise Stopped
            en, used = call(client, args.url, row_prompt, attempt)
            with lock:
                usage.update(used)
            if not bad(vi, en):
                return en
        with lock:
            usage["failed_rows"] += 1
            if usage["rows"] > MIN_ROWS and usage["failed_rows"] > MAX_FAILED * usage["rows"]:
                raise SystemExit(f"{usage['failed_rows']:,} of {usage['rows']:,} rows started failed every "
                                 "try, a systematic failure: stopping")
        return None

    def shard(path: str) -> str:
        """Translate a shard's rows that have no good translation yet: all of them, or on a rerun
        those left null or bad."""
        name, ids, docs, texts = read(path)
        dst = out / name
        text_en = [None] * len(ids)
        if dst.exists():
            t = pq.read_table(dst)
            if t.column("id").to_pylist() != ids:
                raise SystemExit(f"{dst} holds other rows than its segments shard")
            text_en = t.column("text_en").to_pylist()
        todo = [k for k, (vi, en) in enumerate(zip(texts, text_en)) if bad(vi, en)]
        if not todo:
            return f"{name}: complete"
        with lock:
            usage["rows"] += len(todo)
        for k, en in zip(todo, calls.map(lambda k: translate(texts[k], prompt(texts, docs, k)), todo), strict=True):
            text_en[k] = en
        part = out / f"{name}.part"
        pq.write_table(pa.Table.from_pydict({"id": ids, "text_en": text_en}, schema=SCHEMA), part)
        os.replace(part, dst)
        with lock:
            usage["rows_done"] += len(todo)
        return f"{name}: {len(todo)} of {len(ids)} rows translated"

    start = time.time()
    # Several shards at once, so that each shard's last rows don't leave call slots idle.
    with ThreadPoolExecutor(args.concurrency) as calls, ThreadPoolExecutor(4) as shards_pool:
        try:
            for done, job in enumerate(as_completed([shards_pool.submit(shard, s) for s in shards]), 1):
                with lock:
                    u = usage.copy()
                minutes = (time.time() - start) / 60
                log(f"{job.result()}; {done}/{len(shards)} shards, {u['rows_done'] / 60 / minutes:.0f} rows/s, "
                    f"{u['calls']} calls, {u['failed_rows']} rows failed, {u['input_tokens'] / 1e6:.1f}M tokens "
                    f"in, {u['output_tokens'] / 1e6:.1f}M out, {minutes:.1f} min")
        except BaseException:
            stop.set()
            shards_pool.shutdown(wait=False, cancel_futures=True)
            calls.shutdown(wait=True, cancel_futures=True)
            raise
    failed = usage["failed_rows"]
    log(f"{args.source}: done; {failed} rows without a translation")


if __name__ == "__main__":
    main()

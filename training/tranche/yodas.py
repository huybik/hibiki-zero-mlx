#!/usr/bin/env python
"""YODAS3's Vietnamese as a tranche source (`ingest segments yodas`): rows cut from its videos at
their captions' word times, transcribed by Qwen3-ASR
([tranche-yodas](../../docs/agent-memory/tranche-yodas.md)).

Each audio tar of `espnet/yodas3.1`'s data/vi (~550 videos, ~157 h, a <video>.json and <video>.webm
each) is downloaded while the one before it is cut. A video's captions are YouTube's ASR: unpunctuated
and often wrong, but timed word by word. So they only place the rows: runs of words no more than
RUN_GAP_S apart and with no tag ("[âm nhạc]") between them, the longest first up to CAP_S a video
(a longer run cut at a pause), are decoded alone and cut into rows of 5-22.4 s at the quietest point
of the widest pause between two words. Every row is
heard by the ASR server (`filters.Hearer`), whose transcript becomes the row's `text`; its CER against
the caption is the row's check. Written per tar, both pushed as they come:
  <root>/yodas/segments/<tar>.parquet   ingest's schema; doc the video, speaker none
  <root>/yodas/asr_vi/<tar>.parquet     filters.py's part: the transcript again, its language, and
                                        its CER against the caption (the gates' `cer`)
keeping the rows `filters.vi_gate` passes (the caption standing as the transcript).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import subprocess
import tarfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np

RATE = 24_000  # Mimi's rate: the rows are stored at it
ASR_RATE = 16_000
CAP_S = 1200.0  # a video's rows at most: many voices and topics rather than a few long videos
RUN_GAP_S = 2.0  # word starts further apart end a run
MIN_PAUSE_S = 0.33  # a cut needs this between two word starts: a word's own length (~0.29 s at 3.4 syllables/s) plus a gap
WORD_S = 0.2  # a cut's window starts this long after the earlier word's start
LEAD_S = (0.5, 0.02)  # a run's first row starts at the quietest point this long before to just before its first word
TAIL_S = (0.25, 1.2)  # its last row ends at the quietest point this long after its last word starts
FRAME, HOP = 480, 240  # 20 ms energy frames, 10 ms apart, at RATE
SEEK_S = 1.0  # a run's audio is decoded from this long before its span: the Opus decoder settles within 80 ms of a seek
TAG = re.compile(r"^\s*(\[[^\]]*\]\s*)+$")  # "[âm nhạc]", "[tiếng cười]", "[ __ ]": no speech, or censored
MAX_TAG_SHARE = 0.2  # videos whose cues are more often tags are music, not speech
IN_FLIGHT = 2  # videos queued per cutting process
ASR_IN_FLIGHT = 256  # the server's batch (--max-num-seqs): 78 rows/s at 64 in flight, 104 at 256, no more at 512


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def caption(meta: dict) -> tuple[list[tuple[float, str]], list[float], int]:
    """A video's caption words as (start in seconds, word), the starts of its tags, and its cue count."""
    words, tags, cues = [], [], meta["transcript"] or []
    for cue in cues:
        if TAG.match(cue["text"]):
            tags.append(cue["start"] / 1000)
        for w in cue.get("words") or []:
            t, word = (cue["start"] + w["t"]) / 1000, w["w"].strip()
            if TAG.match(word):
                tags.append(t)
            elif word:
                words.append((t, word))
    words.sort(key=lambda w: w[0])
    return words, sorted(tags), len(cues)


def runs_of(words: list[tuple[float, str]], tags: list[float]) -> list[list[tuple[float, str]]]:
    """The caption's runs: words no more than RUN_GAP_S apart with no tag between them."""
    runs, k = [], 0
    for i, (t, w) in enumerate(words):
        while k < len(tags) and tags[k] <= t:
            k += 1
        tag_between = i and k > 0 and tags[k - 1] > words[i - 1][0]
        if not runs or t - words[i - 1][0] > RUN_GAP_S or tag_between:
            runs.append([])
        runs[-1].append((t, w))
    return runs


def capped(run: list[tuple[float, str]], room: float) -> tuple[list[tuple[float, str]], float | None]:
    """A run of more than `room` seconds (a livestream can be hours of unbroken words) cut before the
    last word after a pause that keeps it within room (or the first such word past it), and the
    start of that word, which bounds the run's end; a shorter run whole, with no bound."""
    t0 = run[0][0]
    if run[-1][0] - t0 <= room:
        return run, None
    pauses = [j for j in range(1, len(run)) if run[j][0] - run[j - 1][0] >= MIN_PAUSE_S]
    within = [j for j in pauses if run[j][0] - t0 <= room]
    j = within[-1] if within else pauses[0] if pauses else None
    return (run, None) if j is None else (run[:j], run[j][0])


def quietest(pcm: np.ndarray, off: float, a: float, b: float) -> float:
    """The middle of the quietest 20 ms frame between a and b seconds of pcm, audio from `off` seconds."""
    i, j = max(0, int((a - off) * RATE)), min(len(pcm), int((b - off) * RATE))
    if j - i < FRAME:
        return (a + b) / 2
    n = (j - i - FRAME) // HOP + 1
    frames = pcm[i:i + (n - 1) * HOP + FRAME]
    energy = np.lib.stride_tricks.sliding_window_view(frames, FRAME)[::HOP]
    return off + (i + int((energy ** 2).mean(axis=1).argmin()) * HOP + FRAME // 2) / RATE


def cut_run(pcm: np.ndarray, off: float, run: list[tuple[float, str]], lo: float, hi: float,
            row_s: tuple[float, float]) -> list[tuple[float, float, str]]:
    """One run's rows (start, end, caption words) within [lo, hi], from its audio pcm (from `off`
    seconds): each ends at the widest pause that leaves it row_s long, preferring cuts that strand
    no tail shorter than a row."""
    start = max(0.0, off, quietest(pcm, off, max(lo, run[0][0] - LEAD_S[0]), run[0][0] - LEAD_S[1]))
    end = quietest(pcm, off, run[-1][0] + TAIL_S[0], min(hi, run[-1][0] + TAIL_S[1]))
    t = [w[0] for w in run]
    rows, s, i0 = [], start, 0
    while True:
        if end - s <= row_s[1]:
            if end - s >= row_s[0]:
                rows.append((s, end, " ".join(w for _, w in run[i0:])))
            return rows
        best, best_score = None, -1.0
        for j in range(i0 + 1, len(run)):  # a cut before word j falls in [t[j - 1] + WORD_S, t[j]]
            if t[j - 1] + WORD_S - s > row_s[1]:
                break
            pause = t[j] - t[j - 1]
            if pause < MIN_PAUSE_S or t[j - 1] + WORD_S - s < row_s[0] or t[j] - s > row_s[1]:
                continue
            score = pause - (1.0 if 0 < end - t[j] < row_s[0] else 0.0)
            if score > best_score:
                best, best_score = j, score
        if best is None:  # no pause in reach: skip to the next one and start a row there
            nxt = next((j for j in range(i0 + 1, len(run)) if t[j] - s > row_s[1] and t[j] - t[j - 1] >= MIN_PAUSE_S), None)
            if nxt is None:
                return rows
            s, i0 = quietest(pcm, off, t[nxt - 1] + WORD_S, t[nxt]), nxt
            continue
        c = quietest(pcm, off, t[best - 1] + WORD_S, t[best])
        rows.append((s, c, " ".join(w for _, w in run[i0:best])))
        s, i0 = c, best


def decode(container, stream, a: float, b: float) -> tuple[np.ndarray, float]:
    """A video's audio from SEEK_S before a to just past b seconds, mono at RATE, and the time its
    first sample is at: only the chosen runs are decoded, not the hours of video around them.
    Resampled by scipy: PyAV's AudioResampler starts a thread per host CPU, as voice.decode_16k says."""
    from scipy.signal import resample_poly

    container.seek(int(max(0.0, a - SEEK_S) / stream.time_base), stream=stream)
    chunks, off = [], None
    for frame in container.decode(stream):
        off = frame.time if off is None else off
        chunks.append(frame.to_ndarray())
        if frame.time > b:
            break
    if not chunks:
        return np.zeros(0, np.float32), a
    pcm = np.concatenate(chunks, axis=-1)
    rate = stream.codec_context.sample_rate
    g = np.gcd(RATE, rate)
    pcm = pcm.mean(axis=0) if pcm.shape[0] > 1 else pcm[0]
    return resample_poly(pcm, RATE // g, rate // g).astype(np.float32), off


def wav_16k(pcm: np.ndarray) -> bytes:
    """A row at RATE -> 16 kHz WAV for the ASR server, as filters.wav_16k sends."""
    import soundfile as sf
    from scipy.signal import resample_poly

    buf = io.BytesIO()
    sf.write(buf, resample_poly(pcm, 2, 3).astype(np.float32), ASR_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def video_rows(tar: str, index: int, meta: dict, webm: bytes, row_s: tuple[float, float]) -> tuple[list[dict], str]:
    """A video's rows, each with its Ogg Opus (`audio`) and the 16 kHz WAV the ASR hears (`wav`),
    and why it gives none ("" when it gives some). In a cutting process."""
    import av

    from training.tranche.ingest import encode_opus

    words, tags, cues = caption(meta)
    if not words:
        return [], "no caption words"
    if len(tags) > MAX_TAG_SHARE * cues:
        return [], "music"
    runs = [r for r in runs_of(words, tags) if r[-1][0] - r[0][0] + TAIL_S[0] >= row_s[0]]
    if not runs:
        return [], "no run long enough"
    # The longest runs first, until the video's cap.
    order = sorted(range(len(runs)), key=lambda k: runs[k][0][0] - runs[k][-1][0])
    chosen, total = {}, 0.0  # run index -> the run, cut to the cap, and the bound of its end
    for k in order:
        if total >= CAP_S:
            break
        run, bound = capped(runs[k], CAP_S - total + row_s[1])
        chosen[k] = run, bound
        total += run[-1][0] - run[0][0]
    cuts = []  # in time order: the runs' and each run's rows
    with av.open(io.BytesIO(webm)) as container:
        stream = container.streams.audio[0]
        stream.codec_context.thread_count = 1  # ffmpeg's default is a thread per host CPU
        for k in sorted(chosen):
            run, bound = chosen[k]
            lo = runs[k - 1][-1][0] + TAIL_S[0] if k else 0.0
            hi = bound if bound is not None else runs[k + 1][0][0] - LEAD_S[0] if k + 1 < len(runs) else math.inf
            pcm, off = decode(container, stream, max(lo, run[0][0] - LEAD_S[0]), min(hi, run[-1][0] + TAIL_S[1]))
            hi = min(hi, off + len(pcm) / RATE)  # a video shorter than its captions
            cuts += [(a, b, text, pcm, off) for a, b, text in cut_run(pcm, off, run, lo, hi, row_s)]
    rows, kept = [], 0.0
    for a, b, text, pcm, off in cuts:
        if kept >= CAP_S:
            break
        clip = pcm[int((a - off) * RATE):int((b - off) * RATE)]
        if len(clip) < row_s[0] * RATE:  # a video shorter than its captions
            continue
        kept += len(clip) / RATE
        k = len(rows)
        rows.append({"id": f"yodas/{tar}/{meta['id']}/{k:03d}", "source": "yodas", "doc": f"yodas/{tar}/{meta['id']}",
                     "seq": index * 10_000 + k, "speaker": None, "start_s": round(a, 3), "end_s": round(b, 3),
                     "caption": text, "duration_s": round(len(clip) / RATE, 3), "source_rate": RATE,
                     "audio": {"bytes": encode_opus(clip, RATE), "path": f"yodas_{tar}_{meta['id']}_{k:03d}.opus"},
                     "wav": wav_16k(clip)})
    return rows, "" if rows else "no row in its runs"


def download(url: str, dst: Path) -> Path:
    """curl that drops a connection crawling under 1 MB/s for 30 s and resumes on a new one: some of
    the box's Hub connections crawl at ~50 KB/s and never time out (real-vi-tranche.md)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    for attempt in range(20):
        done = subprocess.run(["curl", "-sSfL", "--speed-limit", "1000000", "--speed-time", "30", "-C", "-",
                               "-o", str(part), url]).returncode == 0
        if done:
            part.replace(dst)
            return dst
        log(f"{dst.name}: download attempt {attempt + 1} failed; again")
        time.sleep(5)
    raise RuntimeError(f"{url}: 20 download attempts failed")


def hear_rows(rows: list[dict], hearer, pool: ThreadPoolExecutor) -> list[Future]:
    return [pool.submit(hearer.hear, row.pop("wav")) for row in rows]


def tar_shard(tar: Path, cutters: ProcessPoolExecutor, workers: int, hearer, asr_pool: ThreadPoolExecutor,
              finisher: ThreadPoolExecutor, root: Path, row_s: tuple[float, float]) -> Future[str]:
    """Cut one downloaded tar and send its rows to the ASR. Returns once it is cut, with the future
    (in `finisher`) that gates the heard rows, writes the tar's segments and asr_vi parquets and
    gives its stats line: the next tar is cut while the ASR hears this one's last rows."""
    name = tar.stem
    began = time.monotonic()
    videos, why = 0, Counter()
    pending: set[Future] = set()
    heard: list[tuple[dict, Future]] = []

    def settle(finished: set[Future]) -> None:
        for done in finished:
            rows, reason = done.result()
            why[reason or "rows"] += 1
            heard.extend(zip(rows, hear_rows(rows, hearer, asr_pool), strict=True))

    with tarfile.open(tar, "r|") as members:
        meta = None
        for member in members:
            data = members.extractfile(member).read()
            if member.name.endswith(".json"):
                meta = json.loads(data)
                continue
            if meta is None or Path(member.name).stem != meta["id"]:
                raise ValueError(f"{tar.name}: {member.name} has no metadata before it")
            pending.add(cutters.submit(video_rows, name, videos, meta, data, row_s))
            videos, meta = videos + 1, None
            # Whichever video is done first: an hours-long one ahead would idle the other cutters.
            while len(pending) >= IN_FLIGHT * workers:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                settle(finished)
    settle(wait(pending).done)
    cut_s = time.monotonic() - began
    return finisher.submit(finish_tar, name, root, heard, videos, why, began, cut_s)


def finish_tar(name: str, root: Path, heard: list[tuple[dict, Future]], videos: int, why: Counter,
               began: float, cut_s: float) -> str:
    import jiwer
    import pyarrow as pa
    import pyarrow.parquet as pq

    from training.tranche.filters import spoken, vi_gate
    from training.tranche.ingest import SCHEMA

    kept, asr, reasons, cers = [], [], Counter(), []
    for row, future in heard:
        language, text = future.result()
        words = row.pop("caption")
        ref, hyp = " ".join(spoken(words)), " ".join(spoken(text))
        cer = jiwer.cer(ref, hyp) if ref else float(bool(hyp))
        cers.append(cer)
        gate = {"language": language, "cer": cer, "asr_vi": text, "text": words}
        if reason := vi_gate(gate):
            reasons["magnitudes" if reason.startswith("the ASR") else reason.split(" ")[0]] += 1
            continue
        kept.append(row | {"text": text})
        asr.append({"id": row["id"], "asr_vi": text, "language": language, "cer": cer})
    kept.sort(key=lambda r: r["seq"])
    order = {r["id"]: k for k, r in enumerate(kept)}
    asr.sort(key=lambda r: order[r["id"]])
    for part, table in (("segments", pa.Table.from_pylist(kept, schema=SCHEMA)),
                        ("asr_vi", pa.table({"id": [r["id"] for r in asr],
                                             "asr_vi": pa.array([r["asr_vi"] for r in asr], pa.string()),
                                             "language": pa.array([r["language"] for r in asr], pa.string()),
                                             "cer": pa.array([r["cer"] for r in asr], pa.float32())}))):
        dst = root / "yodas" / part / f"{name}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(dst)
    hours = sum(r["duration_s"] for r in kept) / 3600
    cers.sort()
    q = lambda p: cers[min(len(cers) - 1, int(p * len(cers)))] if cers else float("nan")  # noqa: E731
    return (f"{name}: {videos} videos ({dict(why)}), {len(heard)} rows cut, {len(kept)} kept ({hours:.2f} h), "
            f"dropped {dict(reasons)}; CER p25/p50/p75/p90 {q(.25):.3f}/{q(.5):.3f}/{q(.75):.3f}/{q(.9):.3f}; "
            f"cut in {cut_s:.0f} s, all in {time.monotonic() - began:.0f} s")


def segments(args: argparse.Namespace) -> None:
    """Every picked tar not yet written: one is cut while the next downloads and the one before is
    heard to its end and written."""
    import os

    from training.paths import cpu_budget
    from training.tranche.filters import ASR_URL, Hearer
    from training.tranche.ingest import ROW_S, SOURCES, Pushes, an_token, shard_files

    if args.push:
        an_token()
    src = SOURCES["yodas"]
    files = shard_files("yodas")
    picked = [files[i] for i in range(len(files))[args.shards]]
    todo = [f for f in picked if not (args.root / "yodas" / "segments" / f"{Path(f).stem}.parquet").exists()]
    log(f"yodas: {len(todo)} of {len(picked)} picked tars to cut, of {len(files)}")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    workers = args.workers or max(1, int(cpu_budget()) - 3)  # the main process, the ASR client and the pushes
    cutters = ProcessPoolExecutor(workers, mp_context=get_context("spawn"))
    asr_pool = ThreadPoolExecutor(ASR_IN_FLIGHT)
    hearer = Hearer(ASR_URL, ASR_IN_FLIGHT)
    fetch = ThreadPoolExecutor(1, thread_name_prefix="download")
    finisher = ThreadPoolExecutor(1, thread_name_prefix="finish")
    scratch = args.root / "yodas" / ".download"
    url = lambda f: f"https://huggingface.co/datasets/{src.repo}/resolve/{src.revision}/{f}"  # noqa: E731
    pushes = {part: Pushes(args.root, part) for part in ("segments", "asr_vi")}

    def written(done: Future[str]) -> None:
        log(done.result())
        if args.push:
            for p in pushes.values():
                p.push("yodas")

    try:
        nxt = fetch.submit(download, url(todo[0]), scratch / Path(todo[0]).name) if todo else None
        last = None  # the tar before, heard and written while this one is cut
        for k, f in enumerate(todo):
            tar = nxt.result()
            nxt = fetch.submit(download, url(todo[k + 1]), scratch / Path(todo[k + 1]).name) if k + 1 < len(todo) else None
            cut = tar_shard(tar, cutters, workers, hearer, asr_pool, finisher, args.root, ROW_S)
            tar.unlink()
            if last:
                written(last)
            last = cut
        if last:
            written(last)
        for p in pushes.values():
            p.wait()
    finally:
        cutters.shutdown(wait=False, cancel_futures=True)
        asr_pool.shutdown(wait=False, cancel_futures=True)
        fetch.shutdown(wait=False, cancel_futures=True)
        finisher.shutdown(wait=False, cancel_futures=True)

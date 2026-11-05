#!/usr/bin/env python
"""Row filters of the tranche, measured before the cache build drops rows on them
([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#filters)).

  overlap            count the segments rows that share a run of OVERLAP_N syllables with a FLEURS vi
                     or MultiMed-ST vi evaluation sentence (data/eval/, built by
                     training.prep.eval_sets); the cache build drops those rows with `overlaps`.
                     Reads only the id and text columns from the Hub; runs on the Mac or the box
  asr <sources...>   hear every segments row with Qwen3-ASR-1.7B on the ASR server, its language
                     detected rather than forced, and write
                     <root>/<public|restricted>/<source>/asr_vi/<shard> (id, asr_vi, language, cer
                     against the corpus transcript, over checks.words); pushes each source's
                     folder every 30 min and at its end. A rerun skips the shards written or pushed,
                     and first pushes any written but not yet pushed

From the repo root, with the HF token in the environment; `asr` on the box with the ASR server up
(see voice.py):
  python -m training.tranche.filters overlap
  /workspace/envs/nanovox/bin/python -m training.tranche.filters asr pho vivoice giga
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from training.tranche.checks import words
from training.tranche.ingest import REPOS, ROOT, SOURCES, push_part

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SETS = ("fleurs-vi/dev", "fleurs-vi/test", "multimed-vi/select", "multimed-vi/final")
# Syllables. Over the first tranche, no row shares a 13-syllable run with an evaluation sentence;
# runs of 10 or fewer match only stock broadcast phrases ("cảm ơn quý vị và các bạn đã quan tâm", 80 rows).
OVERLAP_N = 13
ASR_URL = "http://localhost:8100/v1/audio/transcriptions"
ASR_MAX_TOKENS = 512  # as voice.py: a looping transcript comes back long instead of running on
ASR_TIMEOUT_S = 60
CONCURRENCY = 64
DECODERS = 4  # processes decoding the clips for the server, a few cores beside the TTS run
PUSH_S = 1800


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def eval_grams(eval_dir: Path = REPO_ROOT / "data/eval") -> set[tuple[str, ...]]:
    """Every OVERLAP_N-syllable run of the evaluation sentences. A shorter sentence has none: the
    few there are (9 of 4,655 under 9 syllables, "được", "alo", "vùng nông thôn") would match
    everyday speech, not leaked content."""
    import json

    grams = set()
    for name in EVAL_SETS:
        for line in open(eval_dir / f"{name}.jsonl"):
            w = words(json.loads(line)["text_vi"])
            grams.update(tuple(w[i:i + OVERLAP_N]) for i in range(len(w) - OVERLAP_N + 1))
    return grams


def overlaps(text: str, grams: set[tuple[str, ...]]) -> bool:
    """Whether the transcript shares an OVERLAP_N-syllable run with an evaluation sentence."""
    w = words(text)
    return any(tuple(w[i:i + OVERLAP_N]) in grams for i in range(len(w) - OVERLAP_N + 1))


def overlap(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    grams = eval_grams()
    fs, api = HfFileSystem(), HfApi()

    def read(path: str) -> list[tuple[str, str]]:
        with fs.open(path, "rb", cache_type="none") as fh:  # only these columns' chunks, not the audio
            t = pq.ParquetFile(fh).read(columns=["id", "text"])
        return list(zip(*t.to_pydict().values()))

    for source, spec in SOURCES.items():
        repo = REPOS[spec.dest]
        sha = api.dataset_info(repo).sha
        shards = sorted(fs.glob(f"datasets/{repo}@{sha}/{source}/segments/*.parquet"))
        with ThreadPoolExecutor(16) as pool:
            rows = [r for part in pool.map(read, shards) for r in part]
        hits = [i for i, text in rows if overlaps(text, grams)]
        log(f"{source}: {len(hits)} of {len(rows)} rows overlap the evaluation sets" + (f", e.g. {hits[:5]}" if hits else ""))


def wav_16k(data: bytes) -> bytes:
    """A segment's Ogg Opus -> 16 kHz WAV for the server. Sent as Opus, the server decodes it at
    ~0.7 CPU-seconds a clip (all 8 API processes saturated the box's 27 cores at 21 rows/s)."""
    import io

    import soundfile as sf

    from training.tranche.voice import ENCODE_RATE, decode_16k

    buf = io.BytesIO()
    sf.write(buf, decode_16k(data), ENCODE_RATE, subtype="PCM_16", format="WAV")
    return buf.getvalue()


class Hearer:
    """One HTTP client to the ASR server; each call sends one clip as 16 kHz WAV."""

    def __init__(self, url: str):
        import httpx

        self.url = url
        self.client = httpx.Client(timeout=ASR_TIMEOUT_S, limits=httpx.Limits(max_connections=CONCURRENCY))

    def hear(self, audio: bytes) -> tuple[str, str]:
        """WAV bytes -> (language, transcript). Unforced, the model writes `language <Name><asr_text><text>`,
        and `language None` for audio with no speech."""
        import httpx

        for attempt in range(3):  # the server drops an idle keep-alive connection now and then
            try:
                reply = self.client.post(self.url, files={"file": ("a.wav", audio, "audio/wav")},
                                         data={"model": "asr", "max_completion_tokens": ASR_MAX_TOKENS})
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
        reply.raise_for_status()
        head, tag, text = reply.json()["text"].partition("<asr_text>")
        if not tag or not head.startswith("language "):
            raise ValueError(f"unexpected ASR output: {head[:200]!r}")
        language = head.removeprefix("language ").strip()
        return ("", "") if language == "None" else (language, text.strip())


def asr_shard(source: str, name: str, root: Path, hearer: Hearer, decoders, pool: ThreadPoolExecutor) -> Path:
    import jiwer
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    dest = SOURCES[source].dest
    seg = pq.read_table(hf_hub_download(REPOS[dest], f"{source}/segments/{name}", repo_type="dataset"),
                        columns=["id", "text", "audio"])
    audio = [a["bytes"] for a in seg.column("audio").to_pylist()]
    heard = list(pool.map(hearer.hear, decoders.map(wav_16k, audio, chunksize=32)))
    cer = []
    for text, (_, asr) in zip(seg.column("text").to_pylist(), heard):
        ref, hyp = " ".join(words(text)), " ".join(words(asr))
        cer.append(jiwer.cer(ref, hyp) if ref else float(bool(hyp)))
    table = pa.table({"id": seg.column("id"), "asr_vi": pa.array([h[1] for h in heard], pa.string()),
                      "language": pa.array([h[0] for h in heard], pa.string()),
                      "cer": pa.array(cer, pa.float32())})
    dst = root / dest / source / "asr_vi" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    pq.write_table(table, part)
    part.replace(dst)
    return dst


def asr(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    api = HfApi()
    todo = []
    for source in args.sources:
        files = api.list_repo_files(REPOS[SOURCES[source].dest], repo_type="dataset")
        done = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/asr_vi/")}
        folder = args.root / SOURCES[source].dest / source / "asr_vi"
        todo += [(source, f.rsplit("/", 1)[1]) for f in sorted(files) if f.startswith(f"{source}/segments/")
                 and f.rsplit("/", 1)[1] not in done and not (folder / f.rsplit("/", 1)[1]).exists()]
        if {p.name for p in folder.glob("*.parquet")} - done:  # written by an earlier run that stopped before pushing them
            push_part(args.root, source, "asr_vi")
    todo = todo[: args.limit]
    log(f"{len(todo)} shards to hear: " + ", ".join(f"{s} {sum(t[0] == s for t in todo)}" for s in args.sources))
    import os
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # the decoders' thread pools, under the box's thread limit (voice.py)
    decoders = ProcessPoolExecutor(DECODERS, mp_context=get_context("spawn"))
    hearer = Hearer(args.asr)
    pushed = {s: time.monotonic() for s in args.sources}
    began, rows = time.monotonic(), 0
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        for k, (source, name) in enumerate(todo):
            t = pq.read_table(asr_shard(source, name, args.root, hearer, decoders, pool))
            rows += t.num_rows
            languages = Counter(t.column("language").to_pylist())
            cer = sorted(t.column("cer").to_pylist())
            log(f"{source}/{name}: {t.num_rows} rows, CER median {cer[len(cer) // 2]:.3f}, p90 "
                f"{cer[len(cer) * 9 // 10]:.3f}, not Vietnamese {t.num_rows - languages['Vietnamese']}; "
                f"{rows / (time.monotonic() - began):.0f} rows/s")
            last = k + 1 == len(todo) or todo[k + 1][0] != source
            if last or time.monotonic() - pushed[source] > PUSH_S:
                push_part(args.root, source, "asr_vi")
                pushed[source] = time.monotonic()
    decoders.shutdown()
    log(f"done: {rows} rows in {(time.monotonic() - began) / 3600:.2f} h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("overlap")
    p = steps.add_parser("asr")
    p.add_argument("sources", nargs="+", choices=["pho", "vivoice", "giga"])
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--asr", default=ASR_URL)
    p.add_argument("--limit", type=int, help="hear only this many shards (a staged start)")
    args = parser.parse_args()
    {"overlap": overlap, "asr": asr}[args.step](args)


if __name__ == "__main__":
    main()

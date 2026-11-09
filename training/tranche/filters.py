#!/usr/bin/env python
"""Row filters of the tranche, measured before the cache build drops rows on them
([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#filters)).

  overlap            count the segments rows (held-out shards excepted, `training_texts`) that share
                     a run of OVERLAP_N syllables (or a whole shorter sentence) with a FLEURS vi or
                     MultiMed-ST vi evaluation sentence (data/eval/, built by training.prep.eval_sets);
                     the cache build drops those rows with `overlaps`.
                     Reads only the id and text columns from the Hub; runs on the Mac or the box
  asr <sources...>   hear every segments row with Qwen3-ASR-1.7B on the ASR server, its language
                     detected rather than forced, and write
                     <root>/<source>/asr_vi/<shard> (id, asr_vi, language, cer
                     against the corpus transcript, over `spoken`); pushes each source's
                     folder every 30 min and at its end. A rerun skips the shards written or pushed,
                     and first pushes any written but not yet pushed

From the repo root, with HF_TOKEN_AN (pushes) in the environment; `asr` on the box with the ASR server up
(see voice.py):
  python -m training.tranche.filters overlap
  /workspace/envs/nanovox/bin/python -m training.tranche.filters asr pho vivoice giga
"""
from __future__ import annotations

import argparse
import os
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path

from training.tranche.ingest import REPO, ROOT, SOURCES, push_part

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_SETS = ("fleurs-vi/dev", "fleurs-vi/test", "multimed-vi/select", "multimed-vi/final")
# Syllables. Over the first tranche, no row shares a 13-syllable run with an evaluation sentence;
# runs of 10 or fewer match only stock broadcast phrases ("cảm ơn quý vị và các bạn đã quan tâm", 80 rows).
OVERLAP_N = 13
# An evaluation sentence of MIN_SENTENCE..OVERLAP_N - 1 syllables is matched whole; shorter ones
# ("được", "alo", "vùng nông thôn") would match everyday speech, not leaked content.
MIN_SENTENCE = 9
# Spellings the corpora mix for the same number word, folded to num2words' choice.
NUMBER_VARIANTS = {"ngàn": "nghìn", "linh": "lẻ", "nhăm": "lăm"}
ASR_URL = "http://localhost:8100/v1/audio/transcriptions"
ASR_MAX_TOKENS = 512  # as voice.py: a looping transcript comes back long instead of running on
ASR_TIMEOUT_S = 60
CONCURRENCY = 64
DECODER_SHARE = 6  # processes decoding the clips for the server: a sixth of the CPU quota (4 of 26), beside the TTS run
PUSH_S = 1800


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def spoken(text: str) -> list[str]:
    """Lower-case NFC syllables without punctuation, numbers written out as Vietnamese words:
    PhoAudiobook and GigaSpeech 2 spell numbers, while Qwen3-ASR and FLEURS write digits (on
    PhoAudiobook, digit rows were 48% of those over 0.1 CER before this)."""
    from num2words import num2words

    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"(?<=\d)[.,](?=\d{3}\b)", "", text)  # 1.500.000: thousands separators
    text = re.sub(r"\d+", lambda m: f" {num2words(int(m.group()), lang='vi')} ", text)
    return [NUMBER_VARIANTS.get(w, w) for w in re.sub(r"[^\w\s]", " ", text).split()]


Grams = dict[int, set[tuple[str, ...]]]  # evaluation runs by length


def runs(words: list[str]) -> set[tuple[str, ...]]:
    """A sentence's runs as the overlap check matches them: every OVERLAP_N-syllable run, or the
    whole sentence of MIN_SENTENCE..OVERLAP_N - 1 syllables; none for a shorter one."""
    if len(words) < MIN_SENTENCE:
        return set()
    n = min(len(words), OVERLAP_N)
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def grams_of(texts: Iterable[str]) -> Grams:
    """The runs of these sentences, by length."""
    grams: Grams = {}
    for text in texts:
        if found := runs(spoken(text)):
            grams.setdefault(len(next(iter(found))), set()).update(found)
    return grams


def eval_grams(eval_dir: Path = REPO_ROOT / "data/eval") -> Grams:
    """The runs of the evaluation sentences."""
    import json

    return grams_of(json.loads(line)["text_vi"] for name in EVAL_SETS for line in open(eval_dir / f"{name}.jsonl"))


def overlaps(text: str, grams: Grams) -> bool:
    """Whether the transcript holds one of the runs."""
    w = spoken(text)
    return any(tuple(w[i:i + n]) in found for n, found in grams.items() for i in range(len(w) - n + 1))


def matches(text: str, grams: Grams) -> set[tuple[str, ...]]:
    """The runs the transcript holds."""
    w = spoken(text)
    return {run for n, found in grams.items() for i in range(len(w) - n + 1) if (run := tuple(w[i:i + n])) in found}


def training_texts(revision: str) -> Iterator[tuple[str, list[tuple[str, str, str]]]]:
    """(source, its (id, doc, text) rows in their order) for each segments shard of REPO at
    `revision` but the held-out ones (`Source.heldout`), sixteen shards read at a time; only those
    columns' chunks are read, not the audio. A doc's rows are consecutive (ingest)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def read(path: str) -> list[tuple[str, str, str]]:
        with fs.open(path, "rb", cache_type="none") as fh:
            t = pq.ParquetFile(fh).read(columns=["id", "doc", "text"])
        return list(zip(*t.to_pydict().values()))

    with ThreadPoolExecutor(16) as pool:
        for source, src in SOURCES.items():
            shards = [p for p in sorted(fs.glob(f"datasets/{REPO}@{revision}/{source}/segments/*.parquet"))
                      if p.rsplit("/", 1)[1] not in src.heldout]
            for i in range(0, len(shards), 16):
                for rows in pool.map(read, shards[i:i + 16]):
                    yield source, rows


# Above 0.15 the corpus transcript and the clip differ by phrases (PhoAudiobook's repeated clauses);
# from 0.10 to 0.15 mostly the ASR's misses on names and loanwords, numbers and percentages spelled
# otherwise, a word at a cut edge (tranche-cache#gates).
MAX_CER = 0.15
# GigaSpeech 2's transcripts drop these from numbers ("hơn một chuyến" where the speaker says "hơn
# một nghìn chuyến"), a wrong fact the CER barely sees (tranche-cache#gates).
MAGNITUDES = ("nghìn", "triệu", "tỷ")


def magnitudes(text: str) -> list[int]:
    words = spoken(text)
    return [words.count(m) for m in MAGNITUDES]


def vi_gate(row: dict, max_cer: float = MAX_CER) -> str | None:
    """The first Vietnamese-side filter that drops a row (its `text` and this module's asr_vi
    columns): the cache build drops such rows, so voice.py does not voice them."""
    if row["language"] != "Vietnamese":
        return f"language {row['language'] or 'none'}"
    if round(row["cer"], 6) > max_cer:  # float32: 0.1 is stored as 0.10000000149
        return f"CER {round(row['cer'], 6)} over {max_cer}"
    # Word by word: a swapped magnitude (the ASR's tỷ for the transcript's triệu) is a wrong fact.
    if any(heard > written for heard, written in zip(magnitudes(row["asr_vi"]), magnitudes(row["text"]))):
        return "the ASR hears a nghìn/triệu/tỷ the transcript lacks"
    return None


def overlap(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    grams = eval_grams()
    log("evaluation runs: " + ", ".join(f"{len(found)} of {n}" for n, found in sorted(grams.items())))
    def report(source: str, total: int, found: list[str]) -> None:
        log(f"{source}: {len(found)} of {total} rows overlap the evaluation sets"
            + (f", e.g. {found[:5]}" if found else ""))

    current, total, found = None, 0, []
    for source, rows in training_texts(HfApi().dataset_info(REPO).sha):  # source by source
        if source != current:
            if current is not None:
                report(current, total, found)
            current, total, found = source, 0, []
        total += len(rows)
        found += [i for i, _, text in rows if overlaps(text, grams)]
    if current is not None:
        report(current, total, found)


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
        and `language None` for audio with no speech; a reply without the header is all
        transcript, its language unknown (""), as qwen_asr's own parser reads it. Clips over 30 s,
        which the server would split and join headers into, do not occur (rows are under 22.4 s)."""
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
        raw = reply.json()["text"]
        head, tag, text = raw.partition("<asr_text>")
        if not tag or not head.startswith("language "):
            return "", raw.strip()
        language = head.removeprefix("language ").strip()
        return ("", "") if language == "None" else (language, text.strip())


def asr_shard(source: str, name: str, root: Path, hearer: Hearer, decoders, pool: ThreadPoolExecutor) -> Path:
    import jiwer
    import pyarrow as pa
    import pyarrow.parquet as pq
    from training.tranche.ingest import part_file

    seg = pq.read_table(part_file(root, source, "segments", name), columns=["id", "text", "audio"])
    audio = [a["bytes"] for a in seg.column("audio").to_pylist()]
    heard = list(pool.map(hearer.hear, decoders.map(wav_16k, audio, chunksize=32)))
    cer = []
    for text, (_, asr) in zip(seg.column("text").to_pylist(), heard):
        ref, hyp = " ".join(spoken(text)), " ".join(spoken(asr))
        cer.append(jiwer.cer(ref, hyp) if ref else float(bool(hyp)))
    table = pa.table({"id": seg.column("id"), "asr_vi": pa.array([h[1] for h in heard], pa.string()),
                      "language": pa.array([h[0] for h in heard], pa.string()),
                      "cer": pa.array(cer, pa.float32())})
    dst = root / source / "asr_vi" / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".part")
    pq.write_table(table, part)
    part.replace(dst)
    return dst


def asr(args: argparse.Namespace) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    from training.tranche.ingest import an_token

    an_token()
    api = HfApi()
    todo = []
    for source in args.sources:
        files = api.list_repo_files(REPO, repo_type="dataset")
        done = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/asr_vi/")}
        folder = args.root / source / "asr_vi"
        todo += [(source, f.rsplit("/", 1)[1]) for f in sorted(files) if f.startswith(f"{source}/segments/")
                 and f.rsplit("/", 1)[1] not in done and not (folder / f.rsplit("/", 1)[1]).exists()]
        if {p.name for p in folder.glob("*.parquet")} - done:  # written by an earlier run that stopped before pushing them
            push_part(args.root, source, "asr_vi")
    todo = todo[: args.limit]
    log(f"{len(todo)} shards to hear: " + ", ".join(f"{s} {sum(t[0] == s for t in todo)}" for s in args.sources))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"  # the decoders' thread pools, under the box's thread limit (voice.py)
    from training.paths import cpu_budget

    decoders = ProcessPoolExecutor(max(1, int(cpu_budget() // DECODER_SHARE)), mp_context=get_context("spawn"))
    hearer = Hearer(args.asr)
    pushed = {s: time.monotonic() for s in args.sources}
    began, rows = time.monotonic(), 0
    pool = ThreadPoolExecutor(CONCURRENCY)
    try:
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
    finally:  # after a failure, queued requests and decodes are dropped rather than run
        pool.shutdown(wait=False, cancel_futures=True)
        decoders.shutdown(wait=False, cancel_futures=True)
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

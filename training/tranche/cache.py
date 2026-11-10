#!/usr/bin/env python
"""The tranche's training caches: each segments shard's rows as grounded-v2 cache rows
([tranche-cache](../../docs/agent-memory/tranche-cache.md)), built by the cache build every builder
shares (`training.prep.cache_build`, [cache-build](../../docs/agent-memory/cache-build.md)).

  build <sources...>  every shard whose segments, text_en, en_audio and asr_vi are on the Hub
                      becomes <root>/cache/<source>/shard_<shard stem>.pt:
                      the rows the filters below keep, the English text timed by wav2vec2-base-960h
                      CTC (the `text_timing` Viterbi) and both clips Mimi-encoded in float32, each
                      sample with its `doc` and `seq` so the trainer joins a document's rows in
                      order. Pushes each source's folder every 30 min and at its end; a rerun skips
                      the shards written or pushed, and first pushes any written but not yet pushed.
                      With --direction en-vi the languages trade places for two-way training (the
                      corpus transcript timed by the Vietnamese aligner, the real clip the target,
                      the English TTS the source, tokenized by the Vietnamese swap), into
                      <root>/cache/en-vi/<source>/ and the cache repo's en-vi/<source>/.

A row is dropped, with the first reason that applies (the shard's `rejected`), when
  - Qwen3-ASR heard its clip as another language than Vietnamese, or none;
  - its corpus transcript and that ASR differ by more than --max-cer (CER over `filters.spoken`);
  - the ASR hears more nghìn/triệu/tỷ than the transcript holds (GigaSpeech 2's transcripts drop them);
    (these three are `filters.vi_gate`, which voice.py applies before voicing: from 2026-11-07 such
    rows have a null en_audio, and one dropped at the default --max-cer but kept by a looser one is
    "not voiced");
  - its English stayed broken after the TTS's retries (`en_broken`);
  - the ASR of its English writes Vietnamese letters its text_en lacks (a Vietnamese lead-in);
  - its transcript overlaps an evaluation sentence (`filters.overlaps`; needs data/eval);
  - then on the build's own gates: a clip decodes more than 0.02 s off its stored duration, the row
    exceeds --max-frames, or its English text fails CTC alignment or token timing.

From the repo root, with HF_TOKEN_AN (pushes) in the environment; on the box (CUDA, the training
.venv from training/h100.sh setup):
  .venv/bin/python -m training.tranche.cache build pho vivoice giga
on the Mac (MLX; nice, since the user games on it), into a local root:
  nice -n 19 python -m training.tranche.cache build giga --device mlx --root /Volumes/data/datasets/tranche-mac --limit 2
"""
from __future__ import annotations

import argparse
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from training.paths import DEFAULT_CONFIG_PATH, DEFAULT_MIMI_WEIGHT, DIRECTION_TOKENIZERS, exclusive
from training.prep.cache_build import MODELS, Builder, default_workers, summary
from training.prep.cache_rows import DEFAULT_DIRECTION, DIRECTIONS, Settings, Shard, mimi_hours, write
from training.tranche.filters import MAX_CER, eval_grams, overlaps, vi_gate
from training.tranche.ingest import CACHE_REPO, REPO, ROOT, SOURCES, part_file, push_part

SPLIT = "train"  # the delay draw's key
VI_LETTERS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
PUSH_S = 1800
# The columns the build reads of each part (read_parts).
PART_COLUMNS = {
    "segments": ["id", "doc", "seq", "text", "duration_s", "audio"],
    "text_en": ["id", "text_en"],
    "en_audio": ["id", "en_audio", "en_duration_s", "en_asr", "en_broken"],
    "asr_vi": ["id", "asr_vi", "language", "cer"],
}
PARTS = tuple(PART_COLUMNS)


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
    p.add_argument("--device", choices=list(MODELS), default="cuda")
    p.add_argument("--direction", choices=DIRECTIONS, default=DEFAULT_DIRECTION,
                   help="<source>-<target>: the language heard, then the one spoken and written")
    p.add_argument("--half-ctc", action="store_true", help="wav2vec2 in float16 (MLX) or bfloat16 (CUDA): +~16%% CTC speed (MLX fixed code, unvalidated; CUDA failed quality check)")
    p.add_argument("--workers", type=int, help="decode and alignment processes (default: the device's)")
    p.add_argument("--drop-inputs", action="store_true",
                   help="delete each shard's parts from the HF cache once read (the box's own copies stay): "
                        "a build of every shard then holds a shard or two of inputs, not all of them")
    p.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    p.add_argument("--tokenizer", type=Path, help="default: the direction's (paths.DIRECTION_TOKENIZERS)")
    return parser.parse_args()


# --------------------------------------------------------------------------- shards


def vietnamese_letters(en_asr: str, text_en: str) -> set[str]:
    """Vietnamese letters the English ASR wrote that the English text lacks: a clause the TTS spoke
    in Vietnamese (tranche-tts), or a Vietnamese name the ASR spelled otherwise."""
    have = set(unicodedata.normalize("NFC", text_en).lower())
    return {c for c in unicodedata.normalize("NFC", en_asr).lower() if c in VI_LETTERS} - have


def read_parts(source: str, name: str, files: dict[str, Path],
               columns: dict[str, list[str]] = PART_COLUMNS) -> list[dict[str, Any]]:
    """The shard's rows with these columns of each part's file (segments' among them), in the
    segments' order."""
    import pyarrow.parquet as pq

    def get(part: str) -> list[dict[str, Any]]:
        return pq.read_table(files[part], columns=columns[part]).to_pylist()

    rows = get("segments")
    for part in [p for p in columns if p != "segments"]:
        other = get(part)
        if [r["id"] for r in other] != [r["id"] for r in rows]:
            raise RuntimeError(f"{source}/{part}/{name} does not hold the segments' rows in their order")
        for row, extra in zip(rows, other, strict=True):
            row.update(extra)
    return rows


def gate(row: dict[str, Any], max_cer: float, grams) -> str | None:
    """The first filter that drops the row, before any audio is decoded. The Vietnamese side comes
    first: voice.py skips the rows it drops at the default --max-cer, leaving their en_audio null."""
    if reason := vi_gate(row, max_cer):
        return reason
    if row["en_audio"] is None:
        return "not voiced"  # skipped by voice.py: vi_gate at MAX_CER, or missing text_en; also past a looser --max-cer
    if row["en_broken"]:
        return "en_broken"
    if letters := vietnamese_letters(row["en_asr"], row["text_en"]):
        return f"Vietnamese letters in en_asr: {''.join(sorted(letters))}"
    if overlaps(row["text"], grams):
        return "overlaps an evaluation sentence"
    return None


def read_shard(root: Path, max_cer: float, grams, key: tuple[str, str], drop_inputs: bool = False) -> Shard:
    """The shard fetched, read and gated (in the build's read thread); with `drop_inputs`, the parts
    the HF cache holds are deleted once read, the rows being in memory."""
    source, name = key
    # Each part from the box's own copy first (`ingest.part_file`), the four at once.
    with ThreadPoolExecutor(len(PARTS)) as fetch:
        files = dict(zip(PARTS, fetch.map(lambda part: part_file(root, source, part, name), PARTS), strict=True))
    rows = read_parts(source, name, files)
    if drop_inputs:
        for path in files.values():
            if not path.is_relative_to(root):  # the HF cache's: a snapshot link to its blob
                blob = path.resolve() if path.is_symlink() else None
                path.unlink()
                if blob is not None:
                    blob.unlink(missing_ok=True)
    kept, rejected = [], []
    for index, row in enumerate(rows):
        if reason := gate(row, max_cer, grams):
            rejected.append({"id": row["id"], "reason": reason})
            continue
        # GigaSpeech 2's transcripts are upper case (vi-corpora): lower-cased, as translate.py read them,
        # since an en-vi row writes them.
        text_vi = row["text"].lower() if row["text"].isupper() else row["text"]
        kept.append((index, {
            "id": row["id"], "text_en": row["text_en"], "text_vi": text_vi,
            "vi_duration_s": row["duration_s"], "en_duration_s": row["en_duration_s"],
            "vi_audio": row["audio"]["bytes"], "en_audio": row["en_audio"]["bytes"],
            "fields": {"doc": row["doc"], "seq": int(row["seq"])},
        }))
    return Shard(key, f"{source}/segments/{name}", SPLIT, len(rows), kept, rejected)


def cache_key(direction: str, source: str) -> str:
    """A source's folder under <root>/cache and in the cache repo: its name, under en-vi/ for that direction."""
    return source if direction == DEFAULT_DIRECTION else f"{direction}/{source}"


def cache_dir(root: Path, key: str) -> Path:
    return root / "cache" / key


def shard_file(name: str) -> str:
    return f"shard_{name.removesuffix('.parquet')}.pt"


def todo_shards(args: argparse.Namespace) -> list[tuple[str, str]]:
    """The shards with every input on the Hub and no cache here or there, but those held out for
    evaluation (`Source.heldout`), pushing first the caches an earlier run wrote but did not push."""
    from huggingface_hub import HfApi

    api, todo = HfApi(), []
    files, cached = (api.list_repo_files(r, repo_type="dataset") for r in (REPO, CACHE_REPO))
    for source in args.sources:
        parts = {part: {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/{part}/")}
                 for part in PARTS}
        key = cache_key(args.direction, source)
        pushed = {f.rsplit("/", 1)[1] for f in cached if f.startswith(f"{key}/")}
        local = {p.name for p in cache_dir(args.root, key).glob("shard_*.pt")}
        heldout = SOURCES[source].heldout
        ready = set.intersection(*parts.values()) - heldout
        waiting = len(parts["segments"] - heldout - ready)
        todo += [(source, n) for n in sorted(ready) if shard_file(n) not in pushed | local]
        log(f"{source}: {len(parts['segments'])} shards ({len(heldout)} held out), {len(pushed)} cached on the "
            f"Hub, {len(local - pushed)} only here, {waiting} waiting for an input")
        if local - pushed:
            push_part(args.root, key, "cache")
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
    builder = Builder(args.device, Settings(args.max_frames, direction=args.direction), args.config_path,
                      args.mimi_weight, args.tokenizer or DIRECTION_TOKENIZERS[args.direction], half_ctc=args.half_ctc)
    pushed = {s: time.monotonic() for s in args.sources}
    workers = args.workers or default_workers(args.device)

    def read(key: tuple[str, str]) -> Shard:
        return read_shard(args.root, args.max_cer, grams, key, args.drop_inputs)

    with ThreadPoolExecutor(1) as writes:
        written: Future | None = None  # the last shard's save, which runs while the next one builds
        began, hours = time.monotonic(), 0.0
        for k, shard in enumerate(builder.run(todo, read, workers)):
            source, name = shard.key
            payload = builder.payload(shard, {"file": shard.file, "rows": shard.size}, max_cer=args.max_cer)
            if written is not None:
                written.result()
            key = cache_key(args.direction, source)
            written = writes.submit(write, payload, cache_dir(args.root, key) / shard_file(name))
            hours += mimi_hours(shard.samples)
            now = time.monotonic()
            log(f"{source}/{name}: {summary(shard, hours, now - began)} [{k + 1}/{len(todo)}]")
            last_of_source = k + 1 == len(todo) or todo[k + 1][0] != source
            if last_of_source or now - pushed[source] > PUSH_S:
                written.result()
                push_part(args.root, key, "cache")
                pushed[source] = time.monotonic()
        wall = time.monotonic() - began
        log(f"done: {len(todo)} shards, {hours:.1f} h of Mimi audio in {wall / 3600:.2f} h "
            f"({hours * 3600 / max(wall, 1e-9):.0f}x RT)")


def main() -> None:
    args = parse_args()
    {"build": build}[args.step](args)


if __name__ == "__main__":
    main()

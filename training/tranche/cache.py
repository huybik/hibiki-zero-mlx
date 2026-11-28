#!/usr/bin/env python
"""The tranche's training caches: each segments shard's rows as grounded-v2 cache rows
([tranche-cache](../../docs/agent-memory/tranche-cache.md)), built by the cache build every builder
shares (`training.prep.cache_build`, [cache-build](../../docs/agent-memory/cache-build.md)).

  build <sources...>  every shard whose segments, text_en, en_audio and asr_vi are on the Hub
                      becomes <root>/cache/<source>/shard_<shard stem>.pt:
                      the rows the filters below keep, the English text timed by wav2vec2-base-960h
                      CTC (the `text_timing` Viterbi) and both clips Mimi-encoded in float32, each
                      sample with its `doc` and `seq` so the trainer joins a document's rows in
                      order. Pushes each source's folder every 30 min and at its end, in a thread
                      beside the build (`ingest.Pushes`), and logs `done:` once the last is in; a
                      rerun skips the shards written or pushed, and first pushes any written but not
                      yet pushed.
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

From the repo root, with HF_TOKEN_AN (pushes) in the environment; on the box (CUDA; Python 3.14t with Triton from
training/tranche/box/envs.sh, which the training .venv can stand in for):
  /workspace/envs/cache/bin/python -m training.tranche.cache build pho vivoice giga --direction both --drop-inputs
on the Mac (MLX; nice, since the user games on it), into a local root:
  nice -n 19 python -m training.tranche.cache build giga --device mlx --root /Volumes/data/datasets/tranche-mac --limit 2
"""
from __future__ import annotations

import argparse
import sys
import time
import unicodedata
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from training.paths import DEFAULT_CONFIG_PATH, DEFAULT_MIMI_WEIGHT, DIRECTION_TOKENIZERS, exclusive
from training.prep.cache_build import MODELS, Builder, DualBuilder, default_workers, summary
from training.prep.cache_rows import DEFAULT_DIRECTION, DIRECTIONS, Settings, Shard, mimi_hours, write
from training.tranche.filters import MAX_CER, eval_grams, overlaps, vi_gate
from training.tranche.ingest import CACHE_REPO, REPO, ROOT, SOURCES, Pushes, part_file

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
    p.add_argument("--direction", choices=[*DIRECTIONS, "both"], default=DEFAULT_DIRECTION,
                   help="<source>-<target>: the language heard, then the one spoken and written; "
                        "'both' builds vi-en and en-vi in a single CUDA pass (decodes each clip once)")
    p.add_argument("--half-ctc", action="store_true", help="wav2vec2 in float16 (MLX; on CUDA fp16 is default)")
    p.add_argument("--fp8-ctc", action="store_true", help="CUDA Blackwell FP8 on wav2vec2 transformer linears")
    p.add_argument("--fp8-mimi", action="store_true", help="CUDA Blackwell FP8 on Mimi encoder transformer linears")
    p.add_argument("--prefetch", type=int, default=3, help="Number of shards to prefetch from the Hub ahead of reading")
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
        return "not voiced"  # skipped by voice.py: vi_gate at MAX_CER, text_en missing or unspeakable; also past a looser --max-cer
    if row["en_broken"]:
        return "en_broken"
    if letters := vietnamese_letters(row["en_asr"], row["text_en"]):
        return f"Vietnamese letters in en_asr: {''.join(sorted(letters))}"
    if overlaps(row["text"], grams):
        return "overlaps an evaluation sentence"
    return None


def fetch_shard_parts(root: Path, key: tuple[str, str]) -> dict[str, Path]:
    """Fetch a shard's four parquet parts in parallel (local box copies first, else Hub)."""
    source, name = key
    with ThreadPoolExecutor(len(PARTS)) as fetch:
        return dict(zip(PARTS, fetch.map(lambda part: part_file(root, source, part, name), PARTS), strict=True))


def read_shard(
    root: Path,
    max_cer: float,
    grams,
    key: tuple[str, str],
    drop_inputs: bool = False,
    prefetched: dict[str, Path] | None = None,
) -> Shard:
    """The shard fetched, read and gated (in the build's read thread); filters on lightweight
    metadata columns before extracting audio bytes, avoiding Python dict/bytes allocation for
    rejected rows."""
    import pyarrow.parquet as pq

    source, name = key
    files = prefetched if prefetched is not None else fetch_shard_parts(root, key)

    # Read lightweight metadata columns first; check row-id alignment and run gates before
    # touching the heavy audio binary columns.
    seg_meta = pq.read_table(files["segments"], columns=["id", "doc", "seq", "text", "duration_s"]).to_pylist()
    txt_rows = pq.read_table(files["text_en"], columns=PART_COLUMNS["text_en"]).to_pylist()
    asr_rows = pq.read_table(files["asr_vi"], columns=PART_COLUMNS["asr_vi"]).to_pylist()
    en_meta = pq.read_table(files["en_audio"], columns=["id", "en_duration_s", "en_asr", "en_broken"]).to_pylist()
    en_audio_col = pq.read_table(files["en_audio"], columns=["en_audio"]).column("en_audio")

    seg_ids = [r["id"] for r in seg_meta]
    for part_name, other in (("text_en", txt_rows), ("asr_vi", asr_rows), ("en_audio", en_meta)):
        if [r["id"] for r in other] != seg_ids:
            raise RuntimeError(f"{source}/{part_name}/{name} does not hold the segments' rows in their order")

    en_audio_valid = en_audio_col.is_valid().to_pylist()
    kept_indices: list[int] = []
    kept_meta: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for index, (seg, txt, asr, enm, has_en) in enumerate(
        zip(seg_meta, txt_rows, asr_rows, en_meta, en_audio_valid, strict=True)
    ):
        merged = {**seg, **txt, **asr, **enm, "en_audio": True if has_en else None}
        if reason := gate(merged, max_cer, grams):
            rejected.append({"id": seg["id"], "reason": reason})
            continue
        kept_indices.append(index)
        kept_meta.append(merged)

    if kept_indices:
        vi_audio_col = pq.read_table(files["segments"], columns=["audio"]).column("audio")
        vi_bytes = vi_audio_col.take(kept_indices).combine_chunks().field("bytes").to_pylist()
        en_bytes = en_audio_col.take(kept_indices).combine_chunks().field("bytes").to_pylist()
    else:
        vi_bytes, en_bytes = [], []

    if drop_inputs:
        for path in files.values():
            if not path.is_relative_to(root):  # the HF cache's: a snapshot link to its blob
                blob = path.resolve() if path.is_symlink() else None
                path.unlink()
                if blob is not None:
                    blob.unlink(missing_ok=True)

    kept: list[tuple[int, dict[str, Any]]] = []
    for index, row, vi_b, en_b in zip(kept_indices, kept_meta, vi_bytes, en_bytes, strict=True):
        text_vi = row["text"].lower() if row["text"].isupper() else row["text"]
        kept.append((index, {
            "id": row["id"], "text_en": row["text_en"], "text_vi": text_vi,
            "vi_duration_s": row["duration_s"], "en_duration_s": row["en_duration_s"],
            "vi_audio": vi_b, "en_audio": en_b,
            "fields": {"doc": row["doc"], "seq": int(row["seq"])},
        }))
    return Shard(key, f"{source}/segments/{name}", SPLIT, len(seg_meta), kept, rejected)


def cache_key(direction: str, source: str) -> str:
    """A source's folder under <root>/cache and in the cache repo: its name, under en-vi/ for that direction."""
    return source if direction == DEFAULT_DIRECTION else f"{direction}/{source}"


def cache_dir(root: Path, key: str) -> Path:
    return root / "cache" / key


def shard_file(name: str) -> str:
    return f"shard_{name.removesuffix('.parquet')}.pt"


def todo_shards(args: argparse.Namespace) -> tuple[list[tuple[str, str]], list[str]]:
    """The shards with every input on the Hub and no cache here or there, but those held out for
    evaluation (`Source.heldout`), and the cache folders holding shards an earlier run wrote but did
    not push."""
    from huggingface_hub import HfApi

    api, todo, unpushed = HfApi(), [], []
    files, cached = (api.list_repo_files(r, repo_type="dataset") for r in (REPO, CACHE_REPO))
    directions = DIRECTIONS if args.direction == "both" else (args.direction,)
    for source in args.sources:
        parts = {part: {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/{part}/")}
                 for part in PARTS}
        heldout = SOURCES[source].heldout
        ready = set.intersection(*parts.values()) - heldout
        waiting = len(parts["segments"] - heldout - ready)
        done_all: set[str] | None = None
        for d in directions:
            key = cache_key(d, source)
            pushed = {f.rsplit("/", 1)[1] for f in cached if f.startswith(f"{key}/")}
            local = {p.name for p in cache_dir(args.root, key).glob("shard_*.pt")}
            have = pushed | local
            done_all = have if done_all is None else (done_all & have)
            log(f"{source} ({d}): {len(parts['segments'])} shards ({len(heldout)} held out), {len(pushed)} cached on the "
                f"Hub, {len(local - pushed)} only here, {waiting} waiting for an input")
            if local - pushed:
                unpushed.append(key)
        assert done_all is not None
        todo += [(source, n) for n in sorted(ready) if shard_file(n) not in done_all]
    return todo[: args.limit], unpushed


def build(args: argparse.Namespace) -> None:
    from training.tranche.ingest import an_token

    an_token()
    # Taken first: a second run must not push or load models onto a GPU the first one uses.
    args.root.mkdir(parents=True, exist_ok=True)
    with exclusive(args.root / ".cache.lock", f"another cache build uses {args.root}"):
        todo, unpushed = todo_shards(args)
        log(f"{len(todo)} shards to build: " + ", ".join(f"{s} {sum(t[0] == s for t in todo)}" for s in args.sources))
        pushes = Pushes(args.root, "cache")
        try:  # the lock is held until the last push ends, after a failure too
            for key in unpushed:
                pushes.push(key)
            if todo:
                built = build_shards_dual(args, todo, pushes) if args.direction == "both" else build_shards(args, todo, pushes)
            else:
                built = None
        finally:
            pushes.wait()
    if built:  # once every shard is on the Hub, as whatever waits on this line expects
        hours, wall = built
        log(f"done: {len(todo)} shards, {hours:.1f} h of Mimi audio in {wall / 3600:.2f} h "
            f"({hours * 3600 / max(wall, 1e-9):.0f}x RT)")


def _make_prefetched_reader(
    root: Path, max_cer: float, grams: Any, todo: list[tuple[str, str]], drop_inputs: bool, prefetch: int
):
    """Prefetch up to `prefetch` shards' compressed parquets ahead of `read_shard`."""
    from collections import deque

    pool = ThreadPoolExecutor(max(1, prefetch), thread_name_prefix="prefetch") if prefetch > 0 else None
    pending: dict[tuple[str, str], Future] = {}
    order = deque(todo)

    def fill() -> None:
        if pool is None:
            return
        while len(pending) < prefetch and order:
            nxt = order.popleft()
            pending[nxt] = pool.submit(fetch_shard_parts, root, nxt)

    fill()

    def read(key: tuple[str, str]) -> Shard:
        fut = pending.pop(key, None)
        fill()
        files = fut.result() if fut is not None else None
        return read_shard(root, max_cer, grams, key, drop_inputs, prefetched=files)

    return read, pool


def build_shards(args: argparse.Namespace, todo: list[tuple[str, str]], pushes: Pushes) -> tuple[float, float]:
    """Builds `todo`, handing each source's folder to `pushes` as due; returns the hours of Mimi
    audio built and the build's wall time."""
    grams = eval_grams()
    builder = Builder(
        args.device,
        Settings(args.max_frames, direction=args.direction),
        args.config_path,
        args.mimi_weight,
        args.tokenizer or DIRECTION_TOKENIZERS[args.direction],
        half_ctc=args.half_ctc,
        fp8_ctc=args.fp8_ctc,
        fp8_mimi=args.fp8_mimi,
    )
    gil = "on" if getattr(sys, "_is_gil_enabled", lambda: True)() else "off"
    log(f"models loaded; Python {sys.version.split()[0]}, the GIL {gil}")
    pushed = {s: time.monotonic() for s in args.sources}
    workers = args.workers or default_workers(args.device)
    read, prefetch_pool = _make_prefetched_reader(
        args.root, args.max_cer, grams, todo, args.drop_inputs, args.prefetch
    )
    try:
        with ThreadPoolExecutor(1) as writes:
            written: Future | None = None
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
                    pushes.push(key)
                    pushed[source] = now
            if written is not None:
                written.result()
            return hours, time.monotonic() - began
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=False, cancel_futures=True)


def build_shards_dual(args: argparse.Namespace, todo: list[tuple[str, str]], pushes: Pushes) -> tuple[float, float]:
    """Single-pass dual-direction CUDA build (`vi-en` and `en-vi` simultaneously)."""
    if args.device != "cuda":
        raise ValueError("--direction both is supported on --device cuda only")
    grams = eval_grams()
    builder = DualBuilder(
        Settings(args.max_frames, direction="vi-en"),
        Settings(args.max_frames, direction="en-vi"),
        args.config_path,
        args.mimi_weight,
        DIRECTION_TOKENIZERS["vi-en"],
        DIRECTION_TOKENIZERS["en-vi"],
        fp8_ctc=args.fp8_ctc,
        fp8_mimi=args.fp8_mimi,
    )
    gil = "on" if getattr(sys, "_is_gil_enabled", lambda: True)() else "off"
    log(f"dual models loaded (vi-en + en-vi); Python {sys.version.split()[0]}, the GIL {gil}")
    pushed = {s: time.monotonic() for s in args.sources}
    workers = args.workers or default_workers(args.device)
    read, prefetch_pool = _make_prefetched_reader(
        args.root, args.max_cer, grams, todo, args.drop_inputs, args.prefetch
    )
    try:
        with ThreadPoolExecutor(2) as writes:
            written: list[Future] = []
            began, hours = time.monotonic(), 0.0
            for k, (sh_ve, sh_ev) in enumerate(builder.run(todo, read, workers)):
                source, name = sh_ve.key
                for fut in written:
                    fut.result()
                written = []
                for d, sh in (("vi-en", sh_ve), ("en-vi", sh_ev)):
                    key = cache_key(d, source)
                    dst = cache_dir(args.root, key) / shard_file(name)
                    if not dst.exists():
                        pl = builder.payload(d, sh, {"file": sh.file, "rows": sh.size}, max_cer=args.max_cer)
                        written.append(writes.submit(write, pl, dst))
                hours += mimi_hours(sh_ve.samples) + mimi_hours(sh_ev.samples)
                now = time.monotonic()
                log(f"{source}/{name} [vi-en]: {summary(sh_ve, hours, now - began)} [{k + 1}/{len(todo)}]")
                log(f"{source}/{name} [en-vi]: {summary(sh_ev, hours, now - began)} [{k + 1}/{len(todo)}]")
                last_of_source = k + 1 == len(todo) or todo[k + 1][0] != source
                if last_of_source or now - pushed[source] > PUSH_S:
                    for fut in written:
                        fut.result()
                    written = []
                    for d in DIRECTIONS:
                        pushes.push(cache_key(d, source))
                    pushed[source] = now
            for fut in written:
                fut.result()
            return hours, time.monotonic() - began
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    args = parse_args()
    {"build": build}[args.step](args)


if __name__ == "__main__":
    main()

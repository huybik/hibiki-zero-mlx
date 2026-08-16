#!/usr/bin/env python
"""Stream PhoMT-en-vi-speech parquet shards straight into Mimi code caches.

The full dataset is ~565 GB of parquet (696k rows); a rented box cannot hold
parquet + extracted wavs + caches. This script downloads ONE parquet shard at a
time, Mimi-encodes each row via a tmpfs temp wav (no wavs ever touch disk),
writes one CachedCodeDataset-compatible cache shard per parquet shard, then
deletes the parquet from the HF cache. Peak disk = one parquet shard per worker.

Run N workers as separate processes (worker k takes parquet shards k::N):
  for i in $(seq 0 7); do
    python finetune/cache_phomt_stream.py --worker $i --num-workers 8 \
      --skip-pairs finetune/pairs/phomt_train.jsonl --device cuda &
  done

Sample ids are phomt_s{parquet}r{row} — deterministic regardless of worker
count, so target-delay RNG and resume never depend on process layout.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import ctypes
import gc
import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.cache_codes import (  # noqa: E402
    CACHE_FORMAT,
    FRAME_RATE,
    GROUNDED_CACHE_FORMAT,
    SAMPLE_RATE,
    assemble_codes,
    check_device,
    read_audio,
    require_runtime_deps,
    save_shard,
    target_delay_s,
    text_tokens,
)
from finetune.utils import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_MIMI_WEIGHT,
    DEFAULT_TOKENIZER,
    read_json,
    read_pair_file,
    repo_display_path,
    require_file,
    resolve_repo_path,
)

DATASET = "anquachdev/PhoMT-en-vi-speech"
DATASET_REVISION = "33400f73dde07da539e8326313cbabe20b757740"
TIMBRE_MATCHED_MIN_INDEX = 345_600
LIBC = ctypes.CDLL(None) if sys.platform.startswith("linux") else None


def load_hf_token() -> None:
    """Load a local HF token and anchor a broken external cache to the repo."""
    if not os.environ.get("HF_TOKEN"):
        env = REPO_ROOT / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
    hf_home = os.environ.get("HF_HOME")
    if not hf_home or not Path(hf_home).is_dir():
        os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


def row_key(text_vi: str, text_en: str, vi_dur: str, en_dur: str) -> tuple[str, str, str, str]:
    return (text_vi.strip(), text_en.strip(), vi_dur, en_dur)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream PhoMT parquet -> Mimi cache, no wavs on disk."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CACHE_ROOT / "phomt_stream")
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--skip-pairs",
        type=Path,
        nargs="*",
        default=[],
        help="Pair jsonl files whose rows are already cached (keyed on texts+durations).",
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mimi-weight", type=Path, default=DEFAULT_MIMI_WEIGHT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--keep-parquet",
        action="store_true",
        help="Keep downloaded parquet in the HF cache (Mac run with pre-synced external disk).",
    )
    parser.add_argument("--max-source-duration-s", type=float, default=25.0)
    parser.add_argument("--target-delay-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--limit", type=int, default=0, help="Stop after keeping N rows (smoke); 0=all."
    )
    parser.add_argument(
        "--profile",
        choices=("default", "h100"),
        default="default",
        help="Hardware-tuned execution profile with bounded H100 batches.",
    )
    parser.add_argument("--batch-size", type=int, help="Clips per batched Mimi encode.")
    parser.add_argument(
        "--batch-sample-budget",
        type=int,
        default=None,
        help="Maximum padded raw-audio samples per Mimi encode batch.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help="Rows buffered per encode flush; use 128 for concurrent MPS workers.",
    )
    parser.add_argument(
        "--prefetch-shards",
        type=int,
        default=None,
        help="Parquet shards to download ahead while encoding; use 2 for a single Mac worker.",
    )
    parser.add_argument(
        "--recipe",
        choices=("legacy", "grounded-v2"),
        default="legacy",
        help="grounded-v2 CTC-aligns English text tokens to target speech.",
    )
    parser.add_argument("--alignment-batch-size", type=int)
    parser.add_argument(
        "--alignment-sample-budget",
        type=int,
        help="Maximum padded 16 kHz samples per CTC batch; 0 disables the cap.",
    )
    parser.add_argument(
        "--min-alignment-score",
        type=float,
        default=0.5,
        help="Reject grounded-v2 rows below this mean forced-alignment posterior.",
    )
    parser.add_argument(
        "--sample-shards",
        type=int,
        default=0,
        help="Evenly sample this many dataset shards for a grounded-v2 pilot; 0=all.",
    )
    args = parser.parse_args()
    defaults = (
        {
            "batch_size": 64,
            "batch_sample_budget": 4_000_000,
            "chunk_rows": 512,
            "prefetch_shards": 2,
            "alignment_batch_size": 32,
            "alignment_sample_budget": 4_000_000,
        }
        if args.profile == "h100"
        else {
            "batch_size": 16,
            "batch_sample_budget": 2_000_000,
            "chunk_rows": 512,
            "prefetch_shards": 1,
            "alignment_batch_size": 8,
            "alignment_sample_budget": 0,
        }
    )
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


# Mimi's SEANet convs are ~512-wide at raw 24 kHz, so activation memory scales
# with batch total samples: cap padded B*T per encode (~83 s audio ≈ a few GB
# peak) instead of clip count — short clips batch wide, long clips narrow.
def encode_batch(wavs: list, mimi, torch, batch_size: int, sample_budget: int) -> list:
    """Batched Mimi encode of variable-length mono cpu wavs -> per-clip codes.

    Clips are length-sorted into batches, zero right-padded to a shared
    16-frame-bucket length (bounds Metal per-shape kernel-graph variety),
    then trimmed back to each clip's own frame count. Mimi is causal, so
    the trimmed prefix matches a solo encode bit-exact (A/B-verified).
    """
    device = next(mimi.parameters()).device
    frame = int(mimi.frame_size)
    bucket = frame * 16
    out: list = [None] * len(wavs)
    order = sorted(range(len(wavs)), key=lambda i: int(wavs[i].shape[0]))
    s = 0
    while s < len(order):
        idxs = [order[s]]
        pad_t = math.ceil(int(wavs[order[s]].shape[0]) / bucket) * bucket
        while (
            len(idxs) < batch_size
            and s + len(idxs) < len(order)
            and (len(idxs) + 1)
            * (math.ceil(int(wavs[order[s + len(idxs)]].shape[0]) / bucket) * bucket)
            <= sample_budget
        ):
            idxs.append(order[s + len(idxs)])
        # power-of-2 batch sizes: (B, pad_t) shape combos each compile a Metal
        # kernel graph for the whole conv stack; unquantized B explodes variety.
        while len(idxs) & (len(idxs) - 1):
            idxs.pop()
        pad_t = math.ceil(int(wavs[idxs[-1]].shape[0]) / bucket) * bucket
        s += len(idxs)
        batch = torch.zeros(len(idxs), 1, pad_t)
        for j, i in enumerate(idxs):
            batch[j, 0, : wavs[i].shape[0]] = wavs[i]
        with torch.no_grad():
            codes = mimi.encode(batch.to(device)).cpu()
        for j, i in enumerate(idxs):
            out[i] = codes[j, :, : math.ceil(int(wavs[i].shape[0]) / frame)].long()
    return out


def delete_from_hf_cache(path: Path) -> None:
    blob = path.resolve()
    path.unlink(missing_ok=True)
    if blob != path:
        blob.unlink(missing_ok=True)


def release_host_memory() -> None:
    """Return completed-shard allocations instead of retaining prep-thread arenas."""
    gc.collect()
    if LIBC is not None:
        LIBC.malloc_trim(0)


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker < args.num_workers:
        raise ValueError("--worker must be in [0, --num-workers)")
    if args.prefetch_shards <= 0:
        raise ValueError("--prefetch-shards must be positive")
    if args.chunk_rows <= 0:
        raise ValueError("--chunk-rows must be positive")
    if args.batch_sample_budget <= 0:
        raise ValueError("--batch-sample-budget must be positive")
    if args.alignment_batch_size <= 0:
        raise ValueError("--alignment-batch-size must be positive")
    if args.alignment_sample_budget < 0:
        raise ValueError("--alignment-sample-budget must be non-negative")
    load_hf_token()

    torch, sphn, sentencepiece, loaders = require_runtime_deps()
    device = check_device(torch, args.device)
    if args.profile == "h100":
        if (
            device.type != "cuda"
            or torch.cuda.get_device_capability(device) != (9, 0)
            or "H100" not in torch.cuda.get_device_name(device).upper()
        ):
            raise RuntimeError("--profile h100 requires an NVIDIA H100 (CUDA capability 9.0)")

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    all_shards = sorted(
        name
        for name in list_repo_files(DATASET, repo_type="dataset", revision=DATASET_REVISION)
        if name.startswith("data/") and name.endswith(".parquet")
    )
    indexed_shards = list(enumerate(all_shards))
    source_ranges: dict[str, tuple[int, int]] = {}
    if args.recipe == "grounded-v2":
        state_path = hf_hub_download(
            DATASET, "upload-state.json", repo_type="dataset", revision=DATASET_REVISION
        )
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        source_ranges = {
            item["path"]: (int(item["source_index_min"]), int(item["source_index_max"]))
            for item in state["shards"]
        }
    elif args.sample_shards:
        raise ValueError("--sample-shards is only valid with --recipe grounded-v2")
    if args.sample_shards:
        if args.sample_shards > len(indexed_shards):
            raise ValueError(
                f"--sample-shards={args.sample_shards} exceeds {len(indexed_shards)} eligible shards"
            )
        if args.sample_shards == 1:
            indexed_shards = [indexed_shards[len(indexed_shards) // 2]]
        else:
            last = len(indexed_shards) - 1
            indexed_shards = [
                indexed_shards[index * last // (args.sample_shards - 1)]
                for index in range(args.sample_shards)
            ]
    mine = [
        item
        for ordinal, item in enumerate(indexed_shards)
        if ordinal % args.num_workers == args.worker
    ]
    print(f"[w{args.worker}] {len(mine)}/{len(indexed_shards)} dataset shards assigned")

    skip: set[tuple[str, str, str, str]] = set()
    for pairs_path in args.skip_pairs:
        for row in read_pair_file(pairs_path):
            skip.add(
                row_key(row["text_vi"], row["text_en"], row["vi_duration_s"], row["en_duration_s"])
            )
    if skip:
        print(f"[w{args.worker}] skipping {len(skip)} already-cached rows")

    cfg = read_json(args.config_path)
    mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    tokenizer = sentencepiece.SentencePieceProcessor(
        str(require_file(args.tokenizer, "text tokenizer"))
    )
    num_codebooks = max(int(cfg["dep_q"]), int(cfg["n_q"]) - int(cfg["dep_q"]))
    mimi = loaders.get_mimi(mimi_weight, num_codebooks=num_codebooks, device=device)
    if int(cfg["card"]) != int(mimi.cardinality):
        raise RuntimeError(f"Config card={cfg['card']} != Mimi cardinality={mimi.cardinality}")

    aligner = None
    if args.recipe == "grounded-v2":
        from finetune.text_timing import EnglishCTCAligner

        aligner = EnglishCTCAligner(
            device,
            args.min_alignment_score,
            alignment_backend="batched" if args.profile == "h100" else "serial",
        )

    if args.recipe == "grounded-v2" and args.out_dir == DEFAULT_CACHE_ROOT / "phomt_stream":
        args.out_dir = DEFAULT_CACHE_ROOT / "phomt_grounded_v2"
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_format = GROUNDED_CACHE_FORMAT if aligner is not None else CACHE_FORMAT
    if existing := next(iter(sorted(out_dir.glob("shard_*.pt"))), None):
        actual_format = torch.load(existing, map_location="cpu").get("format")
        if actual_format != expected_format:
            raise RuntimeError(
                f"Refusing to mix cache formats in {out_dir}: {actual_format} != {expected_format}"
            )
    pairs_path = out_dir / f"pairs_w{args.worker}.jsonl"
    rejects_path = out_dir / f"alignment_rejects_w{args.worker}.jsonl"
    shm = Path("/dev/shm")
    tmp_dir = str(shm) if shm.is_dir() else None

    pending_shards = [
        (parquet_idx, shard_name)
        for parquet_idx, shard_name in mine
        if not (out_dir / f"shard_{parquet_idx:05d}.pt").exists()
    ]

    def download_parquet(shard_name: str) -> Path:
        return Path(
            hf_hub_download(DATASET, shard_name, repo_type="dataset", revision=DATASET_REVISION)
        )

    def iter_parquet():
        if args.limit:
            for parquet_idx, shard_name in pending_shards:
                yield parquet_idx, shard_name, download_parquet(shard_name)
            return
        if not pending_shards:
            return
        with ThreadPoolExecutor(max_workers=args.prefetch_shards) as pool:
            futures = deque(
                pool.submit(download_parquet, shard_name)
                for _, shard_name in pending_shards[: args.prefetch_shards]
            )
            for index, (parquet_idx, shard_name) in enumerate(pending_shards):
                local = futures.popleft().result()
                next_index = index + args.prefetch_shards
                if next_index < len(pending_shards):
                    futures.append(pool.submit(download_parquet, pending_shards[next_index][1]))
                yield parquet_idx, shard_name, local

    kept = 0
    accepted = 0
    rejected = 0
    kept_hours = 0.0
    started = time.monotonic()
    for parquet_idx, shard_name, local in iter_parquet():
        if args.limit and kept >= args.limit:
            break
        out_path = out_dir / f"shard_{parquet_idx:05d}.pt"
        shard_started = time.monotonic()
        shard_kept_start = kept
        table = pq.ParquetFile(local).read()
        samples = []
        pair_lines = []
        reject_lines = []
        cpu = torch.device("cpu")

        def encode_chunk(chunk: list[dict]) -> None:
            nonlocal accepted, rejected
            alignments = None
            groups_batch = None
            if aligner is not None:
                from finetune.text_timing import sentencepiece_groups

                groups_batch = [None] * len(chunk)
                alignments = [None] * len(chunk)
                valid_indices = []
                for index, item in enumerate(chunk):
                    try:
                        groups_batch[index] = sentencepiece_groups(
                            item["row"]["text_en"], tokenizer
                        )
                        valid_indices.append(index)
                    except ValueError as exc:
                        alignments[index] = exc
                valid_alignments = aligner.align_many(
                    [chunk[index]["en_align_wav"] for index in valid_indices],
                    [groups_batch[index] for index in valid_indices],
                    args.alignment_batch_size,
                    args.alignment_sample_budget,
                )
                for index, alignment in zip(valid_indices, valid_alignments, strict=True):
                    alignments[index] = alignment
            encode_indices = []
            for index, item in enumerate(chunk):
                alignment = alignments[index] if alignments is not None else None
                if isinstance(alignment, Exception):
                    rejected += 1
                    print(
                        f"[w{args.worker}] rejecting {item['row']['id']}: {alignment}", flush=True
                    )
                    reject_lines.append({"id": item["row"]["id"], "reason": str(alignment)})
                else:
                    encode_indices.append(index)
            codes = encode_batch(
                [chunk[index]["vi_wav"] for index in encode_indices]
                + [chunk[index]["en_wav"] for index in encode_indices],
                mimi,
                torch,
                args.batch_size,
                args.batch_sample_budget,
            )
            n = len(encode_indices)
            for code_index, k in enumerate(encode_indices):
                c = chunk[k]
                vi_codes, en_codes = codes[code_index], codes[n + code_index]
                row = c["row"]
                text_frames = None
                alignment_score = None
                if alignments is None or groups_batch is None:
                    tokens = text_tokens(row["text_en"], tokenizer)
                else:
                    from finetune.text_timing import timed_sentencepiece_tokens

                    alignment = alignments[k]
                    if alignment is None or groups_batch[k] is None:
                        raise RuntimeError(f"Missing CTC alignment result for {row['id']}")
                    try:
                        tokens, text_frames = timed_sentencepiece_tokens(
                            groups_batch[k],
                            alignment,
                            int(en_codes.shape[1]) - c["delay_frames"],
                            c["delay_frames"],
                            int(tokenizer.eos_id()),
                        )
                    except ValueError as exc:
                        rejected += 1
                        print(f"[w{args.worker}] rejecting {row['id']}: {exc}", flush=True)
                        reject_lines.append({"id": row["id"], "reason": str(exc)})
                        continue
                    alignment_score = alignment.score
                assembled = assemble_codes(
                    torch,
                    row,
                    vi_codes,
                    en_codes,
                    tokens,
                    cfg,
                    c["delay_frames"],
                    text_frames,
                )
                sample = {
                    "id": row["id"],
                    "split": row["split"],
                    "codes": assembled,
                    "frames": int(assembled.shape[1]),
                    "vi_frames": int(vi_codes.shape[1]),
                    "en_frames": int(en_codes.shape[1]),
                    "text_tokens": len(tokens),
                    "target_delay_s": c["delay_s"],
                    "target_delay_frames": c["delay_frames"],
                    "vi_audio": row["vi_audio"],
                    "en_audio": row["en_audio"],
                    "text_en": row["text_en"],
                    "text_vi": row["text_vi"],
                    "text_timing": ("contiguous" if alignments is None else "wav2vec2_ctc_word_v1"),
                    "alignment_score": alignment_score,
                }
                if aligner is not None:
                    sample.update(
                        {
                            "alignment_text": " ".join(spoken for _, spoken in groups_batch[k]),
                            "phomt_index_range": (
                                list(source_ranges[shard_name])
                                if shard_name in source_ranges
                                else None
                            ),
                            "cross_lingual_timbre_matched": (
                                shard_name in source_ranges
                                and source_ranges[shard_name][0] >= TIMBRE_MATCHED_MIN_INDEX
                            ),
                        }
                    )
                samples.append(sample)
                pair_row = row
                if aligner is not None:
                    pair_row = row | {
                        "alignment_score": alignment_score,
                        "alignment_text": " ".join(spoken for _, spoken in groups_batch[k]),
                        "phomt_index_range": (
                            list(source_ranges[shard_name]) if shard_name in source_ranges else None
                        ),
                        "cross_lingual_timbre_matched": (
                            shard_name in source_ranges
                            and source_ranges[shard_name][0] >= TIMBRE_MATCHED_MIN_INDEX
                        ),
                    }
                pair_lines.append(pair_row)
                accepted += 1
            if device.type == "mps":
                torch.mps.empty_cache()

        # Producer thread preps chunks (parquet reads, wav decode, delay pad)
        # while the main thread keeps the GPU busy encoding the previous chunk.
        # All GPU work stays on the main thread.
        chunk_q: queue.Queue = queue.Queue(maxsize=2)
        DONE = object()
        prep_err: list[BaseException] = []

        def producer() -> None:
            nonlocal kept, kept_hours
            chunk: list[dict] = []
            try:
                for i in range(table.num_rows):
                    if args.limit and kept >= args.limit:
                        break
                    vi_dur = float(table.column("duration_vi_s")[i].as_py())
                    if args.max_source_duration_s and vi_dur > args.max_source_duration_s:
                        continue
                    en_dur = float(table.column("duration_en_s")[i].as_py())
                    text_vi = str(table.column("vi")[i].as_py()).strip()
                    text_en = str(table.column("en")[i].as_py()).strip()
                    key = row_key(text_vi, text_en, f"{vi_dur:.2f}", f"{en_dur:.2f}")
                    if key in skip:
                        continue
                    skip.add(key)
                    row = {
                        "id": f"phomt_s{parquet_idx:05d}r{i:05d}",
                        "split": "train",
                        "vi_audio": f"hf://{DATASET}/{shard_name}#r{i}",
                        "en_audio": f"hf://{DATASET}/{shard_name}#r{i}",
                        "vi_duration_s": f"{vi_dur:.2f}",
                        "en_duration_s": f"{en_dur:.2f}",
                        "text_vi": text_vi,
                        "text_en": text_en,
                    }
                    delay_s = target_delay_s(row, args.target_delay_ratio, args.seed)
                    delay_frames = int(round(delay_s * FRAME_RATE))
                    with (
                        tempfile.NamedTemporaryFile(suffix=".wav", dir=tmp_dir) as vi_tmp,
                        tempfile.NamedTemporaryFile(suffix=".wav", dir=tmp_dir) as en_tmp,
                    ):
                        vi_tmp.write(table.column("audio_vi")[i].as_py()["bytes"])
                        vi_tmp.flush()
                        en_tmp.write(table.column("audio_en")[i].as_py()["bytes"])
                        en_tmp.flush()
                        vi_wav = read_audio(Path(vi_tmp.name), sphn, torch, cpu)[0, 0]
                        en_raw_wav = read_audio(Path(en_tmp.name), sphn, torch, cpu)[0, 0]
                        en_wav = torch.nn.functional.pad(
                            en_raw_wav, (int(round(delay_s * SAMPLE_RATE)), 0)
                        )
                    chunk.append(
                        {
                            "row": row,
                            "delay_s": delay_s,
                            "delay_frames": delay_frames,
                            "vi_wav": vi_wav,
                            "en_wav": en_wav,
                            "en_align_wav": (
                                sphn.resample(en_raw_wav.numpy(), SAMPLE_RATE, 16_000)
                                if aligner is not None
                                else None
                            ),
                        }
                    )
                    kept += 1
                    kept_hours += vi_dur / 3600.0
                    if len(chunk) >= args.chunk_rows:
                        chunk_q.put(chunk)
                        chunk = []
                if chunk:
                    chunk_q.put(chunk)
            except BaseException as exc:
                prep_err.append(exc)
            finally:
                chunk_q.put(DONE)

        prep = threading.Thread(target=producer, daemon=True)
        prep.start()
        while (ready_chunk := chunk_q.get()) is not DONE:
            encode_chunk(ready_chunk)
        prep.join()
        if prep_err:
            raise prep_err[0]
        if not samples:
            raise RuntimeError(f"Every row in {shard_name} failed CTC alignment")

        payload = {
            "format": expected_format,
            "sample_rate": SAMPLE_RATE,
            "frame_rate": FRAME_RATE,
            "dataset_revision": DATASET_REVISION,
            "alignment_min_score": args.min_alignment_score if aligner is not None else None,
            "config": {
                "n_q": int(cfg["n_q"]),
                "dep_q": int(cfg["dep_q"]),
                "card": int(cfg["card"]),
                "text_card": int(cfg["text_card"]),
                "existing_text_padding_id": int(cfg["existing_text_padding_id"]),
            },
            "samples": samples,
        }
        save_shard(torch, payload, out_path)
        if pair_lines:
            with pairs_path.open("a", encoding="utf-8") as fh:
                for row in pair_lines:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if reject_lines:
            with rejects_path.open("a", encoding="utf-8") as fh:
                for row in reject_lines:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if not args.keep_parquet:
            delete_from_hf_cache(local)
        if device.type == "mps":
            torch.mps.empty_cache()
        print(
            f"[w{args.worker}] {shard_name}: {len(samples)}/{table.num_rows} rows -> "
            f"{repo_display_path(out_path)} "
            f"({(kept - shard_kept_start) / max(time.monotonic() - shard_started, 1e-6):.1f} "
            f"rows/s shard, {kept / max(time.monotonic() - started, 1e-6):.1f} rows/s total; "
            f"{kept} rows / {kept_hours:.1f} VI-h)",
            flush=True,
        )
        payload = samples = pair_lines = reject_lines = table = prep = chunk_q = None
        encode_chunk = producer = None
        release_host_memory()

    elapsed = time.monotonic() - started
    print(
        f"[w{args.worker}] done: {accepted} accepted / {rejected} rejected / "
        f"{kept} attempted / {kept_hours:.1f} VI source hours in {elapsed / 60:.1f} min "
        f"({kept / max(elapsed, 1e-6):.1f} rows/s)"
    )


if __name__ == "__main__":
    main()

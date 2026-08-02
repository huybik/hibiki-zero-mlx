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
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.cache_codes import (  # noqa: E402
    CACHE_FORMAT,
    FRAME_RATE,
    SAMPLE_RATE,
    assemble_codes,
    check_device,
    encode_audio,
    require_runtime_deps,
    save_shard,
    target_delay_s,
    text_tokens,
)
from finetune.fetch_phomt import DATASET, load_hf_token, row_key  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream PhoMT parquet -> Mimi cache, no wavs on disk.")
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
    parser.add_argument("--limit", type=int, default=0, help="Stop after keeping N rows (smoke); 0=all.")
    return parser.parse_args()


def delete_from_hf_cache(path: Path) -> None:
    blob = path.resolve()
    path.unlink(missing_ok=True)
    if blob != path:
        blob.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker < args.num_workers:
        raise ValueError("--worker must be in [0, --num-workers)")
    load_hf_token()

    torch, sphn, sentencepiece, loaders = require_runtime_deps()
    device = check_device(torch, args.device)

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    shards = sorted(
        name
        for name in list_repo_files(DATASET, repo_type="dataset")
        if name.startswith("data/") and name.endswith(".parquet")
    )
    mine = [(idx, name) for idx, name in enumerate(shards) if idx % args.num_workers == args.worker]
    print(f"[w{args.worker}] {len(mine)}/{len(shards)} parquet shards assigned")

    skip: set[tuple[str, str, str, str]] = set()
    for pairs_path in args.skip_pairs:
        for row in read_pair_file(pairs_path):
            skip.add(row_key(row["text_vi"], row["text_en"], row["vi_duration_s"], row["en_duration_s"]))
    if skip:
        print(f"[w{args.worker}] skipping {len(skip)} already-cached rows")

    cfg = read_json(args.config_path)
    mimi_weight = require_file(args.mimi_weight, "Mimi weight")
    tokenizer = sentencepiece.SentencePieceProcessor(str(require_file(args.tokenizer, "text tokenizer")))
    num_codebooks = max(int(cfg["dep_q"]), int(cfg["n_q"]) - int(cfg["dep_q"]))
    mimi = loaders.get_mimi(mimi_weight, num_codebooks=num_codebooks, device=device)
    if int(cfg["card"]) != int(mimi.cardinality):
        raise RuntimeError(f"Config card={cfg['card']} != Mimi cardinality={mimi.cardinality}")

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / f"pairs_w{args.worker}.jsonl"
    shm = Path("/dev/shm")
    tmp_dir = str(shm) if shm.is_dir() else None

    kept = 0
    kept_hours = 0.0
    for parquet_idx, shard_name in mine:
        if args.limit and kept >= args.limit:
            break
        out_path = out_dir / f"shard_{parquet_idx:05d}.pt"
        if out_path.exists():
            continue
        local = Path(hf_hub_download(DATASET, shard_name, repo_type="dataset"))
        table = pq.ParquetFile(local).read()
        samples = []
        pair_lines = []
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
                vi_codes = encode_audio(Path(vi_tmp.name), mimi, sphn, torch, device)
                en_codes = encode_audio(Path(en_tmp.name), mimi, sphn, torch, device, left_pad_s=delay_s)
            tokens = text_tokens(text_en, tokenizer)
            codes = assemble_codes(torch, row, vi_codes, en_codes, tokens, cfg, delay_frames)
            samples.append(
                {
                    "id": row["id"],
                    "split": row["split"],
                    "codes": codes,
                    "frames": int(codes.shape[1]),
                    "vi_frames": int(vi_codes.shape[1]),
                    "en_frames": int(en_codes.shape[1]),
                    "text_tokens": len(tokens),
                    "target_delay_s": delay_s,
                    "target_delay_frames": delay_frames,
                    "vi_audio": row["vi_audio"],
                    "en_audio": row["en_audio"],
                    "text_en": text_en,
                    "text_vi": text_vi,
                }
            )
            pair_lines.append(row)
            kept += 1
            kept_hours += vi_dur / 3600.0

        payload = {
            "format": CACHE_FORMAT,
            "sample_rate": SAMPLE_RATE,
            "frame_rate": FRAME_RATE,
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
            import json

            with pairs_path.open("a", encoding="utf-8") as fh:
                for row in pair_lines:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if not args.keep_parquet:
            delete_from_hf_cache(local)
        if device.type == "mps":
            torch.mps.empty_cache()
        print(
            f"[w{args.worker}] {shard_name}: {len(samples)}/{table.num_rows} rows -> "
            f"{repo_display_path(out_path)} (total {kept} rows / {kept_hours:.1f} VI-h)",
            flush=True,
        )

    print(f"[w{args.worker}] done: {kept} rows / {kept_hours:.1f} VI source hours")


if __name__ == "__main__":
    main()

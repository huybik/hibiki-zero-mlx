#!/usr/bin/env python
"""Fetch anquachdev/PhoMT-en-vi-speech into the vi->en pair pipeline.

Downloads the parquet shards, writes the paired EN/VI WAVs to
remote_dataset/phomt_en_vi/{en,vi}/, and emits a pair jsonl compatible with
cache_codes.py (same 8 fields as build_pairs.py). Complements the FLEURS pairs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetune.utils import DEFAULT_PAIRS_DIR, repo_display_path, write_pair_file  # noqa: E402

DATASET = "anquachdev/PhoMT-en-vi-speech"
DEFAULT_AUDIO_DIR = REPO_ROOT / "remote_dataset" / "phomt_en_vi"


def load_hf_token() -> None:
    """Populate HF_TOKEN from .env and steer the cache off any dead mount."""
    if not os.environ.get("HF_TOKEN"):
        env = REPO_ROOT / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()
    # HF_HOME in this env may point at a dead mount; anchor on the repo cache.
    if not Path(os.environ.get("HF_HOME", "")).is_dir():
        os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch PhoMT-en-vi-speech into pair files.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_PAIRS_DIR / "phomt_train.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--max-source-duration-s",
        type=float,
        default=25.0,
        help="Drop rows with VI audio longer than this (protects MPS memory). 0 disables.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after keeping N pairs; 0=all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_hf_token()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files

    en_dir = args.audio_dir / "en"
    vi_dir = args.audio_dir / "vi"
    en_dir.mkdir(parents=True, exist_ok=True)
    vi_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(
        name
        for name in list_repo_files(DATASET, repo_type="dataset")
        if name.startswith("data/") and name.endswith(".parquet")
    )
    print(f"{len(shards)} parquet shards in {DATASET}")

    rows: list[dict[str, str]] = []
    dropped = 0
    for shard in shards:
        if args.limit and len(rows) >= args.limit:
            break
        path = hf_hub_download(DATASET, shard, repo_type="dataset")
        table = pq.ParquetFile(path).read()
        for i in range(table.num_rows):
            if args.limit and len(rows) >= args.limit:
                break
            vi_dur = float(table.column("duration_vi_s")[i].as_py())
            if args.max_source_duration_s and vi_dur > args.max_source_duration_s:
                dropped += 1
                continue
            idx = len(rows)
            stem = f"phomt_{idx:05d}"
            en_wav = en_dir / f"{stem}.wav"
            vi_wav = vi_dir / f"{stem}.wav"
            en_wav.write_bytes(table.column("audio_en")[i].as_py()["bytes"])
            vi_wav.write_bytes(table.column("audio_vi")[i].as_py()["bytes"])
            rows.append(
                {
                    "id": stem,
                    "split": args.split,
                    "vi_audio": repo_display_path(vi_wav),
                    "en_audio": repo_display_path(en_wav),
                    "vi_duration_s": f"{vi_dur:.2f}",
                    "en_duration_s": f"{float(table.column('duration_en_s')[i].as_py()):.2f}",
                    "text_vi": str(table.column("vi")[i].as_py()).strip(),
                    "text_en": str(table.column("en")[i].as_py()).strip(),
                }
            )

    write_pair_file(rows, args.out, "jsonl")
    hours = sum(float(r["vi_duration_s"]) for r in rows) / 3600.0
    print(
        f"Wrote {len(rows)} pairs ({hours:.2f} VI source hours, dropped {dropped} long) -> "
        f"{repo_display_path(args.out)}"
    )


if __name__ == "__main__":
    main()

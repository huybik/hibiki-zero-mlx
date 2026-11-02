#!/usr/bin/env python
"""Build the real-Vietnamese-speech evaluation sets as `eval.py` pair files.

Writes, under --out-dir (default data/eval):

  fleurs-vi/{dev,test}.jsonl       FLEURS vi_vn read speech (16 kHz) with the English FLoRes text
                                   of the same sentence id; dev selects checkpoints, test reports.
  multimed-vi/{select,final}.jsonl MultiMed-ST Vietnamese `corrected.test` (8 kHz conversational
                                   medical speech, human-corrected English), split in two seeded
                                   halves: select gates checkpoints, final reports.

Each row holds `id`, `vi_audio` (the clip's path), `text_vi`, `text_en` and `duration_s`. These sets
are for evaluation only and never enter training. Needs huggingface_hub and pyarrow; runs on the
Mac or the pod.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from training.paths import repo_display_path, resolve_repo_path

FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_RATE = 16_000
MULTIMED_REPO = "leduckhai/MultiMed-ST"
MULTIMED_REVISION = "d8c3b300b0f2e2ec8fa2113df3bdbc25f5e64668"
MULTIMED_FILE = "vietnamese/corrected.test-00000-of-00001.parquet"
MULTIMED_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    parser.add_argument(
        "--sets", nargs="+", choices=("fleurs", "multimed"), default=["fleurs", "multimed"]
    )
    return parser.parse_args()


def download(repo: str, revision: str, filename: str) -> Path:
    return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    print(f"{repo_display_path(path)}: {len(rows)} rows", flush=True)


def fleurs_tsv(path: Path) -> list[dict[str, str]]:
    """FLEURS rows: sentence id, wav name, raw text, normalized text, characters, samples, gender."""
    columns = ("sentence_id", "file_name", "raw", "normalized", "chars", "samples", "gender")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        return [dict(zip(columns, line, strict=True)) for line in reader]


def build_fleurs(out_dir: Path) -> None:
    english: dict[str, str] = {}
    for split in ("train", "dev", "test"):
        for row in fleurs_tsv(download(FLEURS_REPO, FLEURS_REVISION, f"data/en_us/{split}.tsv")):
            if english.setdefault(row["sentence_id"], row["raw"]) != row["raw"]:
                raise RuntimeError(f"FLEURS en_us sentence {row['sentence_id']} has two texts")
    for split in ("dev", "test"):
        audio_dir = out_dir / "fleurs-vi" / "audio" / split
        audio_dir.mkdir(parents=True, exist_ok=True)
        archive = download(FLEURS_REPO, FLEURS_REVISION, f"data/vi_vn/audio/{split}.tar.gz")
        with tarfile.open(archive) as tar:
            for member in tar:
                if member.isfile() and member.name.endswith(".wav"):
                    target = audio_dir / Path(member.name).name
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Cannot read {member.name} in {archive}")
                    target.write_bytes(source.read())
        rows, missing = [], 0
        for row in fleurs_tsv(download(FLEURS_REPO, FLEURS_REVISION, f"data/vi_vn/{split}.tsv")):
            if row["sentence_id"] not in english:
                missing += 1
                continue
            wav = audio_dir / row["file_name"]
            if not wav.is_file():
                raise RuntimeError(f"FLEURS vi_vn {split} lacks the audio {row['file_name']}")
            rows.append(
                {
                    "id": f"fleurs-vi/{split}/{Path(row['file_name']).stem}",
                    "vi_audio": repo_display_path(wav),
                    "text_vi": row["raw"],
                    "text_en": english[row["sentence_id"]],
                    "duration_s": int(row["samples"]) / FLEURS_RATE,
                    "sentence_id": row["sentence_id"],
                }
            )
        if missing:
            print(f"fleurs-vi/{split}: {missing} rows have no English sentence, skipped")
        write_rows(out_dir / "fleurs-vi" / f"{split}.jsonl", rows)


def build_multimed(out_dir: Path) -> None:
    table = pq.read_table(
        download(MULTIMED_REPO, MULTIMED_REVISION, MULTIMED_FILE),
        columns=["audio", "duration", "text", "English"],
    )
    audio_dir = out_dir / "multimed-vi" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row in enumerate(table.to_pylist()):
        if not row["English"].strip() or not row["text"].strip():
            raise RuntimeError(f"MultiMed-ST row {index} has an empty transcript or translation")
        audio = audio_dir / f"{index:05d}.ogg"
        audio.write_bytes(row["audio"]["bytes"])
        rows.append(
            {
                "id": f"multimed-vi/{index:05d}",
                "vi_audio": repo_display_path(audio),
                "text_vi": row["text"],
                "text_en": row["English"],
                "duration_s": row["duration"],
                "source_segment": row["audio"]["path"],
            }
        )
    order = list(range(len(rows)))
    random.Random(MULTIMED_SEED).shuffle(order)
    half = len(order) // 2
    for name, indexes in (("select", order[:half]), ("final", order[half:])):
        write_rows(out_dir / "multimed-vi" / f"{name}.jsonl", [rows[i] for i in sorted(indexes)])


def main() -> None:
    args = parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    if "fleurs" in args.sets:
        build_fleurs(out_dir)
    if "multimed" in args.sets:
        build_multimed(out_dir)


if __name__ == "__main__":
    main()

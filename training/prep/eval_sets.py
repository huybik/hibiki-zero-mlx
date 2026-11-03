#!/usr/bin/env python
"""Build the evaluation sets as `eval.py` pair files.

Writes, under --out-dir (default data/eval):

  fleurs-vi/{dev,test}.jsonl       FLEURS vi_vn read speech (16 kHz) with the English FLoRes text
                                   of the same sentence id; dev selects checkpoints, test reports.
  multimed-vi/{select,final}.jsonl MultiMed-ST Vietnamese `corrected.test` (8 kHz conversational
                                   medical speech, human-corrected English), split in two seeded
                                   halves: select gates checkpoints, final reports.
  en-vi-val/{cv27,libriheavy}.jsonl  seeded samples of the EN-VI dataset's validation split (Common
                                   Voice dev and Libriheavy dev, Vietnamese TTS in the English
                                   speaker's voice): the training data's own domain, held out.
  en-vi-val-noisy/{cv27,libriheavy}.jsonl  the same rows with every clip under `augment`'s
                                   simulated recording conditions (24 kHz WAV, resampled as eval.py
                                   resamples the clean clips), against which the clean set shows
                                   what the channel alone costs.

Each row holds `id`, `vi_audio` (the clip's path), `text_vi`, `text_en` and `duration_s`. These sets
are for evaluation only and never enter training. Needs huggingface_hub, pyarrow, sphn, soxr and
torch; runs on the Mac or the pod.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import random
import tarfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import sphn
import torch
from huggingface_hub import hf_hub_download

from training.paths import repo_display_path, resolve_repo_path
from training.prep import augment

FLEURS_REPO = "google/fleurs"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_RATE = 16_000
MULTIMED_REPO = "leduckhai/MultiMed-ST"
MULTIMED_REVISION = "d8c3b300b0f2e2ec8fa2113df3bdbc25f5e64668"
MULTIMED_FILE = "vietnamese/corrected.test-00000-of-00001.parquet"
MULTIMED_SEED = 0
EN_VI_REPO = "anquachdev/common-vi-en"
EN_VI_REVISION = "2ce8777d122123609722814f776f0505ea966ca5"
EN_VI_SOURCES = {
    "cv27": ["cv27-00000"],
    "libriheavy": ["libriheavy-dev-00000", "libriheavy-dev-00001"],
}
EN_VI_ROWS = 180  # per source
EN_VI_SEED = 0
OPUS_RATE = 48_000
NOISY_SEED = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=("fleurs", "multimed", "en-vi-val", "en-vi-val-noisy"),
        default=["fleurs", "multimed", "en-vi-val"],
    )
    parser.add_argument(
        "--bank-dir",
        type=Path,
        default=Path("data/rirs_noises"),
        help="Where `augment` keeps OpenSLR 28 (downloaded on first use); for en-vi-val-noisy.",
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


def opus_pre_skip(data: bytes, name: str) -> int:
    """The samples a decoder drops from the start of an Ogg Opus stream (RFC 7845 section 5.1).
    The stream must be mono and without output gain, which sphn would not apply."""
    head = data.find(b"OpusHead")
    if head < 0:
        raise RuntimeError(f"{name}: not an Ogg Opus stream")
    channels = data[head + 9]
    gain = int.from_bytes(data[head + 16 : head + 18], "little", signed=True)
    if channels != 1 or gain:
        raise RuntimeError(f"{name}: Opus stream has {channels} channels and output gain {gain}")
    return int.from_bytes(data[head + 10 : head + 12], "little")


def decode_opus(data: bytes, duration_s: float, name: str) -> Any:
    """Decode a 48 kHz Ogg Opus clip to its stored duration. sphn returns every packet's samples:
    the encoder's pre-skip, which this drops, and the last packet's padding, which the cut at the
    stored duration drops. That matches PyAV, the cache builder's decoder, to the sample. The
    granule positions cannot serve instead: the TTS encoder counts the pre-skip twice, so they
    run 312 samples past the audio."""
    pre_skip = opus_pre_skip(data, name)
    wav, rate = sphn.read_opus_bytes(data)
    if rate != OPUS_RATE:
        raise RuntimeError(f"{name}: Opus decoded at {rate} Hz")
    wav = wav[:, pre_skip:]
    samples = round(duration_s * OPUS_RATE)
    # As the cache builder's duration gate: a truncated clip decodes without error, only shorter.
    # A whole one holds its stored samples and less than one 20 ms packet of padding.
    if not 0 <= wav.shape[-1] - samples < 0.02 * OPUS_RATE:
        decoded = wav.shape[-1] / OPUS_RATE
        raise RuntimeError(f"{name}: decoded {decoded:.3f} s, stored {duration_s:.3f} s")
    return wav[:, :samples]


def build_en_vi_val(out_dir: Path) -> None:
    for source, shards in EN_VI_SOURCES.items():
        tables = [
            pq.read_table(
                download(EN_VI_REPO, EN_VI_REVISION, f"data/validation/{shard}.parquet"),
                columns=["id", "speaker", "text_vi", "text_en", "vi_audio", "vi_duration_s"],
            )
            for shard in shards
        ]
        locations = [
            (shard, index) for shard, table in zip(shards, tables) for index in range(len(table))
        ]
        random.Random(EN_VI_SEED).shuffle(locations)
        audio_dir = out_dir / "en-vi-val" / "audio" / source
        audio_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for shard, index in sorted(locations[:EN_VI_ROWS]):
            row = tables[shards.index(shard)].slice(index, 1).to_pylist()[0]
            wav = audio_dir / f"{row['id'].replace('/', '_')}.wav"
            audio = decode_opus(row["vi_audio"]["bytes"], row["vi_duration_s"], row["id"])
            sphn.write_wav(wav, audio, OPUS_RATE)
            rows.append(
                {
                    "id": f"en-vi-val/{row['id']}",
                    "vi_audio": repo_display_path(wav),
                    "text_vi": row["text_vi"],
                    "text_en": row["text_en"],
                    "duration_s": round(row["vi_duration_s"], 2),
                    "speaker": row["speaker"],
                    "source_row": f"data/validation/{shard}.parquet#r{index}",
                }
            )
        write_rows(out_dir / "en-vi-val" / f"{source}.jsonl", rows)


def build_en_vi_val_noisy(out_dir: Path, bank_dir: Path) -> None:
    bank = augment.Bank(augment.fetch_bank(bank_dir), "cpu")
    for source in EN_VI_SOURCES:
        audio_dir = out_dir / "en-vi-val-noisy" / "audio" / source
        audio_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        clean = (out_dir / "en-vi-val" / f"{source}.jsonl").read_text().splitlines()
        for row in map(json.loads, clean):
            # Resampled as eval.py resamples the clean clip (hibiki_helpers.audio_read), so the
            # two sets differ by the conditions alone.
            wav, rate = sphn.read(resolve_repo_path(Path(row["vi_audio"])))
            clip = torch.from_numpy(sphn.resample(wav, rate, augment.SAMPLE_RATE)[0])
            conditions = bank.draw(f"{NOISY_SEED}:{row['id']}")
            noisy = bank.apply(clip, conditions)
            path = audio_dir / Path(row["vi_audio"]).name
            sphn.write_wav(path, noisy.numpy()[None], augment.SAMPLE_RATE)
            name = row["id"].removeprefix("en-vi-val/")
            rows.append(
                row
                | {"id": f"en-vi-val-noisy/{name}", "vi_audio": repo_display_path(path)}
                | {"conditions": dataclasses.asdict(conditions)}
            )
        write_rows(out_dir / "en-vi-val-noisy" / f"{source}.jsonl", rows)


def main() -> None:
    args = parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    if "fleurs" in args.sets:
        build_fleurs(out_dir)
    if "multimed" in args.sets:
        build_multimed(out_dir)
    if "en-vi-val" in args.sets:
        build_en_vi_val(out_dir)
    if "en-vi-val-noisy" in args.sets:
        build_en_vi_val_noisy(out_dir, resolve_repo_path(args.bank_dir))


if __name__ == "__main__":
    main()

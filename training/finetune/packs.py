"""Per-shard packs of a grounded cache, the only form a run reads its rows in (`rows.load_rows`):
each shard's rows as one flat int32 array plus an index, their docs, their text tracks, and the
rows its build rejected for the aligner's frame grids or whose English is OCR-damaged
(`pack_shard`), in `<cache-dir>/.packed/<shard stem>/`.

    python -m training.finetune.packs [--drop-shards] CACHE_DIR...

builds the packs before the first run; --drop-shards then deletes each packed shard_*.pt (packs
only: the trainer reads nothing else), for caches whose shards are on the Hub.
"""
from __future__ import annotations

import dataclasses
import json
import multiprocessing
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.cache.timing import CAPACITY_REASONS
from training.paths import CACHE_FORMAT, DEFAULT_DIRECTION, DEFAULT_TOKENIZER, cpu_budget

FRAME_RATE = 12.5
PHOMT_ID = re.compile(r"^(phomt_s\d+)r(\d+)$")  # consecutive rows of one source chunk: doc, seq
PACK_WORKERS = 32


@dataclass(frozen=True)
class Layout:
    """The cache's code layout, shared by every shard of a run."""

    dep_q: int  # target codebooks, rows 1..dep_q; the source follows
    card: int  # audio cardinality, also the source EOS
    pad_id: int  # text PAD

    @property
    def rows(self) -> int:
        return 1 + 2 * self.dep_q


# Libriheavy's text is its books' text, and some books were scanned badly: the reader says
# "yesterday" where the text reads "ye^erday" or "suppHed". That text is a wrong target for
# Vietnamese to English and was a wrong source for the Vietnamese translation, so `rows.load_rows`
# leaves such rows out of both directions (en-speech-corpus.md#quality-filters). Packs record the
# rows it matches when they are built: changing it needs a new PACK_VERSION, which on a box that
# dropped its shards (`--drop-shards`) means fetching them all again.
OCR_MARK = re.compile(
    r"[\^{}|\\<>~•¬©®¥■«»]"  # a glyph no reader says: ye^erday, dumb¬ founded, {348} (a page number)
    r"|(?<!\d)(?<!\d )[£¢](?!\s?\d)"  # a currency sign without an amount: o£ hie (not 12£, £5)
    r"|â€|‚Ä|ŌĆ|├"  # UTF-8 text decoded in another encoding: Dante‚Äôs, IŌĆÖm
    r"|\b(?!Mc|Mac|De|Du|La|Le|Van|Von|Fitz)[A-Za-z]*[a-z][A-Z]"  # a capital inside a word: suppHed, caveNo
)


def ocr_damaged(sample: dict[str, Any]) -> bool:
    """Whether a cache row is a Libriheavy row whose English text carries an `OCR_MARK`. Other
    sources' English is typed (Common Voice, PhoMT) or machine-made (the tranche), and it writes
    names such as YouTube."""
    return str(sample["id"]).startswith("libriheavy/") and OCR_MARK.search(sample["text_en"]) is not None


# 3: docs.json; 4: the shard's direction, the index by source and target; 5: tokenizer, text.bin;
# 6: the ids the shard's build rejected for the aligner's frame grids (`capacity`); 7: the ids
# whose English is OCR-damaged (`ocr`)
PACK_VERSION = 7
PACK_DIR = ".packed"  # beside the shards: <cache-dir>/.packed/<shard stem>/


def pack_dir(shard_path: Path) -> Path:
    return shard_path.parent / PACK_DIR / shard_path.stem


def _pack_stamp(shard_path: Path) -> dict[str, int]:
    stat = shard_path.stat()
    return {"version": PACK_VERSION, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _pack_is_current(shard_path: Path) -> bool:
    meta = pack_dir(shard_path) / "pack.json"
    return meta.is_file() and json.loads(meta.read_text())["stamp"] == _pack_stamp(shard_path)


def pack_shard(shard_path: Path) -> None:
    """Write a shard's rows as one flat int32 array plus an index, which loads memory-mapped
    in milliseconds; unpickling the shard's per-row tensors takes ~0.2 s. The rows' text tracks
    also go to `text.bin` in row order, so a run on another tokenizer reads ~3% of the codes to
    check them (`rows.load_rows`)."""
    payload = torch.load(shard_path, map_location="cpu")
    if payload.get("format") != CACHE_FORMAT:
        raise RuntimeError(f"Unsupported cache format in {shard_path}")
    if float(payload["frame_rate"]) != FRAME_RATE:
        raise RuntimeError(f"{shard_path} has frame rate {payload['frame_rate']}")
    config = payload["config"]
    layout = Layout(
        dep_q=int(config["dep_q"]),
        card=int(config["card"]),
        pad_id=int(config["existing_text_padding_id"]),
    )
    if 2 * layout.dep_q != int(config["n_q"]):
        raise RuntimeError(f"{shard_path} has unequal source and target codebooks")
    # A sample's vi_frames and en_frames are each language's clip; the direction says which is heard.
    direction = payload.get("direction", DEFAULT_DIRECTION)
    source, target = direction.split("-")
    chunks, texts, index, ids, docs = [], [], [], [], []
    offset = 0
    for sample in payload["samples"]:
        codes = sample["codes"]
        if codes.dtype != torch.int32 or codes.ndim != 2 or codes.shape[0] != layout.rows:
            raise RuntimeError(
                f"{shard_path} id={sample['id']} codes are {codes.dtype} {tuple(codes.shape)}"
            )
        chunks.append(codes.reshape(-1).numpy())
        texts.append(codes[0].numpy())
        frames = int(codes.shape[1])
        index.append(
            (
                offset,
                frames,
                int(sample[f"{source}_frames"]),
                int(sample[f"{target}_frames"]),
                int(sample["target_delay_frames"]),
            )
        )
        ids.append(str(sample["id"]))
        match = PHOMT_ID.match(str(sample["id"]))
        if "doc" in sample:
            docs.append([str(sample["doc"]), int(sample["seq"])])
        elif match:
            docs.append([match.group(1), int(match.group(2))])
        else:
            docs.append(None)
        offset += codes.numel()
    target = pack_dir(shard_path)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    (np.concatenate(chunks) if chunks else np.empty(0, np.int32)).tofile(temporary / "codes.bin")
    (np.concatenate(texts) if texts else np.empty(0, np.int32)).tofile(temporary / "text.bin")
    np.save(temporary / "index.npy", np.array(index, dtype=np.int64).reshape(-1, 5))
    (temporary / "ids.json").write_text(json.dumps(ids))
    (temporary / "docs.json").write_text(json.dumps(docs))
    (temporary / "pack.json").write_text(
        json.dumps({"stamp": _pack_stamp(shard_path), "layout": dataclasses.asdict(layout), "direction": direction,
                    "tokenizer": payload.get("tokenizer", DEFAULT_TOKENIZER.name),
                    "capacity": [str(item["id"]) for item in payload.get("rejected", [])
                                 if any(reason in item["reason"] for reason in CAPACITY_REASONS)],
                    "ocr": [str(sample["id"]) for sample in payload["samples"] if ocr_damaged(sample)]})
    )
    shutil.rmtree(target, ignore_errors=True)
    temporary.rename(target)


def pack(cache_dirs: list[Path]) -> list[Path]:
    """The shards of `cache_dirs`, each with a current pack (`pack_shard`), packed in parallel
    where a shard is new, changed or packed by another `PACK_VERSION`. A shard is a
    `shard_*.pt`, or a pack whose `.pt` `--drop-shards` deleted (packs only on the box): that pack
    is used as it is, and must be of this `PACK_VERSION`; a pack left without its shard any other
    way raises, so deleting a shard never leaves its rows in a run unnoticed. Either way a
    shard's path is `<cache-dir>/<stem>.pt`, in name order, so dropping files keeps the rows' order."""
    shard_paths = []
    for d in cache_dirs:
        built = {path.name for path in d.glob("shard_*.pt") if path.exists()}  # not a broken link
        dropped = [d / f"{p.name}.pt" for p in (d / PACK_DIR).glob("shard_*") if f"{p.name}.pt" not in built]
        for path in dropped:
            meta = pack_dir(path) / "pack.json"
            info = json.loads(meta.read_text()) if meta.is_file() else {}
            if not info.get("dropped"):
                raise RuntimeError(f"{pack_dir(path)} has no shard {path.name} and was not dropped by --drop-shards: "
                                   f"fetch the shard from the Hub, or delete the pack")
            if info["stamp"]["version"] != PACK_VERSION:
                raise RuntimeError(f"{pack_dir(path)} is not a version {PACK_VERSION} pack and {path.name} is gone: "
                                   f"fetch it from the Hub to pack it again")
        shard_paths += sorted([d / name for name in built] + dropped)
    stale = [path for path in shard_paths if path.exists() and not _pack_is_current(path)]
    if stale:
        print(f"[streams] packing {len(stale)} of {len(shard_paths)} cache shards", flush=True)
        # Forked like train.py's DataLoader workers (named: 3.14 defaults to forkserver): packing
        # touches only files and host memory, and fork neither re-runs the caller's script, pickles
        # the arguments' state nor starts a fresh BLAS thread pool per worker.
        workers = min(PACK_WORKERS, max(1, int(cpu_budget())), len(stale))
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            list(pool.map(pack_shard, stale))
    return shard_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--drop-shards", action="store_true")
    parser.add_argument("cache_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    packed = pack(args.cache_dirs)
    print(f"{len(packed):,} shards' packs current in {len(args.cache_dirs)} cache directories")
    if args.drop_shards:
        built = [path for path in packed if path.exists()]
        freed = sum(path.lstat().st_size for path in built)  # a link frees only itself
        for path in built:
            meta = pack_dir(path) / "pack.json"  # marked first: a pack without its shard loads only so
            meta.write_text(json.dumps(json.loads(meta.read_text()) | {"dropped": True}))
            path.unlink()
        print(f"dropped {len(built):,} shards, {freed / 1e9:.1f} GB")

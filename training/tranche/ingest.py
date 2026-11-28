#!/usr/bin/env python
"""Ingest: the tranche's real Vietnamese segments, one parquet per source shard, in the corpus's
own order ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#pipeline)), into the tranche
repos' layout (training/tranche/repo.py, which also holds the segments' schema).

  segments yodas     download each YODAS3 tar at its pinned commit, cut its rows from whole videos
                     (training/tranche/yodas.py), hear them on the ASR server and write
                     <root>/yodas/segments/<tar>.parquet and its asr_vi; the download is deleted
  push <source>      upload <root>/<source>/segments, or with --part another stage's folder
                     (text_en, en_audio, asr_vi; cache: <root>/cache/<source>, to CACHE_REPO), to
                     the tranche repo (a rerun skips files already committed), only while it is gated

A rerun skips shards already written. In the ingest env, beside the ASR server; from the repo root
on the box:
  /workspace/envs/ingest/bin/python -m training.tranche.ingest segments yodas --shards 0:25
"""
from __future__ import annotations

import argparse
from pathlib import Path

from training.tranche import yodas
from training.tranche.repo import ROOT, SOURCES, push


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("segments")
    p.add_argument("source", choices=["yodas"])
    p.add_argument("--shards", type=lambda s: slice(*(int(n) if n else None for n in s.split(":"))),
                   default=slice(None), help="a slice of the source's tars, e.g. 0:23")
    p.add_argument("--workers", type=int, default=None,
                   help="the cutting processes, the CPU quota less 3 by default")
    p.add_argument("--push", action=argparse.BooleanOptionalAction, default=True,
                   help="push each tar's segments and asr_vi as it is written")
    p = steps.add_parser("push")
    p.add_argument("source", choices=sorted(SOURCES))
    p.add_argument("--part", choices=["segments", "text_en", "en_audio", "asr_vi", "cache"], default="segments",
                   help="the folder to upload: the segments, their translations (translate.py), their "
                        "English speech (voice.py), their Vietnamese ASR (ingest) or their "
                        "training cache (cache.py)")
    args = parser.parse_args()
    if args.step == "segments":
        yodas.segments(args)
    else:
        push(args.root, args.source, args.part)


if __name__ == "__main__":
    main()

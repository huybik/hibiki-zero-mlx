#!/usr/bin/env python
"""Publish the grounded caches to the cache repo while they are built, on two machines.

`cache_codes` goes top down through the source shards on the Mac and bottom up (--bottom-up) on
another machine; beside each runs this step, with the same --out-dir and --shards. A pass commits
the cache shards the Hub lacks to `<repo>/<out-dir name>/<split>/`, a few GB at a time, then
downloads those it has and this machine lacks, built on the other, so the builder here skips them
and the two meet in between. The Hub is the record of progress: a shard there is never sent again,
so a stop loses only the unfinished commit. With --follow it looks again every 10 min until every
source shard is cached here and on the Hub.

Run from the repo root with HF_TOKEN exported (write access to --repo):
  python -m training.prep.publish_caches --out-dir DIR --shards GLOB... [--follow]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# As in the builder: macOS's default Arrow pool keeps freed buffers.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import CommitOperationAdd, HfApi, RepoFile  # noqa: E402
from huggingface_hub.errors import HfHubHTTPError  # noqa: E402
from huggingface_hub.utils import disable_progress_bars  # noqa: E402

from training.paths import durable, exclusive  # noqa: E402
from training.prep.cache_codes import DATASET_DIR, SHARDS, cache_path, source_shards  # noqa: E402
from training.prep.publish_en_vi_speech import COMMIT_BYTES, retried  # noqa: E402
from training.prep.translate_en_speech import log  # noqa: E402

REPO = "huybik/hibiki-zero-vi-full-sft"  # a dataset repo; the PhoMT cache is there too
FOLLOW_WAIT = 600  # a cache shard is built every ~4 min


def settings(shard: dict[str, Any]) -> dict[str, Any]:
    """What a cache shard holds besides its rows and source: format, rates, gates, model config."""
    return {k: v for k, v in shard.items() if k not in ("samples", "rejected", "source")}


def publish(out_dir: Path, sources: list[Path], repo: str, follow: bool) -> None:
    """Commit the local cache shards the Hub lacks, then bring in the Hub's shards missing here."""
    disable_progress_bars()  # in a log, their redraws come to ~1 MB a commit
    api = HfApi()
    caches = {cache_path(out_dir, source): source for source in sources}

    def hub(path: Path) -> str:
        return f"{out_dir.name}/{path.relative_to(out_dir).as_posix()}"

    def commit_next() -> tuple[set[str], int]:
        """List the Hub and commit the next batch on top of that listing's revision, so a commit
        landing in between (the other machine's) fails this one, and it lists again at once.
        Returns the listing and how many shards were committed (none when nothing is left)."""
        while True:
            sha = api.repo_info(repo, repo_type="dataset").sha
            remote = {
                f.path
                for f in api.list_repo_tree(repo, recursive=True, revision=sha, repo_type="dataset")
                if isinstance(f, RepoFile) and f.path.startswith(f"{out_dir.name}/")
            }
            todo = [p for p in caches if p.exists() and hub(p) not in remote]
            batch, size = [], 0
            for p in todo:
                if batch and size + p.stat().st_size > COMMIT_BYTES:
                    break
                batch.append(p)
                size += p.stat().st_size
            if not batch:
                return remote, 0
            try:
                api.create_commit(
                    repo,
                    [CommitOperationAdd(hub(p), p) for p in batch],
                    repo_type="dataset",
                    commit_message=f"Add {len(batch)} cache shards from {hub(batch[0])}",
                    parent_commit=sha,
                )
            except HfHubHTTPError as e:
                if e.response.status_code not in (409, 412):  # anything but a moved branch
                    raise
                log(f"the Hub moved past {sha[:7]} during the commit; listing again")
                continue
            log(
                f"committed {len(batch)} shards from {hub(batch[0])} ({size / 1e9:.1f} GB), "
                f"{len(todo) - len(batch)} left"
            )
            return remote, len(batch)

    while True:
        if not out_dir.is_dir():  # a dropped drive: never download onto the internal disk
            raise SystemExit(f"{out_dir} is gone")
        while True:
            remote, committed = retried("committing to the Hub", commit_next)
            if not committed:
                break
        missing = [p for p in caches if not p.exists() and hub(p) in remote]
        built = next((p for p in caches if p.exists()), None)
        if missing and built:  # pulled shards are checked against one built here
            reference = settings(torch.load(built, weights_only=True, mmap=True))
            for path in missing:
                pull(api, repo, out_dir, hub(path), path, caches[path], reference)
        if not follow or all(p.exists() and hub(p) in remote for p in caches):
            return
        time.sleep(FOLLOW_WAIT)


def pull(
    api: HfApi,
    repo: str,
    out_dir: Path,
    name: str,
    path: Path,
    source: Path,
    reference: dict[str, Any],
) -> None:
    """Put the Hub's cache shard of `source`, built on the other machine, in place once it checks
    out: the settings of the shards built here, the same source file at the same size, and every
    source row once, kept or rejected."""
    got = Path(
        retried(
            f"downloading {name}",
            lambda: api.hf_hub_download(
                repo, name, repo_type="dataset", local_dir=out_dir / ".pull"
            ),
        )
    )
    shard = torch.load(got, weights_only=True, mmap=True)
    rows = [s["id"] for s in shard["samples"]] + [r["id"] for r in shard["rejected"]]
    expected = {"file": f"data/{source.parent.name}/{source.name}", "bytes": source.stat().st_size}
    if not (
        settings(shard) == reference
        and shard["source"] == expected
        and sorted(rows) == sorted(pq.read_table(source, columns=["id"])["id"].to_pylist())
    ):
        raise RuntimeError(f"the Hub's {name} is not the cache of {source}; it is left in {got}")
    del shard  # its mmap
    if path.exists():  # built here while it downloaded: the Hub keeps its copy, this one stays
        log(f"kept {name}, built here while the Hub's copy downloaded")
    else:
        durable(got)
        path.parent.mkdir(exist_ok=True)
        got.replace(path)
        durable(path.parent)
        log(f"took {name}, built on the other machine")
    shutil.rmtree(out_dir / ".pull")  # the download's metadata


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="the builder's --out-dir; its name is the Hub prefix",
    )
    p.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    p.add_argument("--shards", nargs="+", default=SHARDS, help="the builder's --shards")
    p.add_argument("--repo", default=REPO)
    p.add_argument(
        "--follow",
        action="store_true",
        help="keep going until every source shard is cached and on the Hub",
    )
    args = p.parse_args()
    sources = source_shards(args.dataset_dir, args.shards)
    args.out_dir.mkdir(exist_ok=True)  # not its parents: a dropped drive fails here
    with exclusive(
        args.out_dir / ".publish_caches.lock", f"another publish_caches run is using {args.out_dir}"
    ):
        publish(args.out_dir, sources, args.repo, args.follow)


if __name__ == "__main__":
    sys.exit(main())

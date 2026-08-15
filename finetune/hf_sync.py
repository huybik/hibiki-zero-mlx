#!/usr/bin/env python
"""Periodically upload the newest complete checkpoint pair to a HF model repo.

Run detached alongside training so an instance recycle does not lose the latest
model and optimizer state. Existing Hub files and history are never deleted.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

POLL_S = 600
SETTLE_S = 120  # skip files still being written


def newest_pair(run_dir: Path) -> tuple[Path, Path] | None:
    trainers = sorted(run_dir.glob("trainer_step*.pt"))
    for trainer in reversed(trainers):
        step = trainer.stem.removeprefix("trainer_")
        model = run_dir / f"model_{step}.safetensors"
        if not model.is_file():
            continue
        now = time.time()
        if all(now - f.stat().st_mtime > SETTLE_S for f in (trainer, model)):
            return model, trainer
    return None


def main() -> None:
    run_dir = Path(sys.argv[1])
    repo = sys.argv[2]
    api = HfApi()
    uploaded: str | None = None
    while True:
        pair = newest_pair(run_dir)
        if pair and pair[1].name != uploaded:
            model, trainer = pair
            try:
                print(f"uploading {model.name} + {trainer.name}", flush=True)
                api.create_commit(
                    repo_id=repo,
                    repo_type="model",
                    operations=[
                        CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f))
                        for f in (model, trainer)
                    ],
                    commit_message=f"Sync {trainer.stem.removeprefix('trainer_')}",
                )
                uploaded = trainer.name
                print(f"synced step {uploaded}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive, retry next poll
                print(f"sync failed, will retry: {exc}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()

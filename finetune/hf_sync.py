#!/usr/bin/env python
"""Periodically upload the newest complete checkpoint pair to a private HF repo.

Survives instance recycle: model .safetensors + trainer .pt land on HF, older
remote checkpoints are pruned. Run detached alongside training:
    python finetune/hf_sync.py finetune/runs/vi_full huybik/hibiki-zero-vi-full-sft
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

POLL_S = 600
SETTLE_S = 120  # skip files still being written


def newest_pair(run_dir: Path) -> tuple[Path, Path] | None:
    trainers = sorted(run_dir.glob("trainer_step*.pt"))
    for trainer in reversed(trainers):
        step = trainer.stem.removeprefix("trainer_")
        adapters = list(run_dir.glob(f"*_{step}.safetensors"))
        if not adapters:
            continue
        now = time.time()
        if all(now - f.stat().st_mtime > SETTLE_S for f in (trainer, adapters[0])):
            return adapters[0], trainer
    return None


def main() -> None:
    run_dir = Path(sys.argv[1])
    repo = sys.argv[2]
    api = HfApi()
    uploaded: str | None = None
    while True:
        pair = newest_pair(run_dir)
        if pair and pair[1].name != uploaded:
            adapter, trainer = pair
            try:
                # Free quota first: HF's private-storage limit counts LFS objects
                # across history, so stale checkpoints must be deleted AND squashed
                # away before the new pair fits.
                keep = {adapter.name, trainer.name}
                for remote in api.list_repo_files(repo):
                    # Prune only root-level step checkpoints: files in subfolders
                    # (phase1/, phase2/ archives) and non-step names are permanent.
                    if (
                        "/" not in remote
                        and remote.startswith(("model_step", "trainer_step", "adapter_step"))
                        and remote.endswith((".safetensors", ".pt"))
                        and remote not in keep
                    ):
                        api.delete_file(remote, repo_id=repo)
                api.super_squash_history(repo_id=repo)
                print(f"uploading {adapter.name} + {trainer.name}", flush=True)
                for f in (adapter, trainer):
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo)
                uploaded = trainer.name
                print(f"synced step {uploaded}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive, retry next poll
                print(f"sync failed, will retry: {exc}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()

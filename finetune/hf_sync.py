#!/usr/bin/env python
"""Maintain rolling disaster-recovery checkpoints in a public HF model repo."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

POLL_S = 60
REMOTE_INTERVAL = int(os.environ.get("HIBIKI_HF_SYNC_INTERVAL", "9000"))
if REMOTE_INTERVAL <= 0:
    raise ValueError("HIBIKI_HF_SYNC_INTERVAL must be positive")
REMOTE_KEEP = 2
STAGE_ROOT = ".hf_sync"
REMOTE_ROOT = os.environ.get("HIBIKI_HF_PREFIX", "full_run").strip("/")
if not REMOTE_ROOT or ".." in REMOTE_ROOT.split("/"):
    raise ValueError("HIBIKI_HF_PREFIX must be a non-empty relative repository path")
STEP_MODEL = re.compile(r"model_step(\d+)\.safetensors$")
STEP_TRAINER = re.compile(r"trainer_step(\d+)\.pt$")
STAGED_PAIR = re.compile(r"checkpoint_step(\d+)$")
STAGED_BEST = re.compile(r"best_step(\d+)$")
RUN_METADATA = (
    "run_config.json",
    "sample_manifest.jsonl",
    "source_derangement.json",
    "source_asr.json",
    "source_asr_replay.json",
    "post_source_eos_translation.json",
    "validation_post_source_eos_translation.json",
    "full_data_receipt.json",
    "post_source_eos_extension.json",
)
RUN_ARTIFACTS = (
    "eval_derangement.json",
    "greedy_eval_log.jsonl",
    "train_log.jsonl",
    "val_log.jsonl",
)
EVAL_ARTIFACTS = (
    "metrics.json",
    "predictions.csv",
    "correct/metrics.json",
    "correct/predictions.csv",
    "shuffled/metrics.json",
    "shuffled/predictions.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync rolling recovery checkpoints to a public HF model repo."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("repo", help="Existing public model repo as owner/name.")
    parser.add_argument(
        "--watch-pid",
        type=int,
        help="Poll while this training PID lives, then sync the newest final pair and exit.",
    )
    return parser.parse_args()


def checkpoint_pairs(run_dir: Path) -> list[tuple[int, Path, Path]]:
    models = {
        int(match.group(1)): path
        for path in run_dir.glob("model_step*.safetensors")
        if (match := STEP_MODEL.fullmatch(path.name))
    }
    trainers = {
        int(match.group(1)): path
        for path in run_dir.glob("trainer_step*.pt")
        if (match := STEP_TRAINER.fullmatch(path.name))
    }
    return [(step, models[step], trainers[step]) for step in sorted(models.keys() & trainers.keys())]


def remote_path(path: str) -> str:
    return f"{REMOTE_ROOT}/{path}"


def remote_files(repo: str) -> dict[str, int]:
    prefix = f"{REMOTE_ROOT}/"
    return {
        item.rfilename.removeprefix(prefix): item.size
        for item in HfApi().model_info(repo, files_metadata=True).siblings
        if item.rfilename.startswith(prefix) and item.size is not None
    }


def remote_pairs(files: dict[str, int]) -> list[int]:
    models = {
        int(match.group(1))
        for path in files
        if (match := re.fullmatch(r"checkpoints/model_step(\d+)\.safetensors", path))
    }
    trainers = {
        int(match.group(1))
        for path in files
        if (match := re.fullmatch(r"checkpoints/trainer_step(\d+)\.pt", path))
    }
    return sorted(models & trainers)


def staged_pair_matches(files: dict[str, int], stage: Path, step: int) -> bool:
    model = stage / f"model_step{step:06d}.safetensors"
    trainer = stage / f"trainer_step{step:06d}.pt"
    return (
        files.get(f"checkpoints/{model.name}") == model.stat().st_size
        and files.get(f"checkpoints/{trainer.name}") == trainer.stat().st_size
    )


def stage_files(run_dir: Path, kind: str, step: int, sources: tuple[Path, ...]) -> Path:
    root = run_dir / STAGE_ROOT
    root.mkdir(exist_ok=True)
    target = root / f"{kind}_step{step:06d}"
    if target.is_dir():
        return target
    temp = root / f".{target.name}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    try:
        for source in sources:
            os.link(source, temp / source.name)
        temp.replace(target)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def stage_best(run_dir: Path, state: dict[str, object]) -> Path:
    step = int(state["step"])
    root = run_dir / STAGE_ROOT
    root.mkdir(exist_ok=True)
    target = root / f"best_step{step:06d}"
    if target.is_dir():
        return target
    temp = root / f".{target.name}.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir()
    try:
        model = run_dir / str(state["model"])
        os.link(model, temp / model.name)
        (temp / "best.json").write_text(
            json.dumps(state, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp.replace(target)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def remove_stage(path: Path) -> None:
    for child in path.iterdir():
        child.unlink()
    path.rmdir()


def clean_staging(run_dir: Path) -> None:
    root = run_dir / STAGE_ROOT
    if not root.is_dir():
        return
    for path in root.glob(".*.tmp"):
        shutil.rmtree(path)


def upload_pair(repo: str, stage: Path, step: int) -> None:
    model = stage / f"model_step{step:06d}.safetensors"
    trainer = stage / f"trainer_step{step:06d}.pt"
    remote_model = f"checkpoints/{model.name}"
    remote_trainer = f"checkpoints/{trainer.name}"
    print(f"Uploading recovery checkpoint step {step}...", flush=True)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model,
        path_in_repo=remote_path(remote_model),
        repo_id=repo,
        commit_message=f"Upload recovery model step {step}",
    )
    # The trainer is the pair's commit marker: publish it only after the model.
    api.upload_file(
        path_or_fileobj=trainer,
        path_in_repo=remote_path(remote_trainer),
        repo_id=repo,
        commit_message=f"Upload recovery trainer step {step}",
    )
    files = remote_files(repo)
    expected = {remote_model: model.stat().st_size, remote_trainer: trainer.stat().st_size}
    if any(files.get(path) != size for path, size in expected.items()):
        raise RuntimeError(f"remote checkpoint verification failed for step {step}")
    remove_stage(stage)
    print(f"Synced recovery checkpoint step {step}.", flush=True)


def prune_remote_pairs(run_dir: Path, repo: str) -> None:
    files = remote_files(repo)
    complete = remote_pairs(files)
    keep = set(complete[-REMOTE_KEEP:])
    if (run_dir / "post_source_eos_extension.json").is_file():
        if 1_000 not in complete:
            raise RuntimeError("remote step-1000 extension anchor is missing")
        keep.add(1_000)
    trainer_deletes: list[str] = []
    model_deletes: list[str] = []
    for path in files:
        trainer_match = re.fullmatch(r"checkpoints/trainer_step(\d+)\.pt", path)
        model_match = re.fullmatch(r"checkpoints/model_step(\d+)\.safetensors", path)
        if trainer_match and int(trainer_match.group(1)) not in keep:
            trainer_deletes.append(path)
        if model_match and int(model_match.group(1)) not in keep:
            model_deletes.append(path)
    # Remove trainer commit markers first so interruption leaves no usable-looking orphan.
    if trainer_deletes:
        HfApi().delete_files(
            repo, [remote_path(path) for path in trainer_deletes],
            commit_message="Prune old recovery trainers",
        )
    if model_deletes:
        HfApi().delete_files(
            repo, [remote_path(path) for path in model_deletes],
            commit_message="Prune old recovery models",
        )


def best_state(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / "best.json"
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    step = int(state["step"])
    model_name = str(state["model"])
    if model_name != f"best_step{step:06d}.safetensors" or not (run_dir / model_name).is_file():
        raise RuntimeError("best.json does not reference a valid sibling model")
    float(state["correct"]["bleu"])
    float(state["correct"]["chrf"])
    if not state.get("promotion_eligible"):
        raise RuntimeError("best.json references an ineligible checkpoint")
    return state


def remote_best_steps(files: dict[str, int]) -> list[int]:
    models = {
        int(match.group(1))
        for path in files
        if (match := re.fullmatch(r"best/best_step(\d+)\.safetensors", path))
    }
    markers = {
        int(match.group(1))
        for path in files
        if (match := re.fullmatch(r"best/best_step(\d+)\.json", path))
    }
    return sorted(models & markers)


def upload_best(repo: str, stage: Path, step: int) -> None:
    model = stage / f"best_step{step:06d}.safetensors"
    metadata = stage / "best.json"
    remote_model = f"best/{model.name}"
    remote_metadata = f"best/best_step{step:06d}.json"
    print(f"Uploading best model from step {step}...", flush=True)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=model,
        path_in_repo=remote_path(remote_model),
        repo_id=repo,
        commit_message=f"Upload best model step {step}",
    )
    api.upload_file(
        path_or_fileobj=metadata,
        path_in_repo=remote_path(remote_metadata),
        repo_id=repo,
        commit_message=f"Upload best metadata step {step}",
    )
    files = remote_files(repo)
    expected = {remote_model: model.stat().st_size, remote_metadata: metadata.stat().st_size}
    if any(files.get(path) != size for path, size in expected.items()):
        raise RuntimeError(f"remote best-model verification failed for step {step}")
    remove_stage(stage)
    print(f"Synced best model from step {step}.", flush=True)


def prune_remote_best(repo: str) -> None:
    files = remote_files(repo)
    complete = remote_best_steps(files)
    keep = complete[-1] if complete else None
    marker_deletes = [
        path
        for path in files
        if (match := re.fullmatch(r"best/best_step(\d+)\.json", path))
        and int(match.group(1)) != keep
    ]
    model_deletes = [
        path
        for path in files
        if (match := re.fullmatch(r"best/best_step(\d+)\.safetensors", path))
        and int(match.group(1)) != keep
    ]
    if marker_deletes:
        HfApi().delete_files(
            repo, [remote_path(path) for path in marker_deletes],
            commit_message="Prune old best metadata",
        )
    if model_deletes:
        HfApi().delete_files(
            repo, [remote_path(path) for path in model_deletes],
            commit_message="Prune old best models",
        )


def sync_run_metadata(run_dir: Path, repo: str, files: dict[str, int]) -> None:
    api = HfApi()
    for name in RUN_METADATA:
        source = run_dir / name
        if not source.is_file():
            continue
        destination = f"metadata/{name}"
        if files.get(destination) == source.stat().st_size:
            continue
        api.upload_file(
            path_or_fileobj=source,
            path_in_repo=remote_path(destination),
            repo_id=repo,
            commit_message=f"Upload run metadata {name}",
        )
        if remote_files(repo).get(destination) != source.stat().st_size:
            raise RuntimeError(f"remote run metadata verification failed for {name}")


def compact_artifact_paths(run_dir: Path) -> list[Path]:
    paths = [run_dir / name for name in RUN_ARTIFACTS if (run_dir / name).is_file()]
    for eval_dir in sorted(run_dir.glob("greedy_step[0-9][0-9][0-9][0-9][0-9][0-9]")):
        eval_paths = [eval_dir / name for name in EVAL_ARTIFACTS]
        missing = [path.relative_to(run_dir) for path in eval_paths if not path.is_file()]
        if missing:
            raise RuntimeError(f"incomplete paired evaluation artifacts: {missing}")
        paths.extend(eval_paths)
    return paths


def sync_run_artifacts(run_dir: Path, repo: str) -> None:
    paths = compact_artifact_paths(run_dir)
    if not paths:
        return
    relatives = [path.relative_to(run_dir).as_posix() for path in paths]
    HfApi().upload_folder(
        folder_path=run_dir,
        path_in_repo=remote_path("artifacts"),
        repo_id=repo,
        allow_patterns=relatives,
        commit_message="Upload training evaluation artifacts",
    )
    files = remote_files(repo)
    mismatches = {
        relative: (path.stat().st_size, files.get(f"artifacts/{relative}"))
        for path, relative in zip(paths, relatives, strict=True)
        if files.get(f"artifacts/{relative}") != path.stat().st_size
    }
    if mismatches:
        raise RuntimeError(f"remote evaluation artifact verification failed: {mismatches}")
    print(f"Synced {len(paths)} compact evaluation artifacts.", flush=True)


def sync_once(run_dir: Path, repo: str, final: bool) -> None:
    clean_staging(run_dir)
    files = remote_files(repo)
    sync_run_metadata(run_dir, repo, files)
    files = remote_files(repo)
    uploaded = set(remote_pairs(files))
    local_pairs = checkpoint_pairs(run_dir)
    stage_root = run_dir / STAGE_ROOT
    staged_paths = list(stage_root.iterdir()) if stage_root.is_dir() else []
    staged_pairs = {
        int(match.group(1)): path
        for path in staged_paths
        if (match := STAGED_PAIR.fullmatch(path.name))
    }
    for step, stage in list(staged_pairs.items()):
        if step in uploaded and staged_pair_matches(files, stage, step):
            remove_stage(stage)
            del staged_pairs[step]
        elif step in uploaded:
            uploaded.remove(step)
    remote_latest = max(uploaded, default=-1)
    wanted = [pair for pair in local_pairs if pair[0] % REMOTE_INTERVAL == 0]
    if final and local_pairs and local_pairs[-1] not in wanted:
        wanted.append(local_pairs[-1])
    wanted = [pair for pair in wanted if pair[0] > remote_latest]
    newest_pair = max([*staged_pairs, *(pair[0] for pair in wanted)], default=None)
    for step, stage in staged_pairs.items():
        if step != newest_pair or step < remote_latest:
            remove_stage(stage)
    if newest_pair is not None and newest_pair > remote_latest:
        stage = staged_pairs.get(newest_pair)
        if stage is None:
            _, model, trainer = next(pair for pair in wanted if pair[0] == newest_pair)
            stage = stage_files(run_dir, "checkpoint", newest_pair, (model, trainer))
        upload_pair(repo, stage, newest_pair)
    prune_remote_pairs(run_dir, repo)

    files = remote_files(repo)
    uploaded_best = set(remote_best_steps(files))
    staged_paths = list(stage_root.iterdir()) if stage_root.is_dir() else []
    staged_best = {
        int(match.group(1)): path
        for path in staged_paths
        if (match := STAGED_BEST.fullmatch(path.name))
    }
    for step, stage in list(staged_best.items()):
        model = stage / f"best_step{step:06d}.safetensors"
        marker = stage / "best.json"
        if (
            step in uploaded_best
            and files.get(f"best/{model.name}") == model.stat().st_size
            and files.get(f"best/best_step{step:06d}.json") == marker.stat().st_size
        ):
            remove_stage(stage)
            del staged_best[step]
        elif step in uploaded_best:
            uploaded_best.remove(step)
    remote_best = max(uploaded_best, default=-1)
    local_best = best_state(run_dir)
    candidates = [*staged_best]
    if local_best is not None and int(local_best["step"]) > remote_best:
        candidates.append(int(local_best["step"]))
    newest_best = max(candidates, default=None)
    for step, stage in staged_best.items():
        if step != newest_best or step < remote_best:
            remove_stage(stage)
    if newest_best is not None and newest_best > remote_best:
        stage = staged_best.get(newest_best)
        if stage is None:
            assert local_best is not None and int(local_best["step"]) == newest_best
            stage = stage_best(run_dir, local_best)
        upload_best(repo, stage, newest_best)
    prune_remote_best(repo)
    if final:
        sync_run_artifacts(run_dir, repo)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate_run_identity(run_dir: Path, repo: str) -> None:
    local = run_dir / "run_id.json"
    if not local.is_file():
        raise RuntimeError("missing local run_id.json")
    remote = Path(hf_hub_download(repo, remote_path("run.json")))
    if remote.read_bytes() != local.read_bytes():
        raise RuntimeError("local run identity does not match the disaster-recovery repo")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    info = HfApi().model_info(args.repo)
    if info.private:
        raise RuntimeError("disaster-recovery model repo must be public")
    validate_run_identity(run_dir, args.repo)

    if args.watch_pid is None:
        sync_once(run_dir, args.repo, final=True)
        return

    failures = 0
    while process_alive(args.watch_pid):
        try:
            sync_once(run_dir, args.repo, final=False)
            failures = 0
        except Exception as exc:  # noqa: BLE001 - retry transient network failures during training
            failures += 1
            if failures == 3:
                raise RuntimeError("checkpoint sync failed three consecutive times") from exc
            print(f"Sync failed, will retry: {exc}", flush=True)
        time.sleep(POLL_S)

    for attempt in range(1, 4):
        try:
            sync_once(run_dir, args.repo, final=True)
            return
        except Exception as exc:  # noqa: BLE001 - give the final upload bounded retries
            if attempt == 3:
                raise
            print(f"Final sync failed ({attempt}/3), will retry: {exc}", flush=True)
            time.sleep(POLL_S)


if __name__ == "__main__":
    main()

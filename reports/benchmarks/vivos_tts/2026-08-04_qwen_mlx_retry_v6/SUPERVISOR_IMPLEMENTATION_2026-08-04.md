# Retry-v6 guarded postprocess supervisor

Date: 2026-08-04. This phase added orchestration only. The four production-attested generator implementation files were not modified, and no cache, package, Hugging Face access, or manual-review waiver was performed.

## Contract

`training-data/supervise_vivos_qwen_postprocess_v6.py` is one idempotent state machine for the exact 10,950-row / 1,391-group production plan. Its live SHA-256 is `4a7acde0b8bd08340a54f758b1d057a67cc3e3b444252a0ab18213945a196df8`.

- While attempt 0 is alive it checks only the root completion sentinel and the launch-bound PID every 30 seconds. It does not scan or hash partial groups.
- After attempt 0 exits, the exact CPU validator runs once. Only exact `complete`, zero-media-error scope can enter QA.
- QA uses `/Volumes/data/envs/hibiki-vivos-qa/bin/python`; retry generation uses `/Volumes/data/envs/hibiki-vivos-mlx-0.4.7/bin/python`; validation, selection, and finalization use conda base.
- Attempt generation and MPS QA are sequential. Round 1 or 2 runs only from the immutable retry manifest emitted by the preceding frozen selector.
- Every subprocess has append-only start/spawn/completion records and a timestamped combined stdout/stderr log. Terminal artifacts are validated before a stage is skipped on resume.
- Finalization has no review file or waiver and stops at `pending_manual_review` (or terminal machine `no_go`). Downstream cache and publication are outside this supervisor.

The immutable config binds the production plan, policy, source plan, production attestation, supervisor, validator, QA, wrapper, runner, benchmark generator, compaction helper, and recurrent helper by path and SHA-256.

## Verification

```bash
/opt/homebrew/Caskroom/miniconda/base/bin/python -m py_compile training-data/supervise_vivos_qwen_postprocess_v6.py
/opt/homebrew/Caskroom/miniconda/base/bin/ruff check training-data/supervise_vivos_qwen_postprocess_v6.py
```

Both passed. A CPU-only rehearsal used the completed historical v6 artifacts: attempt 0 was exact at 64 rows / 8 groups, attempt 1 at 21 rows / 7 groups, and the archived selection reproduced GO at 50 accepted rows through round 1. The simulated state sequence ended at `pending_manual_review`. Separate boundary probes proved duplicate lock rejection and that a persisted `halted` state takes precedence over existing success artifacts. Machine-readable results are in `supervisor_validation.json`.

## Failed launch retained

The first idle supervisor launch used script SHA-256 `7e0006c7ca093074d19b404672f086fd4a55c87b6e50adf8eebd11f6a738e04c`. During the wait probe, review found that a restart could return early on an existing attempt-0 validation artifact before honoring a prior `halted` state. The supervisor alone was stopped; attempt-0 generation was untouched. No command history existed, so validation, QA, and MLX retry generation had not started. Its immutable config and launch remain under:

`/Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6_full/supervisor/`

The shared state boundary and interrupted-command bookkeeping were corrected before the replacement launch.

## Live launch

- Generation session: `hibiki_vivos_qwen_v6_attempt0_20260804`
- Supervisor session: `hibiki_vivos_qwen_v6_postprocess_r1_20260804`
- Supervisor root: `/Volumes/data/datasets/hibiki_vi_v2/qa/vivos_qwen3_tts_mlx_retry_v6_full/supervisor_r1/`
- Config SHA-256: `a71ba8cae6249c144bf0412bef357b520fb2f33392ceef38bea432f9c94b6b5b`
- Launch SHA-256: `7bc2460ee325ecd5c0d0944308c55f14175a039c8377c8b698d5a9a85bd86a9c`
- State: `waiting_attempt0`, unchanged after more than one polling interval
- Supervisor stdout/stderr: `supervisor.log` (empty while waiting)
- State/events: `state.json`, `events.jsonl`
- Future command/timing record: `command_history.jsonl` plus `logs/`

At launch, process inspection found one model workload: the existing attempt-0 MLX generator PID 72367. There was no QA process, retry generator, or second model process. The supervisor remained at zero CPU and did not create command history while waiting.

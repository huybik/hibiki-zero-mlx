# Handover from the Japan box (2026-11-08 22:46 UTC+7)

Vast instance 59285612; its rental ends 2026-11-08 22:53 UTC+7. For the local session.

- All jobs stopped cleanly at 21:45 UTC+7 (`end_rental.sh`, exit 0). GigaSpeech 2: 213 of 240
  archives have en_audio on the Hub; 27 are left to voice (74–99, 185). pho and vivoice are complete.
- PR #83 is merged (c897ea7). `docs/agent-memory/tranche-box.md#state` has the next box's order:
  asr, filters and voice for the 27 archives (~4 h), then the cache build, round 4 SFT, evals and
  GRPO. Optimize each job heavily before its full run (`CONTEXT.md` invariant, user 2026-11-08).
- Not done: the cache-build profile (never started). What to measure is in
  `docs/agent-memory/tranche-cache.md#scale-up`; do it first on the next box.
- To do (user, 2026-11-08): an auto-rent watcher. Once 59285612 is gone, poll Vast for one
  H100 SXM (700 W) under $2.20/h in total and rent the first match. Requirements:
  - a driver that supports CUDA 13.0 or newer (the repo's `.venv` is torch 2.15+cu134; the
    Japan box had driver 595.91; voice's env is cu130, the ASR server's cu128);
  - disk of at least 350 GB (the Japan box used 286 of 350);
  - a fast download link (the bring-up pulls caches and models from the Hub).
  Then bring the box up as `tranche-box.md#bring-up` says. It needs the user's Vast API key,
  which the box never had.

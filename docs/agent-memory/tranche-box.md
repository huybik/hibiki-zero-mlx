# The tranche on a rented box

Read when bringing up a box for the tranche's jobs, or running them there: the scripts in
`training/tranche/box/`, the jobs' order, what a new box needs. Back to [CONTEXT](../../CONTEXT.md).
Sections: [State](#state), [Bring-up](#bring-up), [Jobs](#jobs), [Operating](#operating).

## State

**H100 SXM Japan (Vast instance 60447659, 2026-11-13 – 2026-11-16 08:00 UTC+7, $2.00/h)**: 700 W,
350 GB disk, a 28-CPU quota; the checkout on `main`, brought up from the Hub on 2026-11-13. It
holds the two-way run's caches (packs only), evaluation sets, start weight and its best and final
models (`runs/twoway_sft/`). Nothing else is on it: no tranche parts, envs or TTS/ASR models. It
replaced 60019118 on the same host (destroyed 2026-11-13 at the user's request). Between those two
boxes only the public ssh ports carried data, at 6–12 MB/s, so the new box fetched every cache's
shards from the Hub (~50–150 MB/s with `HF_HUB_DISABLE_XET=1`, ~25 min), then packed and dropped
them one directory at a time (`python -m training.finetune.streams --drop-shards DIR`; all at once
would not fit). Only the files not on the Hub went box to box: the Vietnamese tokenizer swap's
files, `fleurs-en`, `en-vi-val-en`, `giga-heldout`, run logs and `twoway_start_vi6k`.

**The H200 Indonesia box (59864576) went offline on 2026-11-10** at step ~15,000 of round 4d:
everything local was lost (round 4d's checkpoints, round 4b's test outputs `eval/cmp_step*`;
their numbers are in the [plan](../vi_en_parity_plan.md#order)). Round 4b's final pair is on the
Hub only (deleted from the H200 after a hash check against the Hub).

**VRAM budget (critical; the H200's 141 GB)**: nanovox TTS uses 109 GB VRAM at peak; real ASR (vLLM) 29 GB; together
= 138 GB → confirmed OOM kills (3 GB headroom insufficient). Voice cannot run with real ASR.
Without any ASR server voice.py fails (connection refused after 10 × 10 s retries). Result: giga/89–99
excl 98 (10 shards) are unvoiced — absent from Hub, not corrupted; recoverable on a future box with
proper ASR. After voice stopped: cache 25 GB + SFT 73 GB = 98 GB → fine. GRPO + cache = 91–128 GB → fine.

**Voice: DONE (stopped 2026-11-09 ~14:30 UTC)**. 230/240 giga archives voiced; giga/86 and 98 held
out; giga/89–99 excl 98 unvoiced (ASR/VRAM conflict). Never restart ASR while cache or SFT runs.

**Cache: DONE (2026-11-09 17:51 UTC)**. All 228 buildable GigaSpeech 2 archives cached (pho 224,
viVoice 354), on the Hub (fetched into `/workspace/tranche/repo/cache/` for SFT). The last 69 built in
1.24 h at 2,743× after the CUDA build was optimized 1,342× → 3,286× ([cache-build](cache-build.md#speed-on-cuda)).
Archives below 74 had no local inputs: one-shard read-ahead starved the GPU to ~1,450× until their
parts were prefetched 8 at a time (a builder prefetch is drafted, not merged).

**SFT round 4b: DONE (2026-11-09 18:25 – 2026-11-10 01:16 UTC, commit `1d26c15`)**:
`full_epoch4b`, 6,737 h of source (540,689 streams, 4.44M rows), 57,368 steps at 0.41 s,
`--batch-frames 8192 --eval-batch-size 128 --keep-checkpoints 1`, LR 3e-6, seed 46. Its final
model is its best (`full_epoch4b/` of the model repo, with its pair and logs); results in the
[plan](../vi_en_parity_plan.md#order), with its test sets against rounds 2 and 3
(`box/test_evals.sh`, copied to `/workspace/`, 11 min a model; outputs in each run's
`eval/cmp_step*/`, local only).

**SFT round 4 redo: `full_epoch4e` (`restart.sh sft`)**: round 4b's arguments with 500
`en_speech` shards for 150, `--batch-frames 4608 --eval-batch-size 64 --save-every 4000
--val-every 28000`: 5.09M rows, 8,456 h, 131,659 steps, ~10.3 h at the H100's 0.283 s a step.
The user (2026-11-10): redo round 4 if its test sets were not good; then, shown them, stop the
first redo (`full_epoch4c`, round 4b's data, deleted at step 1,760), add `en_speech` shards and
restart. `full_epoch4d` was this run on the H200, lost with it; its Hub prefix holds only run
metadata, so the redo took a new name. Its first Hub pair comes with the first validation, at
step 28,000. Stop it with `kill -TERM` on the launcher's pid alone (`/workspace/logs/sft.pid`)
and wait for its final sync.

**Next after round 4e (user, 2026-11-10):** its and round 4b's test sets, then two-way before
GRPO: the `en-vi` caches and one two-way run on all of round 4e's rows, the pilot's test at its
first validation ([two-way](two-way.md#two-way-run)); GRPO then runs once, on the two-way model.

**GRPO: DONE, `grpo_twoway` (`grpo_launch.sh weights/twoway_final.safetensors`, commit
`91e3c6f`)**, from the two-way SFT's final model: ~20 s an update with its window's generation,
~6.7 s for the update itself, and ~3.6 min of validation every 200. Resumed at update 800 on
2026-11-14 00:37 UTC+7 and stopped at 1,800 by its stop rule (06:51 UTC+7); the best, update
1,400, ships ([grpo](grpo.md#running) has its evaluations and test sets). It runs from the box's checkout, `/workspace/hibiki-zero` on `main` (the
`wt-*` worktrees are only for edits while a run holds the checkout); log `/workspace/logs/grpo.log`,
launcher pid in `grpo.pid`. `/workspace/auto_pipeline.sh`, `grpo_trigger.sh` and
`monitor_voice_done.sh` are leftovers of a chained attempt and not in the repo.

At the Japan box's end (2026-11-08 ~22:00 UTC+7; Vast instance 59285612): pho and vivoice are
complete (en_audio and caches on the Hub). GigaSpeech 2's 240 archives all have segments, text_en and
asr_vi (filters finished 2026-11-08 06:38 UTC+7); 213 have en_audio on the Hub, and `end_rental.sh`
(2026-11-08 21:46 UTC+7) listed 27 left to voice (74–99 and 185). Of those, 86 and 98 are now held
out for evaluation and never voiced ([giga-heldout](sft-evaluation.md#evaluation-sets)), so 25 are
left. The Hub's `giga/en_audio/` is the record: a voice run lists what is left. Next, on a new
box: asr, filters (it idles, logging "0 shards to hear"; the watchdog expects it up) and voice for
the rest (~8.6 min an archive alone on the H100; H200 speed unmeasured), then cache (~8 h for the
~128 archives left at GigaSpeech 2's current build speed; profile first with `--limit 2`
([tranche-cache](tranche-cache.md#scale-up))), sft and grpo as in [Jobs](#jobs), each optimized before
its full run (below).

The user's standing decisions for this work (the box's own agent memory does not travel):
- Minimize rental cost, not wall time: one GPU, its throughput optimized; no extra GPUs for speed
  (2026-11-07: "i want to minimize renting cost, not time").
- Quality may change for speed if it stays good (2026-11-07), measured on the quality test
  ([tranche-tts](tranche-tts.md)).
- The agent merges its own PRs once the review loop and the end-to-end check pass (2026-11-06).
- Go step by step: start a job by hand, check its result, then decide the next one; no chained
  jobs or automatic pipelines, and background waiters only notify (2026-11-09; 2026-11-13: "go
  step by step, check the result of each step to decide what to do next, dont build automatic
  pipeline").
- Optimize every job heavily before its full run, each in its own way (2026-11-08; voicing,
  optimized mid-run, cost more; [CONTEXT](../../CONTEXT.md#essential-invariants)).

## Bring-up

A rented H100 (Vast.ai) with `/workspace` as the working directory. The user rents a new box when
one expires; everything a job writes goes to the Hub as it runs, so a new box resumes from there
(user, 2026-11-07: "we can always rent a new box, just need to upload everything before it expire").

1. Clone the repo to `/workspace/hibiki-zero`; `training/h100.sh setup` (the `.venv`).
2. `/workspace/.env` (mode 600): `HF_TOKEN_AN`, a write token of anquachdev, for the tranche repos'
   pushes; huybik's token in `$HF_HOME/token` for the run recovery repo. Never print either.
3. `cp training/tranche/box/*.sh /workspace/`; `/workspace/envs-min.sh` builds the `nanovox`
   (voice.py), `asr` (the Qwen3-ASR server) and `cache` (the cache builds, free-threaded Python 3.14;
   installs Rust for safetensors) envs and downloads the models at their pinned revisions. Use `HF_HUB_DISABLE_XET=1` for large downloads (Xet hung or crawled at ~5 MB/s on the
   Japan box; plain HTTP ran ~140 MB/s).
4. The jobs read the tranche repos' parts locally under `/workspace/tranche/repo/<source>/<part>/`
   and fall back to the Hub (`ingest.part_file`), so voice and filters need no download first.

Round 4 SFT's inputs, all on the Hub (`restart.sh sft` reads them locally):
- the tranche caches: `anquachdev/mimi-en-vi-speech-20k` (dataset, `<source>/shard_*.pt`) into
  `/workspace/tranche/repo/cache/<source>/`;
- 500 shards of the noisy en_speech cache, listed in `box/en_speech_sub.txt` (of the 1,832 of
  `huybik/hibiki-zero-vi-full-sft` (dataset) `en-vi-grounded-v2-noisy/train/`: the first 150,
  ~720 h, round 4b's, by `random.Random(46).sample`; 350 more by the same seed from the sorted
  rest, 2026-11-10; 2,481 h, 20 GB plus 19 GB of packs; 266 s to fetch with
  `HF_HUB_DISABLE_XET=1`) into `data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train/`,
  the validation cache `en-vi-grounded-v2/validation/` into `data/cache/`, and the evaluation sets
  ([sft-evaluation](sft-evaluation.md)) under `data/eval/`;
- round 3's model, `huybik/hibiki-zero-vi-full-sft` (model) `tranche_epoch1b/checkpoints/model_step026176.safetensors`,
  into `runs/`, linked as `runs/tranche_epoch1b/model_step026176.safetensors` (the path `restart.sh` reads).

## Jobs

In order (user, 2026-11-07: all voicing first, then every cache, then SFT, then GRPO; 2026-11-10:
the two-way caches and run before GRPO):

| Job | Start | Done when |
|---|---|---|
| asr | `restart.sh asr` (the Qwen3-ASR server on :8100, beside filters and voice) | not needed after voice |
| filters | `restart.sh filters` (asr_vi for every ingested shard) | it logs "0 shards to hear" |
| voice | `restart.sh voice` (en_audio for every shard with text_en and asr_vi) | it logs "0 shards to voice" and none waits for asr_vi |
| cache | `restart.sh cache` ([tranche-cache](tranche-cache.md)) | every voiced shard has its cache on the Hub |
| sft | `restart.sh sft` (round 4, `full_epoch4b`; a rerun resumes, [sft-recovery](sft-recovery.md)) | its best checkpoint, then the test evals ([sft-evaluation](sft-evaluation.md)) |
| envi-cache | `restart.sh envi-cache`: every `en-vi` cache and the English evaluation sets ([two-way](two-way.md#two-way-run)) | "en_speech en-vi done", then the tranche builder's "done:" line, logged once its last push is in (a rerun with nothing left logs "0 shards to build", then pushes any leftovers before it exits) |
| two-way | `streams --drop-shards`, `init_weights` to `weights/twoway_start_vi6k.safetensors` and its baseline, then `restart.sh two-way` (`twoway_sft_4608`, [two-way](two-way.md#two-way-run); `twoway_sft` ran before at 8,192 frames) | its best checkpoint, then the test sets both ways |
| direction | `restart.sh direction` (`twoway_direction`: update 1,400 given the direction table, `weights/twoway_direction` written if missing; [two-way](two-way.md#direction)) | its 12,000 updates, then the obedience and before/after tests |
| grpo | `grpo_launch.sh <weight>` (from the checkout, [grpo](grpo.md#running)), on the two-way model | its updates; stop with TERM to the launcher alone |
| student | `restart.sh student` (`student_dep10_direction`, cut from the direction run's chosen checkpoint if missing; [phone-student](phone-student.md#next)) | its 190,126 updates (~12 h), then the test sets both ways; stop with TERM to the launcher alone |
| student-heads | `restart.sh student-heads` (`student_heads14_direction`) after `student`, once the GRPO run's last pair (on the Hub) is off the disk: its preflight needs 39 GiB | the same steps, then the test sets both ways |

Each job logs to `/workspace/logs/<job>.log` with its PID in `<job>.pid`; each runs as its own
process group. Stop one with `kill -- -$(cat /workspace/logs/<job>.pid)` or restart it with
`restart.sh <job>`, never `pkill -f` (voice's spawned workers look like vLLM's processes).

## Operating

- `watchdog.sh`, run in the background, checks every 10 min and exits printing the first problem:
  a job down, a new "voice run failed", voice or filters silent 30 min, the ASR server's health
  failing twice, or voice finding 0 shards. Rerun it after handling one; stop it before a pause.
- `pause_run.sh CMD...` runs a GPU benchmark at the next shard boundary: it stops voice and
  filters right after a shard is written (an open shard's voicing is lost when voice stops),
  runs the command from `/workspace/hibiki-zero` and restarts both. Nothing more fits beside
  production: voice and the ASR server hold ~70 of the 80 GB.
- Moving production voice to an unmerged branch for its end-to-end check: stop voice at a
  boundary, `git checkout --detach origin/<branch>` in `/workspace/hibiki-zero`, `restart.sh
  voice`; back to `main` after the merge. Never commit in `/workspace/hibiki-zero` while an SFT
  runs (a resume needs its launch commit): work in a worktree.
- Before the rental ends: `end_rental.sh HH:MM HH:MM` (UTC+7), detached with `setsid` (its header
  has the line), waits from the first time for voice's next shard boundary (at most until the
  second), stops voice, filters and the ASR server, pushes en_audio (voice itself pushes only every
  ~35 min) and prints what the Hub holds and what is left to voice; it exits 1 if a voiced shard is
  not on the Hub. Leave an hour: the boundary takes up to ~9 min (~13 when a run is starting), the
  stop up to 6, the push a few. The watchdog then reports the jobs down: stop it first.
- Threads and pools follow the container's CPU quota (`/sys/fs/cgroup/cpu.max`, 26.88 CPUs on the
  Japan box, `nproc` 224), as `restart.sh` and `training.paths.cpu_budget()` read it.
- Logs hold binary bytes: `grep -a`.

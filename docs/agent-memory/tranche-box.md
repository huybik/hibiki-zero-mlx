# The tranche on a rented box

Read when bringing up a box for the tranche's jobs, or running them there: the scripts in
`training/tranche/box/`, the jobs' order, what a new box needs. Back to [CONTEXT](../../CONTEXT.md).
Sections: [State](#state), [Bring-up](#bring-up), [Jobs](#jobs), [Operating](#operating).

## State

**H200 Indonesia (Vast instance 59864576, 2026-11-09 onwards)**: at commit `acce58a` (`cache/ctc-half`).
83 valid giga archives on Hub (3 half-precision shards deleted and shard_153 rebuilt float32
2026-11-09). Cache run ongoing (giga only; pho and vivoice complete).

**VRAM budget (critical)**: nanovox TTS uses 109 GB VRAM at peak; real ASR (vLLM) 29 GB; together
= 138 GB → confirmed OOM kills (3 GB headroom insufficient). Voice cannot run with real ASR.
Without any ASR server voice.py fails (connection refused after 10 × 10 s retries). Result: giga/89–99
excl 98 (10 shards) are unvoiced — absent from Hub, not corrupted; recoverable on a future box with
proper ASR. After voice stopped: cache 25 GB + SFT 73 GB = 98 GB → fine. GRPO + cache = 91–128 GB → fine.

**Voice: DONE (stopped 2026-11-09 ~14:30 UTC)**. 230/240 giga archives voiced; giga/86 and 98 held
out; giga/89–99 excl 98 unvoiced (ASR/VRAM conflict). Never restart ASR while cache or SFT runs.

**GRPO profile: DONE (2026-11-09)**. Results:

| gen-batch | rollouts/s | peak GiB |
|---|---|---|
| 64 | 1.75 | 49.2 |
| 128 | 3.33 | 57.5 |
| **192** | **6.66** | **65.7** |

| micro-batch | update frames/s | peak GiB |
|---|---|---|
| 2 | 7,132 | 74.1 |
| **3** | **8,827** | **102.9** |
| 4 | OOM | — |

Optimal: gen-batch 192, micro-batch 3. With cache: 102.9 + 25 = ~128 GiB (13 GiB headroom). Baseline BLEU 25.99.
GRPO worktree at /workspace/wt-grpo (branch grpo/trainer, .venv at /workspace/hibiki-zero/.venv symlinked,
weights symlinked). After SFT checkpoint: `grpo_launch.sh <weight>` from wt-grpo with gen-batch 192 micro-batch 3.

**SFT round 4: RUNNING (2026-11-09, step ~170+)**. cache + SFT = ~99 GiB. SFT data: pho (224 ✓),
vivoice (354 ✓), en_speech_sub (150 ✓), all eval sets ✓, checkpoint model_step026176.safetensors ✓.

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
   (voice.py) and `asr` (the Qwen3-ASR server) envs and downloads the models at their pinned
   revisions. Use `HF_HUB_DISABLE_XET=1` for large downloads (Xet hung or crawled at ~5 MB/s on the
   Japan box; plain HTTP ran ~140 MB/s).
4. The jobs read the tranche repos' parts locally under `/workspace/tranche/repo/<source>/<part>/`
   and fall back to the Hub (`ingest.part_file`), so voice and filters need no download first.

Round 4 SFT's inputs, all on the Hub (`restart.sh sft` reads them locally):
- the tranche caches: `anquachdev/mimi-en-vi-speech-20k` (dataset, `<source>/shard_*.pt`) into
  `/workspace/tranche/repo/cache/<source>/`;
- 150 shards of the noisy en_speech cache, listed in `box/en_speech_sub.txt` (drawn by
  `random.Random(46).sample` from the 1,832 of `huybik/hibiki-zero-vi-full-sft` (dataset)
  `en-vi-grounded-v2-noisy/train/`; ~720 h) into `data/cache/en_speech_sub/en-vi-grounded-v2-noisy/train/`,
  the validation cache `en-vi-grounded-v2/validation/` into `data/cache/`, and the evaluation sets
  ([sft-evaluation](sft-evaluation.md)) under `data/eval/`;
- round 3's model, `huybik/hibiki-zero-vi-full-sft` (model) `tranche_epoch1b/checkpoints/model_step026176.safetensors`,
  into `runs/tranche_epoch1b/`.

## Jobs

In order (user, 2026-11-07: all voicing first, then every cache, then SFT, then GRPO):

| Job | Start | Done when |
|---|---|---|
| asr | `restart.sh asr` (the Qwen3-ASR server on :8100, beside filters and voice) | not needed after voice |
| filters | `restart.sh filters` (asr_vi for every ingested shard) | it logs "0 shards to hear" |
| voice | `restart.sh voice` (en_audio for every shard with text_en and asr_vi) | it logs "0 shards to voice" and none waits for asr_vi |
| cache | `restart.sh cache` ([tranche-cache](tranche-cache.md)) | every voiced shard has its cache on the Hub |
| sft | `restart.sh sft` (round 4, `full_epoch4`; a rerun resumes, [sft-recovery](sft-recovery.md)) | its best checkpoint, then the test evals ([sft-evaluation](sft-evaluation.md)) |
| grpo | `grpo_launch.sh <weight>` (PR #59, worktree `/workspace/wt-grpo`, [grpo_plan](../grpo_plan.md)) | its updates; stop with TERM to the launcher alone |

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

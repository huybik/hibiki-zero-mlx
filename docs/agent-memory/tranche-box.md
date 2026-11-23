# The tranche on a rented box

Read when bringing up a box for the tranche's jobs, or running them there: the scripts in
`training/tranche/box/`, the jobs' order, what a new box needs. Back to [CONTEXT](../../CONTEXT.md).
Sections: [State](#state), [Bring-up](#bring-up), [Jobs](#jobs), [Operating](#operating).

## State

**No box now** (2026-11-22 21:55 UTC+7). Next on a box: the heads student's on-policy run on the
new EN→VI data, `both_heads_z2` ([on-policy-distillation](on-policy-distillation.md#the-run-on-the-next-box),
which says what fits in 32 GB), then the palettization finish and the 3B's experiments, which need
a bigger card. `h100.sh` takes an RTX 5090 since 2026-11-23 ([sft-recovery](sft-recovery.md#launcher));
that run needs 60 GiB of RAM in the container (the last 5090 had 31).

**The box 2026-11-22: an RTX 5090** (Vast 61906288, South Korea, $0.558/h with 350 GB of disk; 32 GB,
capped at 500 W by the host, driver 580, a 15.4-CPU quota, 31 GB of RAM, 15 GB of `/dev/shm`;
destroyed 2026-11-22 21:55 UTC+7 at the user's request, after its run). It built the rest of the
EN→VI data, alone after the Mac stopped (user: "Cache build first"; [envi-data](envi-data.md#the-box-run)),
and ran nothing else: `both_heads_z` waits for the next box. Its records and scripts are in
`/Volumes/data/hibiki-zero-evals/box-61906288/`.

**The box 2026-11-19 to 2026-11-22: an RTX PRO 6000 Blackwell Workstation Edition** (Vast 61345966,
destroyed 2026-11-22 04:25 UTC at the user's request; Beijing, $1.217/h; 96 GB, 600 W, driver
580.178.04, a 61-CPU quota, 171 GiB of RAM, 300 GB of disk). Its network sat behind a slow proxy
([Bring-up](#a-box-behind-a-slow-proxy)).
- It ran `student_heads14_cb16` (`restart.sh student-heads`, 16 codebooks, from `ce80c0b`,
  started 2026-11-20 05:19 UTC): 0.694 s an update, ~37 h for 190,126 updates plus 12 validations
  ([student-heads](student-heads.md#runs)). `student_heads14` (8 codebooks) stopped for it at
  ~22,600; its 16,000 pair and best are on the Hub.
- The user (2026-11-20): run it to the end and score its test sets both ways, without asking.
  Then upload everything and destroy the box. The user (2026-11-21): "should run grpo after
  distil". So the distilled best goes through GRPO first (`grpo_launch.sh`, as `grpo_student`
  ran: [student-grpo](student-grpo.md#run)), and the test sets score GRPO's best. GRPO loads,
  samples and scores through the same `common.build_lm`, `get_lmgen` and forward as the heads
  student's training and validation. The rooms' noise bank was fetched ahead of time.
- One session runs the box's jobs from 2026-11-21 (user: "you in control of the training on the box
  now"). After the heads run, in order, one GPU job at a time:
  1. its test sets on its best (`test_heads.sh`);
  2. the box validation of PRs #149–#151 (user: "after the heads run and before grpo"): short runs
     with no Hub sync, under `runs/val_*`, logs in `/workspace/logs/val/`;
  3. the student's on-policy stage, aimed at the best phone model (user, 2026-11-21: "i'm not
     asking for plain grpo, i'm asking for best outcome"): 200-update probes from the distilled
     best (distillation alone at three rates; both terms, z-scored and centered; plain GRPO as the
     control only), one run of the winner, then the 6-bit palettization finish; then the 3B's
     experiments (scheduled sampling, anchored GRPO). The 3B's start, the two-way SFT's final
     model, is `weights/twoway_final.safetensors` (fetched 2026-11-21, sha256 checked).
  It stopped before the run (user, 2026-11-22: "let's finish everything and stop before the long
  run"; "can destroy this pro 6000 box when everything done"), once 1, 2 and the probes of 3 were
  done ([probe results](on-policy-distillation.md#probe-results)). Before it was destroyed:
  - every run's pairs, bests and logs were on the Hub, with the probes' last uploads made without
    Xet ([Bring-up](#a-box-behind-a-slow-proxy)); `audit_hub.py` (box-only) matched every weight
    and run file over 10 MB by sha256 against the Hub's LFS objects;
  - what the Hub lacked went up: the heads students' start weights and configs, the tokenizers,
    the swap's init map and the base 3B's `config.json` under the model repo's `weights/` (the
    box's paths), `student_heads14`'s 20,000 pair beside its 16,000 one, and the evaluation sets
    built here (`fleurs-en`, `en-vi-val-en`, `en-vi-val-en-noisy`) under the dataset repo's `eval/`;
  - left off: the upstream Hibiki weights, `giga-heldout` (built where it runs), the caches' packs
    (built from the Hub's shards);
  - the run records (no weights, optimizer states or WAVs), `/workspace/logs`, `bench/` and the
    box-only scripts were copied to `/Volumes/data/hibiki-zero-evals/box-61345966/`.
- **All of `en_speech`, both ways** (user, 2026-11-21: "let's build both vi - en and en - vi"). Round
  4e's test (500 shards against 150: FLEURS −1.3) also changed the batch and the updates, and
  spoke only for VI→EN, whose `en_speech` source is TTS; for EN→VI `en_speech` holds the only real
  English speech (2,481 of 8,456 h). The `vi-en` caches of all 1,832 train shards, noisy and
  clean, are on the Hub (78 GB each): a run fetches them, so `vi-en` needs no data work; using all
  of them is the next run's shard list (round 4e listed 500). `en-vi` had round 4e's 500:
  `/workspace/envi_rest.py` (box-only) builds the other 1,332 beside the heads run (started
  2026-11-21 08:00 UTC; log `logs/envi_rest.log`). Sources by aria2c 25 at a time (321 GB,
  ~16 MB/s), `cache_codes --direction en-vi` in `/workspace/envs/cache` (the `.venv` copied, with
  free-threaded av 19.0.2) capped at 30% of the card, into `/workspace/en_vi_rest_out` (outside
  the training caches, which runs started meanwhile read); each batch goes to the Hub's
  `en-vi/en-vi-grounded-v2/train/`, is checked against its listing and deleted here. While
  `logs/envi_rest.pause` exists no new build starts (for profiles). Profiled on 2 shards: 21.3 h
  of audio built in 122 s, 98.7% of rows kept; the heads run went 0.69 → ~1.4 s an update while it
  built. The first batch of 25 (512 h of Mimi audio, both clips) took 1,903 s at ~970× real time on
  the shared GPU, 719 s of download (~8 MB/s) and 180 s of upload: the whole build would hold half
  the GPU ~14 h and the noisy copies ~3 h more, moving the heads run's end ~10 h. Stopped
  2026-11-21 ~08:55 UTC: the user does the data preparation in a separate session ("continue with
  training, i will do the data preparation on a separate session"). On the Hub then: 527 of the
  1,832 `en-vi` shards, no noisy `en-vi` copy. Left on the box for it: `/workspace/envi_rest.py`
  (resumes from the Hub's listings: builds the missing clean shards, makes every shard's noisy
  copy with `mimi_cache_cuda` (PR #153, merged as `7090cc6`; validated on this box both ways,
  [noisy-source](noisy-source.md#checks)) from `/workspace/wt-noisy` (main at `7090cc6`,
  detached) at `--seed 1` into
  `en-vi/en-vi-grounded-v2-noisy/`, uploads a batch while the next builds; holds its GPU steps while
  `logs/envi_rest.pause` exists), `/workspace/envs/cache`. The Mac and then the RTX 5090 box built them
  instead ([envi-data](envi-data.md)). `en-vi-val-en-noisy` is
  built in `/workspace/hibiki-zero/data/eval/` (2026-11-21).
- The previous box, an H100 SXM in Japan (60447659), was destroyed on 2026-11-15. Its results are
  on the Mac in `/Volumes/data/hibiki-zero-evals/box-60447659/`, without weights
  ([sft-evaluation](sft-evaluation.md#recording-results)). Its weights are on the Hub.
- A box brings everything up from the Hub: ~50–150 MB/s with `HF_HUB_DISABLE_XET=1` on a direct
  link. Box-to-box copies over the public ssh ports ran only 6–12 MB/s.
- The swap's init map (`tokenizer_spm_48k_multi6_2_vi6k_init.json`) is on the Hub since 2026-11-22
  (model repo, `weights/`). The
  trainers need it beside the swap to read any base-tokenizer cache (`paths.swap_slots`).
  - This box rebuilt it with `training.vi_tokenizer.build run`. The build is deterministic, and
    its tokenizer model came out byte-identical to `release/twoway_grpo1400/`'s.
  - A copy is also kept on the Mac with the last box's results, under `weights-small/`.

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
1.24 h at 2,743× after the CUDA build was optimized 1,342× → 3,286× ([cache-speed-cuda](cache-speed-cuda.md)).
Archives below 74 had no local inputs: one-shard read-ahead starved the GPU to ~1,450× until their
parts were prefetched 8 at a time (a builder prefetch is drafted, not merged).

**SFT round 4b: DONE (2026-11-09 18:25 – 2026-11-10 01:16 UTC, commit `1d26c15`)**:
`full_epoch4b`, 6,737 h of source (540,689 streams, 4.44M rows), 57,368 steps at 0.41 s,
`--batch-frames 8192 --eval-batch-size 128 --keep-checkpoints 1`, LR 3e-6, seed 46. Its final
model is its best (`full_epoch4b/` of the model repo, with its pair and logs); results in the
[plan](../vi_en_parity_plan.md#order), with its test sets against rounds 2 and 3
(`box/test_evals.sh`, copied to `/workspace/`, 11 min a model; outputs in each run's
`eval/cmp_step*/`, local only).

**SFT round 4 redo: `full_epoch4e`**: round 4b's arguments with 500
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

**GRPO: DONE, `grpo_twoway` (`grpo_launch.sh` from `weights/twoway_final.safetensors`, commit
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

### A box behind a slow proxy

The Beijing host (61345966) routes all international traffic through a transparent proxy: its DNS
answers in 198.18.0.0/15.
- Throughput: ~1.5–2.5 MB/s a connection and 10–43 MB/s in all, and single connections drop.
  Domestic mirrors give 10–15 MB/s a connection.
- The env: `h100.sh setup` stalled 70 min on four CUDA wheels. The image's
  `UV_HTTP_TIMEOUT=500` waits 8 min on each stalled connection. The same pins from the mirrors
  installed in 4 min:
  - the torch wheel from `mirrors.aliyun.com/pytorch-wheels/cu134/`, its sha256 checked against
    PyTorch's index;
  - everything else from Tsinghua's PyPI (`pypi.tuna.tsinghua.edu.cn/simple`);
  - the Rust builds' crates from Tsinghua's crates.io index, set as a source replacement in
    `~/.cargo/config.toml`.
- The Hub: 8,007 files (173 GB) took ~3.7 h with `aria2c`, 8 connections a file and 8 files at
  a time. The CDN links are signed for ~1 h, so they were resolved in ~6 GB batches.
  `hf download` got ~2 MB/s a file and stalled.
- Disk: shards and packs of every training cache (~150 GB each) don't fit together on 300 GB.
  Pack each cache with `streams --drop-shards` as soon as its download completes. A new
  `PACK_VERSION` (7 since 2026-11-22) refuses dropped packs of the old one: delete a folder's old
  `.packed/<shard>/` folders, then fetch and drop its shards again, a folder at a time, as the disk
  holds a folder's shards beside the packs (a dropped pack left in place counts as fetched, below).
  - A fetch that checks only the files would fetch the dropped shards again, and a trainer
    starting meanwhile packs half-written ones. A shard whose `pack.json` is marked `dropped`
    counts as fetched.
- `fleurs-en`, `en-vi-val-en` and `en-vi-val-en-noisy` are in the dataset repo's `eval/` since
  2026-11-22 (built on the box with `training.prep.eval_sets`).
- Xet uploads stalled twice at the end of a probe's final sync (2026-11-22): ~250 KB/s at 99% of
  the model. Plain LFS ran 20–35 MB/s. The stalled sync was stopped by its PID (its files stay
  staged), and the run's one-shot final sync rerun with `HF_HUB_DISABLE_XET=1`
  (`python -m training.finetune.hf_sync runs/RUN huybik/hibiki-zero-vi-full-sft`, with
  `HF_HOME` and `HIBIKI_HF_PREFIX=RUN`). On such a box, export `HF_HUB_DISABLE_XET=1` before
  `grpo_launch.sh`, so the run's own sync skips Xet.
- The image (`vastai/vllm-omni`) starts vLLM-Omni serving Qwen-Image, a model UI and Ray.
  - Stop them with `supervisorctl stop vllm-omni model-ui ray`, and drop their entries from
    `/etc/portal.yaml` so they skip on a restart.
  - vLLM's DiffusionWorker survives the stop as an orphan. It held 55 GB of the GPU and pulled
    54 GB of Qwen-Image into `$HF_HOME/hub/blobs`. Kill it by the PID that `nvidia-smi` lists, and
    delete the download.
  - The image exports `UV_OVERRIDE` (an NCCL pin) and `UV_INDEX_STRATEGY=unsafe-best-match`.
    Unset both for `setup`, or the env differs from the pinned one.
- Don't `pkill -f` a pattern that also appears in the ssh command running it: it kills its own
  shell.

Round 4 SFT's inputs, all on the Hub (read locally):
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
| envi-cache | `restart.sh envi-cache`: every `en-vi` cache and the English evaluation sets ([two-way](two-way.md#two-way-run)) | "en_speech en-vi done", then the tranche builder's "done:" line, logged once its last push is in (a rerun with nothing left logs "0 shards to build", then pushes any leftovers before it exits) |
| two-way | `streams --drop-shards`, `init_weights` to `weights/twoway_start_vi6k.safetensors` and its baseline, then `restart.sh two-way` (`twoway_sft_4608`, [two-way](two-way.md#two-way-run); `twoway_sft` ran before at 8,192 frames) | its best checkpoint, then the test sets both ways |
| grpo | `grpo_launch.sh RUN <weight> [ARGS]` (from the checkout, [grpo](grpo.md#running)), on the two-way model or a phone student | its updates; stop with TERM to the launcher alone |
| student | `restart.sh student` (`student_dep10`, cut from GRPO update 1,400 if missing; [phone-student](phone-student.md#next)) | its 190,126 updates (~12 h), then the test sets both ways; stop with TERM to the launcher alone |
| student-distill | `restart.sh student-distill` (`student_dep10_anneal`, from `student_dep10_distill`'s model 32,000, the teacher GRPO update 1,400; [phone-student](phone-student.md#distillation)) | its `--max-steps`, then the test sets both ways; stop with TERM to the launcher alone |
| student-heads | `restart.sh student-heads` (`student_heads14_cb16`, 16 codebooks, distilled from GRPO update 1,400 from the start; [student-heads](student-heads.md#runs)): running on the RTX PRO 6000 ([State](#state)), ~37 h there (0.694 s an update, 50.9 GiB peak) | the same steps, then the test sets both ways |

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

# The tranche on a rented box

Read when bringing up a box for the tranche's jobs, or running them there: the scripts in
`training/tranche/box/`, the jobs' order, what a new box needs. Back to [CONTEXT](../../CONTEXT.md).
Sections: [State](#state), [Bring-up](#bring-up), [Jobs](#jobs), [Operating](#operating).

## State

**Box 62513802** (2026-11-25, rented by the user's 5090 watcher at 11:47 UTC): an RTX 5090, Russia,
$0.511/h, `ssh -p 17399 root@109.195.243.87`; 32 GB at 575 W, driver 595.91.07, an i5-14600K with
a 19.2-CPU quota, 123 GiB of RAM, 350 GB of disk. It ran the heads student's 6-bit palettization
finish ([student-palettize](student-palettize.md#the-6-bit-run)), then the level-stage test
([runs](exposure-bias-runs.md#the-level-stage-test)) and PR #187's check.
- The Hub gives ~45–58 MB/s down but takes uploads at only ~1–2 MB/s: a final pair (15.4 GB)
  takes 2–4 h, so wait for the run's sync before destroying the box.
- To the Mac over ssh: ~1.8 MB/s a stream, ~7.4 MB/s over 8 (`split`, parallel `scp`; 12 dropped
  a connection, so re-fetch the missing parts and check the sha256).

The one before, 62366622 (2026-11-24 to 25; an RTX 5090, Switzerland, $0.671/h, 500 W of 600), ran
`both_heads_z` 200 → 1,800 and `both_heads_z2` 400 → 1,200 ([runs](heads-on-policy-runs.md#both_heads_z-continued));
destroyed 2026-11-25 09:53 UTC at the user's request once both runs' Hub files matched by sha256.
- Uploads from the Mac run at ~20 KB/s: the box clones the repo from GitHub. The Hub gave ~80 MB/s.
- The default openslr mirror serves the rooms' bank at ~250 KB/s a connection; `aria2c -x16` over
  it and `openslr.trmal.net` took ~2 min (`fetch_bank` checks the SHA-256 either way).

Past boxes' records and scripts (no weights, but 62170174's) are in
`/Volumes/data/hibiki-zero-evals/box-<id>/`: 62366622 (above; `run_both_heads_z.tgz` and
`run_both_heads_z2.tgz`, the tests and logs of both runs),
62188551 (an RTX 5090, 32 GB, UAE, $0.631/h; it ran `both_heads_z2`,
[heads-on-policy-runs](heads-on-policy-runs.md#both_heads_z2), destroyed 2026-11-24),
61906288 (an RTX 5090, South Korea; it built the rest of the EN→VI data, [envi-data](envi-data.md#the-box-run)),
62170174 (an H100 SXM, Japan; the 3B's experiments 1 and 2,
[exposure-bias-runs](exposure-bias-runs.md), destroyed 2026-11-23), 61345966 (the RTX PRO 6000 below) and 60447659 (an H100 SXM, Japan).

**The box 2026-11-19 to 2026-11-22: an RTX PRO 6000 Blackwell Workstation Edition** (Vast 61345966,
destroyed 2026-11-22 04:25 UTC at the user's request; Beijing, $1.217/h; 96 GB, 600 W, driver
580.178.04, a 61-CPU quota, 171 GiB of RAM, 300 GB of disk). Its network sat behind a slow proxy
([Bring-up](#a-box-behind-a-slow-proxy)).
- It ran `student_heads14_cb16` (16 codebooks, from `ce80c0b`, started 2026-11-20 05:19 UTC): 0.694 s an update, ~37 h for 190,126 updates plus 12 validations
  ([student-heads](student-heads.md#runs)). `student_heads14` (8 codebooks) stopped for it at
  ~22,600; its 16,000 pair and best are on the Hub.
- The user (2026-11-20): run it to the end and score its test sets both ways, without asking.
  Then upload everything and destroy the box. The user (2026-11-21): "should run grpo after
  distil". So the distilled best goes through GRPO first (`grpo_launch.sh`, as `grpo_student`
  ran: [student-grpo](student-grpo.md#run)), and the test sets score GRPO's best. GRPO loads,
  samples and scores through the same `model.build_lm`, `get_lmgen` and forward as the heads
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
  done ([probe results](heads-on-policy-runs.md#probe-results)). Before it was destroyed:
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
  of them is the next run's shard list (round 4e listed 500). `en-vi` had round 4e's 500; this
  box built 27 more beside the heads run (`cache.en_speech --direction en-vi`, capped at 30% of the
  card) before the user took the data preparation to a separate session (2026-11-21: "continue with
  training, i will do the data preparation on a separate session"). Profiled on 2 shards: 21.3 h
  of audio built in 122 s, 98.7% of rows kept; the heads run went 0.69 → ~1.4 s an update while it
  built. A batch of 25 (512 h of Mimi audio, both clips) took 1,903 s at ~970× real time on the
  shared GPU, 719 s of download (~8 MB/s) and 180 s of upload. The Mac and then the RTX 5090 box
  built the rest and every noisy copy (`training/cache/noisy.py`, [envi-data](envi-data.md)).
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

**Voice: DONE (stopped 2026-11-09 ~14:30 UTC)**. 230/240 giga archives voiced; giga/86 and 98 held
out; giga/89–99 but 98 unvoiced (absent from the Hub): the H200's TTS then and the ASR server did not
fit its GPU together.

**Cache: DONE (2026-11-09 17:51 UTC)**. All 228 buildable GigaSpeech 2 archives cached (pho 224,
viVoice 354), on the Hub (fetched into `/workspace/tranche/repo/cache/` for SFT). The last 69 built in
1.24 h at 2,743× after the CUDA build was optimized 1,342× → 3,286× ([cache-speed-cuda](cache-speed-cuda.md)).
Archives below 74 had no local inputs: one-shard read-ahead starved the GPU to ~1,450× until their
parts were prefetched 8 at a time (a builder prefetch is drafted, not merged).

**SFT round 4b: DONE (2026-11-09 18:25 – 2026-11-10 01:16 UTC, commit `1d26c15`)**:
`full_epoch4b`, 6,737 h of source (540,689 streams, 4.44M rows), 57,368 steps at 0.41 s,
`--batch-frames 8192 --eval-batch-size 128 --keep-checkpoints 1`, LR 3e-6, seed 46. Its final
model is its best (`full_epoch4b/` of the model repo, with its pair and logs); results in the
[plan](../vi_en_parity_plan.md#order), with its test sets against rounds 2 and 3 (eight sets in
one `eval.py` call, 11 min a model; outputs in each run's `eval/cmp_step*/`, local only).

**SFT round 4 redo: `full_epoch4e`**: round 4b's arguments with 500
`en_speech` shards for 150, `--batch-frames 4608 --eval-batch-size 64 --save-every 4000
--val-every 28000`: 5.09M rows, 8,456 h, 131,659 steps, ~10.3 h at the H100's 0.283 s a step.
The user (2026-11-10): redo round 4 if its test sets were not good; then, shown them, stop the
first redo (`full_epoch4c`, round 4b's data, deleted at step 1,760), add `en_speech` shards and
restart. `full_epoch4d` was this run on the H200, lost with it; its Hub prefix holds only run
metadata, so the redo took a new name.

**GRPO: DONE, `grpo_twoway` (`grpo_launch.sh` from `weights/twoway_final.safetensors`, commit
`91e3c6f`)**, from the two-way SFT's final model: ~20 s an update with its window's generation,
~6.7 s for the update itself, and ~3.6 min of validation every 200. Resumed at update 800 on
2026-11-14 00:37 UTC+7 and stopped at 1,800 by its stop rule (06:51 UTC+7); the best, update
1,400, ships ([grpo](grpo.md#running) has its evaluations and test sets). It runs from the box's checkout, `/workspace/hibiki-zero` on `main` (the
`wt-*` worktrees are only for edits while a run holds the checkout); log `/workspace/logs/grpo.log`,
launcher pid in `grpo.pid`.

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
3. `cp training/tranche/box/*.sh /workspace/`; `/workspace/envs.sh [env ...]` builds, side by side, the `ingest`
   (YODAS's ingest), `vieneu` (voice.py and its engine, `vieneu/`), `asr` (the Qwen3-ASR server), `cache` (the cache builds, free-threaded
   3.14; installs Rust for safetensors) and `mt` (translate.py) envs, all without arguments, and downloads their models
   at their pinned revisions. Use `HF_HUB_DISABLE_XET=1` for large downloads (Xet crawled at ~5 MB/s; HTTP ~140).
4. The jobs read the tranche repos' parts locally under `/workspace/tranche/repo/<source>/<part>/`
   and fall back to the Hub (`repo.part_file`), so voice and the cache build need no download first.

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
  Pack each cache with `packs --drop-shards` as soon as its download completes. A new
  `PACK_VERSION` (7 since 2026-11-22) refuses dropped packs of the old one: delete a folder's old
  `.packed/<shard>/` folders, then fetch and drop its shards again, a folder at a time, as the disk
  holds a folder's shards beside the packs (a dropped pack left in place counts as fetched, below).
  - A fetch that checks only the files would fetch the dropped shards again, and a trainer
    starting meanwhile packs half-written ones. A shard whose `pack.json` is marked `dropped`
    counts as fetched.
- `fleurs-en`, `en-vi-val-en` and `en-vi-val-en-noisy` are in the dataset repo's `eval/` since
  2026-11-22 (built on the box with `training.finetune.eval_sets`).
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
- Record a detached job's PID from inside it: `setsid -f bash -c 'echo $$ > PIDFILE; exec CMD'`.
  `nohup setsid CMD & echo $! > PIDFILE` saved another PID, a kill by it missed the job, and two
  fetchers packed one folder at once (`packs --drop-shards`): one failed on the other's dropped
  shards (`FileNotFoundError`), so the folder was fetched again.

### Fetching from the Hub

The H100 box (62170174, 2026-11-23; its script `/workspace/fetch_z2.py`):
- One connection gets ~17–27 MB/s (a 12.5 GB weight ~10 min); 16 files at a time
  (`snapshot_download(max_workers=16)`, `HF_HUB_DISABLE_XET=1`) 110–150 MB/s, the nine caches'
  172 GB in ~20 min.
- The Hub limits requests per account: `eval/*` hit HTTP 429 at ~3,800 of its 7,150 files, most of
  the rest `multimed-vi`'s 3,437 clips (a test set only), and a download beside it sat in 429
  retries for 12 min. GRPO's validation needs six sets: `fleurs-vi`, `fleurs-en`, `en-vi-val`,
  `en-vi-val-noisy`, `en-vi-val-en`, `en-vi-val-en-noisy`; fetch those by name.

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
  into `runs/`, linked as `runs/tranche_epoch1b/model_step026176.safetensors`.

## Jobs

The tranche's order (user, 2026-11-07): all voicing first, then every cache, then SFT, then GRPO.
`restart.sh <job>` starts each of its jobs. The `student-slices-*` jobs share their env (`HIBIKI_HF_REPO`, the CPU
quota's threads), training and validation caches and validation sets, set once before the jobs' cases:

| Job | What | Done when |
|---|---|---|
| asr | the Qwen3-ASR server on :8100, in FP8, beside voice and YODAS's ingest | not needed after voice |
| voice | `voice.py run yodas`: en_audio for every YODAS shard with text_en and asr_vi, rerun as shards arrive ([tranche-tts](tranche-tts.md#running)) | it logs "0 shards to voice" |
| voice-en-speech | `voice.py run en_speech`: the EN-VI dataset's Vietnamese heard, its broken rows voiced again ([tranche-tts](tranche-tts.md#vietnamese)) | it exits 0 after its last shard |
| cache | every source's caches both ways in one pass ([tranche-cache](tranche-cache.md)) | every voiced shard has its caches on the Hub |
| student-slices-5090, student-slices-decay-5090 | the slices student's Stage 1 and its decay ([student-slices](student-slices.md)) | its `--max-steps`, then `test_sets.sh` |
| student-slices-refit-5090, student-slices-pal6-5090 | the slices' refit on the on-policy trunk and its 6-bit finish ([slices-on-policy](slices-on-policy.md)) | the same |
| student-slices-wide-5090, student-slices-wide-warm-5090, and their `-decay-5090` | the wide slices, from random or the teacher's depformer, then their decays ([student-slices](student-slices.md#wide-slices)) | the same |

GRPO starts with `grpo_launch.sh RUN <weight> [ARGS]` instead (from the checkout, [grpo](grpo.md#running)); stop
it with TERM to the launcher alone. `test_sets.sh CONFIG WEIGHT OUT_DIR` scores a model on the 13
test sets.

Each job logs to `/workspace/logs/<job>.log` with its PID in `<job>.pid`; each runs as its own
process group. Stop one with `kill -- -$(cat /workspace/logs/<job>.pid)` or restart it with
`restart.sh <job>`, never `pkill -f` (voice's spawned workers look like vLLM's processes).
Under MPS (`nvidia-cuda-mps-control -d`), a GPU process that dies abnormally (killed out of
memory) faults the MPS server, which then refuses every new client ("Server teardown in
progress", error 807) until all its clients exit: restart asr, then voice (2026-11-26). Its
spawned workers outlive a killed process, in the job's group: `restart.sh` stops the whole group
even when its leader is gone (36 of voice's, ~16 GB, were left so).

## Operating

- Moving production voice to an unmerged branch for its end-to-end check: stop voice at a
  boundary, `git checkout --detach origin/<branch>` in `/workspace/hibiki-zero`, `restart.sh
  voice`; back to `main` after the merge. Never commit in `/workspace/hibiki-zero` while an SFT
  runs (a resume needs its launch commit): work in a worktree.
- Before the rental ends, stop voice at a shard boundary (an open shard's voicing is lost) and check
  that the Hub holds every voiced shard: voice pushes only every 30 min, and a rerun pushes first what
  an earlier run wrote but did not push.
- Threads and pools follow the container's CPU quota (`/sys/fs/cgroup/cpu.max`, 26.88 CPUs on the
  Japan box, `nproc` 224), as `restart.sh` and `training.paths.cpu_budget()` read it.
- Logs hold binary bytes: `grep -a`.

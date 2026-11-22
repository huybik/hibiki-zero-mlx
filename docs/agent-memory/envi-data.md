# EN→VI data for the next two-way run

Read when preparing or checking the data of the next two-way run: `en_speech` in both directions,
the noisy copies of every English source (`en_speech`'s and the tranche's voice-cloned English), the
Mac run that builds them (`envi_mac.py`), the box run that meets it from the other end
(`envi_box.py`) and where each folder is on the Hub. The build itself is in
[cache-build](cache-build.md), the noisy copies in [noisy-source](noisy-source.md#noisy-cache), the
two-way run in [two-way](two-way.md).
Back to [CONTEXT.md](../../CONTEXT.md) · [Decision](#decision) · [On the Hub](#on-the-hub) ·
[The Mac run](#the-mac-run) · [The box run](#the-box-run) · [Speed](#speed)

## Decision

- The next two-way run trains on all of `en_speech` in both directions, with noisy source copies
  for both (user, 2026-11-21: "let's build both vi - en and en - vi", "also how about noisy
  input?"), and on noisy copies of the tranche's `en-vi` rows, whose English source is VoxCPM2 TTS
  ("also cover EN→VI rows (~6,000 h) noisy rows").
- Why EN→VI needed it: in the first two-way run ~80% of EN→VI rows heard synthetic English (the
  tranche's `en-vi` rows) and only round 4e's 500 `en_speech` shards were real recordings, while
  ~7,600 h of real English sat unused; and no English source had ever been noisy. For VI→EN the
  same two steps were worth +9 (real speech) and +4.8 (noisy channel) chrF on FLEURS test.
- The work runs on the Mac and its external drive (user, 2026-11-21: "can use the mac and external
  drive"), not on the training box, where a build beside a run halves the run's speed
  ([tranche-box](tranche-box.md#state)).
- The `en_speech` shards built from 2026-11-22 on time their Vietnamese text with
  wav2vec2-base-vietnamese-250h (non-commercial) on the Neural Engine, the first 528 and the tranche's
  with omniASR (user, 2026-11-22: "let's use wav2vec2-base-vietnamese-250h on the neural engine";
  [two-way](two-way.md#vietnamese-aligner)). Each sample records its aligner in `text_timing`.
- Every noisy `en-vi` copy uses `--seed 1` (the `vi-en` copies took seed 0; the draws key on the
  seed and the row id, which both directions share) and `--fraction` 0.6.

## On the Hub

| Folder | Repo (dataset) | Shards | Built by |
|---|---|---:|---|
| `en-vi-grounded-v2/train` (VI→EN clean) | `huybik/hibiki-zero-vi-full-sft` | 1,832 | the Mac, 2026-10/11 |
| `en-vi-grounded-v2-noisy/train` (VI→EN noisy, seed 0) | same | 1,832 | the pod |
| `en-vi/en-vi-grounded-v2/train` (EN→VI clean) | same | 1,832 when done | 527 the boxes (CUDA, TF32), the rest the Mac; omniASR timed the first 528 |
| `en-vi/en-vi-grounded-v2-noisy/train` (EN→VI noisy, seed 1) | same | 1,832 when done | the Mac |
| `en-vi/en-vi-grounded-v2/validation` | same | 3 | the box |
| `en-vi/{giga,pho,vivoice}` (tranche EN→VI clean) | `anquachdev/mimi-en-vi-speech-20k` | 228, 224, 354 | the box |
| `en-vi/{giga,pho,vivoice}-noisy` (tranche EN→VI noisy, seed 1) | same | 228, 224, 354 when done | the Mac |

- The `en_speech` shard names are the VI→EN cache's (`shard_cv27-*`, `shard_libriheavy-*`); the 7
  `cv-anquach` parquets of the dataset are their own cache and not in these folders.
- Give a run a tranche folder or its `-noisy` copy, never both: streams group a tranche's rows by
  `doc` alone, and a copy keeps its row's `doc` and `seq`, so both copies of a sentence would sit
  side by side (1,859 such pairs on viVoice in the validation). `en_speech` rows have no `doc`, so
  its clean and noisy copies land in different streams. A noisy copy keeps 40% of rows clean.
- The trainer takes whatever a folder holds: train on a noisy folder only once the Hub lists all of
  its shards (the run's last line is "done: the Hub holds every shard").
- A shard built on the Mac and one built on a box differ only by near-ties
  ([cache-build](cache-build.md#across-devices)); a noisy shard records its seed, fraction,
  direction and dataset in `source_augment` and each row's conditions in `source_conditions`.
- Validated end to end (2026-11-21, viVoice noisy shards, `cv27-00035` clean and noisy): the
  trainer's packing, tokenizer check, stream plan and assembly take them as box-built shards; a
  Mac-built clean shard has the box's fields and statistics; noisy copies change only the source
  codebooks of the selected rows (59–62%), and re-applying a row's recorded conditions reproduces
  its codes on the CPU.
- The aligner switch validated end to end (2026-11-22, `libriheavy-large-00012-00103` whole): the
  production build (`--ane-ctc`, 208× at thermal state "fair", 21.7 GB peak) kept 3,147 of 3,218
  rows (58 over the frame cap, the VI→EN shard's same 58; 12 with letters outside the alphabet; 1
  text too long; none under 0.55, the lowest 0.68); its noisy copy; the trainer's loading, stream
  plan and assembly beside omniASR-built and VI→EN shards. VI→EN rows built after it are
  byte-identical to the Hub's.
- The evaluation set `en-vi-val-en-noisy` (2 × 180 rows) is built on the training box in
  `/workspace/hibiki-zero/data/eval/` (2026-11-21). `eval_sets` run from another checkout whose
  `data/` is a symlink writes the clips' absolute paths (`repo_display_path` resolves the link);
  its jsonl were rewritten to the repo-relative `data/eval/…` the other sets use.

## The Mac run

`/Volumes/data/datasets/envi_mac/envi_mac.py` (outside the repo, as the box's `envi_rest.py` was),
run from a checkout of main (`.claude/worktrees/envi-run`, from c1e778c on, so its builds use the
current Vietnamese aligner) with the conda Python, under `caffeinate -is`, by `run.sh` beside
it, which restarts it after a failure (at most 20 times, 5 minutes apart); log `run.log`. Started
2026-11-21 17:23 UTC+7 (viVoice and PhoAudiobook copies on the Hub, 354 and 224, that evening);
running since 2026-11-22 09:06 with the `nohup` line in `run.sh`.

- `run.sh` clears the proxy variables first: a shell Claude Code starts routes HTTPS through
  TeamClaude (`127.0.0.1:3456`), and no download or upload goes through that proxy (user,
  2026-11-22).
- Each upload attempt runs in a process of its own, killed and tried again past a deadline of
  120 s plus 5 MB/s of the batch (`UPLOAD_MIN_BYTES_S`). A Hub upload connection from this Mac can
  crawl at ~10 kB/s for hours without a timeout, the link idle at 350 Mbit/s: twice on 2026-11-22
  (04:38, through the proxy, which held the GPU 1.5 h once the next batch was done; 08:05,
  direct), while fresh connections ran 13–20 MB/s.

- Jobs in order, in batches of 20 shards: `tranche` (each tranche `en-vi` shard's noisy copy),
  `noisy` (the 527 `en_speech` `en-vi` shards the boxes built), `build` (the other 1,305:
  `cache_codes --device mlx --direction en-vi --ane-ctc`, then the noisy copy).
- Each batch works in its own `work/<batch>/`: its inputs come down by curl (the dataset parquets
  at `paths.EN_VI_REVISION`, the tranche's `en_audio` parts, the clean shards) while the batch
  before it runs on the GPU; its outputs move to `stage/<batch>/`, upload while the next batch runs,
  are checked against the Hub's listing by name and size, then deleted. A restart uploads what is
  staged, drops `work/` and plans again from the Hub's listings.
- It plans again from the Hub after each pass and ends ("done: the Hub holds every shard") only when
  a plan comes out empty, so a shard another machine published clean but not noisy gets its copy.
- Two ends (user, 2026-11-22): the Mac goes top down in name order, job by job (`tranche`, `noisy`,
  `build`); another machine, run from its own session, goes bottom up, and the two meet through the
  Hub. Before each batch's GPU steps the Mac lists the Hub again and skips the shards published
  meanwhile; a batch already on the GPU may still make a shard the other machine has just
  published, and the later upload replaces it (the same shard, near-ties apart). The other
  machine:
  - builds with main at c1e778c or later, so its shards are timed by the same aligner
    (wav2vec2-base-vietnamese-250h, `min_score` 0.55; [two-way](two-way.md#vietnamese-aligner));
    its CUDA path, the English aligner's code, matched the Mac's build of a shard on the RTX 5090
    ([cache-speed-cuda](cache-speed-cuda.md#rtx-5090));
  - publishes clean shards to `en-vi/en-vi-grounded-v2/train/` and their noisy copies, `--seed 1`
    and `--fraction` 0.6, to `en-vi/en-vi-grounded-v2-noisy/train/` (tranche copies to
    `en-vi/<source>-noisy/` of the tranche cache repo);
  - goes bottom up: [The box run](#the-box-run).
- `touch /Volumes/data/datasets/envi_mac/pause` holds new batches off the GPU (downloads go on);
  delete it to go on. A batch already on the GPU finishes first.
- To stop it, kill `run.sh`, then `envi_mac.py`, then its GPU step, which outlives its parent
  (`pkill -f 'training.prep.(noisy_cache|cache_codes)'`): a restart reuses `work/` and would race a
  leftover step. Its Hub client has a 300 s timeout (`huggingface_hub`'s own waits forever).
- GigaSpeech 2's noisy copies hold ~10 GB more memory than the others: the next shard's clips
  (~5 GB decoded) are queued before the current one encodes.
- The noisy fetch reads the batch's local copy (`--dataset-dir`), so each shard's
  `source_augment.dataset` names a `work/` folder rather than the Hub's dataset (the same files, at
  `EN_VI_REVISION` or the tranche repo's main); its seed, fraction and direction are as recorded.
- `huggingface_hub`'s downloads drop on this Mac every 10–40 MB ("peer closed connection") and
  crawl through their resumes; curl with `-C -` reads the same files at ~30–40 MB/s. Its Xet
  uploads stalled, so the driver sets `HF_HUB_DISABLE_XET=1` before importing it. Both were
  measured through the proxy above; neither has been tried direct since.

## The box run

The other end since 2026-11-22: the RTX 5090 box (Vast 61906288, [tranche-box](tranche-box.md#state))
runs `/workspace/envi_box/envi_box.py`, `envi_mac.py` on CUDA, by `/workspace/envi_box/run.sh`
(the Mac's `run.sh` on the box: at most 20 restarts, 5 minutes apart; log `run.log` beside it).
Copies of both are kept in `/Volumes/data/datasets/envi_mac/box/`. The user had it build first and
`both_heads_z`'s run wait ([on-policy-distillation](on-policy-distillation.md)).

- The jobs and each job's shards in the reverse of the Mac's order: `build` from the last name down,
  then `noisy`, then `tranche`.
- The cache env alone (`/workspace/envs/cache`, built as `envs-min.sh` builds it) runs the driver
  and its steps; `h100.sh setup` refuses the 5090 (`require_gpu`), and nothing here needs the
  `.venv`. huybik's token is read from `$HF_HOME/token`, `HF_TOKEN_AN` from `/workspace/.env`.
- Batches of 40 shards (the GPU idles ~40 s between batches) and `noisy_cache encode
  --batch-seconds 320`, for the 32 GB card; the build at `CUDAModels.window` 512, for the box's
  31 GB of RAM ([cache-speed-cuda](cache-speed-cuda.md#memory)).
- `touch /workspace/envi_box/pause` holds new batches off the GPU. Stop it with `kill` on the PIDs
  of `run.sh` (`run.pid`), `envi_box.py` and its GPU step, never `pkill -f` from an ssh command
  line holding the same pattern ([tranche-box](tranche-box.md#a-box-behind-a-slow-proxy)).

## Speed

On the M4 Pro, 2026-11-21, the Mac otherwise lightly used (speed is real time on the audio the
GPU encodes):

| Step | Speed | Note |
|---|---:|---|
| `en-vi` build (`cache_codes --device mlx --ane-ctc`) | 259–323× | Mimi audio of both clips; the aligner's transformer on the Neural Engine; wav2vec2-base-vietnamese-250h 4–13% faster than omniASR, interleaved (2026-11-22); omniASR 150–161× all on the GPU |
| noisy copy (`noisy_cache encode --device mlx`) | 330–400× | the re-encoded English; 2 decode workers, the conditions on the CPU |
| download (curl, 4 at a time) | ~30–40 MB/s | |
| upload (plain LFS, `HF_HUB_DISABLE_XET=1`) | ~20 MB/s | through Xet a 255 MB batch crawled at ~25 KB/s and stalled |

So, with the Mac to itself: the tranche's copies ~15 h (~5,200 h re-encoded), the 527 `en_speech`
copies ~5 h, the 1,305 builds ~65 h with their copies (~14,000 h of Mimi audio at ~260×, then
~4,500 h re-encoded): ~3.5 days, estimated from the first shards (the English hours from the parts'
sizes). The RTX PRO 6000 beside the heads run built at ~970×: ~14 h for the build, ~3 h for the
copies ([tranche-box](tranche-box.md#state)).

The build was profiled after the run started (2026-11-21): omniASR's CTC took ~68% of the GPU.
Its transformer then ran on the Neural Engine beside Mimi (`--ane-ctc`, 1.7–1.9× fp32, text tokens a
frame apart as often as in float16; [cache-build](cache-build.md#speed-on-the-mac)). The profiling
scripts are in `/Volumes/data/datasets/envi_mac/ane/`. On 2026-11-22 the aligner became
wav2vec2-base-vietnamese-250h (95M parameters against 325M), 4–13% faster again: Mimi, which every
row needs, now holds most of the GPU, so the aligner has little left to give.

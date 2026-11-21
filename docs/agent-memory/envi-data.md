# EN→VI data for the next two-way run

Read when preparing or checking the data of the next two-way run: `en_speech` in both directions,
the noisy copies of every English source (`en_speech`'s and the tranche's voice-cloned English), the
Mac run that builds them (`envi_mac.py`) and where each folder is on the Hub. The build itself is in
[cache-build](cache-build.md), the noisy copies in [noisy-source](noisy-source.md#noisy-cache), the
two-way run in [two-way](two-way.md).
Back to [CONTEXT.md](../../CONTEXT.md) · [Decision](#decision) · [On the Hub](#on-the-hub) ·
[The Mac run](#the-mac-run) · [Speed](#speed)

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
- Every noisy `en-vi` copy uses `--seed 1` (the `vi-en` copies took seed 0; the draws key on the
  seed and the row id, which both directions share) and `--fraction` 0.6.

## On the Hub

| Folder | Repo (dataset) | Shards | Built by |
|---|---|---:|---|
| `en-vi-grounded-v2/train` (VI→EN clean) | `huybik/hibiki-zero-vi-full-sft` | 1,832 | the Mac, 2026-10/11 |
| `en-vi-grounded-v2-noisy/train` (VI→EN noisy, seed 0) | same | 1,832 | the pod |
| `en-vi/en-vi-grounded-v2/train` (EN→VI clean) | same | 1,832 when done | 527 the boxes (CUDA, TF32), the rest the Mac |
| `en-vi/en-vi-grounded-v2-noisy/train` (EN→VI noisy, seed 1) | same | 1,832 when done | the Mac |
| `en-vi/en-vi-grounded-v2/validation` | same | 3 | the box |
| `en-vi/{giga,pho,vivoice}` (tranche EN→VI clean) | `anquachdev/mimi-en-vi-speech-20k` | 228, 224, 354 | the box |
| `en-vi/{giga,pho,vivoice}-noisy` (tranche EN→VI noisy, seed 1) | same | 228, 224, 354 when done | the Mac |

- The `en_speech` shard names are the VI→EN cache's (`shard_cv27-*`, `shard_libriheavy-*`); the 7
  `cv-anquach` parquets of the dataset are their own cache and not in these folders.
- A shard built on the Mac and one built on a box differ only by near-ties
  ([cache-build](cache-build.md#across-devices)); a noisy shard records its seed, fraction,
  direction and dataset in `source_augment` and each row's conditions in `source_conditions`.
- The evaluation set `en-vi-val-en-noisy` (2 × 180 rows) is built on the training box in
  `/workspace/hibiki-zero/data/eval/` (2026-11-21). `eval_sets` run from another checkout whose
  `data/` is a symlink writes the clips' absolute paths (`repo_display_path` resolves the link);
  its jsonl were rewritten to the repo-relative `data/eval/…` the other sets use.

## The Mac run

`/Volumes/data/datasets/envi_mac/envi_mac.py` (outside the repo, as the box's `envi_rest.py` was),
run from this branch's worktree with the conda Python, under `caffeinate -is`; log `run.log` beside
it. Started 2026-11-21 17:23 UTC+7.

- Jobs in order, in batches of 20 shards: `tranche` (each tranche `en-vi` shard's noisy copy),
  `noisy` (the 527 `en_speech` `en-vi` shards the boxes built), `build` (the other 1,305:
  `cache_codes --device mlx --direction en-vi`, then the noisy copy).
- Each batch works in its own `work/<batch>/`: its inputs come down by curl (the dataset parquets
  at `eval_sets.EN_VI_REVISION`, the tranche's `en_audio` parts, the clean shards) while the batch
  before it runs on the GPU; its outputs move to `stage/<batch>/`, upload while the next batch runs,
  are checked against the Hub's listing by name and size, then deleted. A restart uploads what is
  staged, drops `work/` and plans again from the Hub's listings.
- Before each batch's GPU steps it lists the Hub again and skips the shards published meanwhile, so
  a box can take shards from the other end: run it bottom up, since the Mac goes top down in name
  order (the box's `envi_rest.py` also goes top down; reverse its lists first).
- `touch /Volumes/data/datasets/envi_mac/pause` holds new batches off the GPU (downloads go on);
  delete it to go on. A batch already on the GPU finishes first.
- The noisy fetch reads the batch's local copy (`--dataset-dir`), so each batch's `manifest.json`
  names its `work/` folder; the shards carry the settings that matter.
- `huggingface_hub`'s downloads drop on this Mac every 10–40 MB ("peer closed connection") and
  crawl through their resumes; curl with `-C -` reads the same files at ~30–40 MB/s.

## Speed

On the M4 Pro, 2026-11-21, the Mac otherwise lightly used (speed is real time on the audio the
GPU encodes):

| Step | Speed | Note |
|---|---:|---|
| `en-vi` build (`cache_codes --device mlx`) | 167–170× | Mimi audio of both clips; `cv27` shards (10.5 and 21 h); omniASR's CTC alone runs 110× |
| noisy copy (`noisy_cache encode --device mlx`) | 330–400× | the re-encoded English; 2 decode workers, the conditions on the CPU |
| download (curl, 4 at a time) | ~30–40 MB/s | |
| upload (Xet) | 2 shards in 9–19 s | |

So, with the Mac to itself: the tranche's copies ~15 h (~5,200 h re-encoded), the 527 `en_speech`
copies ~5 h, the 1,305 builds ~97 h with their copies (~14,000 h of Mimi audio at ~167×, then
~4,500 h re-encoded): ~5 days, estimated from the first shards (the English hours from the parts'
sizes). The RTX PRO 6000 beside the heads run built at ~970×: ~14 h for the build, ~3 h for the
copies ([tranche-box](tranche-box.md#state)).

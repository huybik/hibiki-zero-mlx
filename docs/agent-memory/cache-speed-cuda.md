# Cache build speed and memory on CUDA

Read when running, tuning or bringing up a grounded cache build (`CUDAModels` in
`training/prep/cache_build.py`) or a noisy copy (`noisy_cache encode --device cuda`) on a GPU,
above all on a new GPU or a box with little RAM: what each lever measured, and on which GPU. How
the build works is in [cache-build](cache-build.md#cuda). Back to [CONTEXT.md](../../CONTEXT.md) ·
[Memory](#memory) · [RTX 5090](#rtx-5090) · [H100 and H200](#h100-and-h200)

## Memory

- A build holds the decoded clips of several windows at once (being decoded, being filled, queued,
  on the GPU), in `/dev/shm` and the main process (both clips at 24 kHz for Mimi, the aligned side's
  at 16 kHz for the CTC, float32). So `CUDAModels.window` sets the build's RAM: on Libriheavy ~5 GB
  plus ~10 MB a window row. 512 rows since 2026-11-22; it was 4,096, sized on boxes with 137 GB or
  more. On four Libriheavy shards on the RTX 5090 (31 GB; anonymous + shared memory, page cache
  excluded):

  | Window | Build | Peak RAM |
  |---:|---:|---:|
  | 4,096 | killed out of memory (a 40-shard batch) | |
  | 2,048 | 2,018× | 26.0 GB |
  | 1,024 | 2,106× | 15.6 GB |
  | 512 | 2,133× | 10.4 GB |
- A Vast container's `/dev/shm` may be smaller than its RAM (15 GB of 31 on the RTX 5090): the
  clips in flight must fit there too.
- Profile a build on several shards, never one: a shard under one window's rows never holds a
  second window, so it hides the memory a long build needs (one 3,218-row shard ran at 4,096 rows,
  then 40 shards were killed).
- A build killed outright (SIGKILL, the out-of-memory killer) cannot clean up after itself. Its
  `/dev/shm` clips hold RAM until deleted (four killed builds left ~11 GB), and its spawned
  workers lived on, blocked on a job queue whose write end they hold (14 idle workers, ~1.4 GB, on
  the RTX 5090). Now each build's directory holds a lock its build keeps (`pcm_dir`), and a build
  first removes every directory whose lock it can take; the workers of the build and of the noisy
  copy exit once their parent is gone (`paths.exit_with_parent`). Read a memory profile with no
  `cache_build_pcm_*` in `/dev/shm` and no `spawn_main` process left from a dead build.
- 512 rows has not run on an H100 or H200, whose builds ran at 4,096: there a window passes in
  ~1.5 s, and the GPU's queue empties twice a window (CTC to Mimi, then to the next window's CTC),
  so measure 512 against a larger window there before a full run.
- The GPU: `nvidia-smi` showed 16 GB for one shard and 25–30 GB over four or a batch of 40
  (PyTorch's cache keeps what it freed); the 32 GB card ran them all.
- `noisy_cache encode` at `--batch-seconds` 320 and 640 (RTX 5090, interleaved twice): 2,128–2,131×
  against 2,137–2,145×, 27.0 against 29.9 GB of the card, codes equal but for near-ties (29 of
  3,147 rows, 99.9989%). Its main process peaked at 4.7 GB of RAM on one shard.

## RTX 5090

Vast 61906288 (2026-11-22): 32 GB, driver 580 (CUDA 13.0), a 15.4-CPU quota, 31 GB of RAM, 15 GB of
`/dev/shm`; the cache env (`/workspace/envs/cache`, free-threaded 3.14) alone. Measured on the
`en-vi` build, wav2vec2-base-vietnamese-250h timing the text:

- One shard (`libriheavy-large-00012-00103`, 3,218 rows): 1,676× with the start (~14 s of loading and
  the first window's decodes, the GPU idle), ~2,200× after it, the GPU 99% busy, the CPU 6–15%.
  Four Libriheavy shards at window 512: 2,133×, the GPU 98% busy while active.
- A production batch of 40 shards (`envi_box.py`, [envi-data](envi-data.md#the-box-run)): the
  build 933 s, the noisy copies 265 s, the upload of both 66 s; 13.2 GB of RAM at peak, the GPU
  98% busy while active. The full run's first batch (Libriheavy-medium, 801 h against that
  batch's 571 h) built in 1,316 s and made its noisy copies in 364 s: the same ~2,200×, on longer
  shards. At that size a 40-shard batch takes ~29 min of GPU, ~15 h for the run's 1,264 builds
  if the Mac took none. A batch's RAM peaks in the noisy fetch, not the build: ~18 GB for ~17 s
  (it reads 16 local parquets' audio at once, `--jobs 16`), against ~13 GB in the build and the
  encode (batches 2 and 3 of the full run).
- Against the Mac's `--ane-ctc` build of that shard (c1e778c): the same 3,147 rows kept and 71
  rejected for the same reasons, every non-code field equal, 22 rows with one word a 12.5 Hz frame
  apart, scores within 0.0105, audio codes 94.9% equal (first codebooks 99.8%): TF32's near-ties,
  as in [cache-build](cache-build.md#across-devices). The Vietnamese aligner's CUDA path, the
  English aligner's code, had been checked on the CPU only.
- `noisy_cache encode`: ~2,100× (7.08 h of re-encoded English in 12 s). Importing `sphn` turns the
  free-threaded interpreter's GIL back on; `noisy_cache` once did through `eval_sets`' constants, and
  with the GIL forced off it ran 2,202–2,209× against 2,118–2,135×. The constants are in `paths` now.
- The noisy fetch reads a local shard's sources in ~1 s (2 s on a 25 h shard). Between batches the
  GPU idles ~40 s: each step's start and the fetch.

## H100 and H200

**H200 (2026-11-09, GigaSpeech 2 shards 228–229 rebuilt and compared with their Hub copies):
1,342× → 3,286× real time**, the GPU 92–94% busy at ~640 W (was 66%, ~500 W). Steps, each measured
on the same rows (bench: a 4,094-row window for the models, whole shards for the build):

| Step | Build | Why |
|---|---:|---|
| fp32, encode in the main thread (before) | 1,342× | GPU idle while the main thread planned, unpickled PCM, collected |
| TF32 (`CUDA_TF32`) + encode in a GPU thread, the next window queued | 2,014× | CTC 11.7 → 4.1 s, Mimi 13.8 → 7.9 s a window |
| decode **and** `plan` in the workers, clips through `/dev/shm` (`cache_rows.Planned`) | 2,311× | the pool's one result pipe moved ~1.4 MB a row: ~30 s of main thread a shard |
| pinned double-buffered staging, batch k+1 launched before k is collected | 3,298× | the GPU thread's host work hid behind kernels: a window 16.5 → 12 s |
| `CUDA_BATCH_S` 1,280 → 320 | 3,286× | models alone 11.3 → 10.7 s a window, 27 → 7.4 GiB peak; end to end within noise |

- TF32's effect on the rows: 0.021% of text token frames move (at most 4 frames), no row changes
  its accept/reject; Mimi's first codebook 99.92% equal, residual ones 96–99%. These are near-ties:
  the fp32 decode of TF32 codes is as close to the source as that of fp32 codes (SNR 9.043 against
  9.040 dB on 256 clips). Shards before 2026-11-09 16:40 UTC are fp32, the rest TF32; both are
  valid encodings. The noisy re-encode (`noisy_cache`) keeps `cuda_mimi`'s fp32 default.
- Rejected: bf16 Mimi (half the residual codes rewritten, ~10% faster than TF32); bf16 CTC (slower
  than TF32, 0.43% of token frames moved; for omniASR too, the `en-vi` aligner: 82 s against 65 s
  on GigaSpeech 2 shards 0–1, the H100 SXM, 2026-11-10); `cudnn.benchmark` (no change); `torch.compile` on Mimi:
  at 1,280 s batches it hits `CUDA_ERROR_ILLEGAL_ADDRESS` (64 channels × 30.7M samples, past int32
  indexing), and at 320 s it runs 14% faster but its codes are wrong (19.7% equal, 0.76 dB SNR).
- A run's start costs ~10 s (the first shard's read and window of decodes). Mimi's time is mostly
  its first SEANet layers at 24 kHz (memory-bound convolutions, ELU, pads), CTC's its layers. The
  workers use ~15 ms of CPU a row (decode 8.9, soxr 2.3, Viterbi ~3): the 30-CPU box idles.
- **The en-vi build (omniASR CTC, the Japan H100 SXM, 2026-11-11): 3,101× → 3,471×**, the GPU
  86% → 98% busy. Each step on GigaSpeech 2 shards 0–2 (the speed between the first shard out and
  the last, inputs fetched first, nothing pushed), every shard compared with its Hub copy:

  | Step | Build | Why |
  |---|---:|---|
  | before | 3,101× | in each ~20 s window Mimi held the GPU at 100% for ~10 s, the CTC at 35–85% for ~9 s |
  | a decode job of 32 rows (`DECODE_JOB_ROWS`), an alignment job of a CTC batch (`align_rows`) | 3,344× | a job a row: the main thread 100 → 51 s of CPU, the pool's manager thread 44 → 8 s, all under the GIL the GPU thread needs |
  | the CTC encoder without its host syncs (`_encoder`, `_within`) | 3,409× | a boolean-mask write, the mask's padding check and a blocking mask copy emptied the GPU's queue every CTC batch |
  | free-threaded Python 3.14 (`/workspace/envs/cache`), Mimi's rope eager | 3,471× | no GIL between the GPU thread and the threads feeding it; CTC 91 → 84 s |

  The text timing came out as the old build's own rebuild (the same 8, 11 and 4 token frames moved
  against the Hub copies, every accept and reject alike). Audio codes: 99.95% equal on 3.12, 99.47%
  on 3.14, all from Mimi's rope run eager (the 3.12 `.venv` under `NO_TORCH_COMPILE=1` gives the same
  codes byte for byte; PyAV 19 decodes as 15.1 to subnormals), near-ties as TF32's. Rejected: CTC batches of 1,280 s (3,344×); Triton under `PYTHON_GIL=0` (died
  without a traceback in its first window). Without Triton (it turns the GIL back on) Mimi's one
  compiled function runs eager (`NO_TORCH_COMPILE=1`); safetensors is built from source.
- What is left on it: a CTC window of 4,096 clips (17,265 s) is 6.3 s of kernels against ~7 s of
  wall: float32 memory-efficient attention 17%, copies 15% (attention outputs, the feature
  extractor's layer norms), TF32 matmuls ~24%, convolutions ~20%, layer norm 7%, GELU 6%. Faster
  needs lower precision (bf16 CTC was rejected) or fused kernels (`torch.compile` needs Triton, so
  the GIL); Mimi's ~8 s a window is untouched. Unmeasured: one worker less, as the main process now
  uses ~1.8 CPUs and 20% of the cgroup's periods throttle. `CUDAModels._encoder` copies
  transformers 5.22.1's encoder forward without its syncs: check it again on an upgrade.
- Method (the scripts were per session): rebuild whole shards already built and compare them, bench a
  window's models for code equality and decode SNR; judge precision by reconstruction.
- CPU: check the main process is not the one throttled (`/sys/fs/cgroup/cpu.stat`): 25 workers on
  the Japan box's 26.9-CPU quota throttled 14% of the 100 ms periods in bursts (2026-11-07).


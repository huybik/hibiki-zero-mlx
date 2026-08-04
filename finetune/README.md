# Vietnamese LoRA Scaffold

Minimal local scaffold for the Vietnamese -> English LoRA water test. It uses the existing
FLEURS vi/en manifests and Hibiki/Mimi weights; it does not touch inference/runtime files.

This is a mechanics README, not the next quality recipe. See the linked
[data](../docs/data_generation_plan.md), [training](../docs/training_plan.md),
and [validation](../docs/validation_plan.md) plans before launching a scaled run.

Run from the repo root with the project Python:

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python
$PY finetune/build_pairs.py --splits train validation test
$PY finetune/cache_codes.py --pairs finetune/pairs/train.jsonl
$PY finetune/cache_codes.py --pairs finetune/pairs/validation.jsonl --out-dir finetune/cache/validation
$PY finetune/train_lora.py --cache-dir finetune/cache/train --max-steps 10
$PY finetune/validate_lora.py \
  --cache-dir finetune/cache/validation \
  --adapter finetune/runs/vn_lora/adapter_step000010.safetensors \
  --batch-size 8
$PY finetune/train_lora.py --resume-checkpoint finetune/runs/vn_lora/trainer_step000010.pt --max-steps 100 --mps-empty-cache-every 10
$PY finetune/eval_lora.py \
  --pairs finetune/pairs/validation.jsonl \
  --adapter finetune/runs/vn_lora/adapter_step000010.safetensors \
  --limit 2 \
  --text-only
```

`cache_codes.py` and `train_lora.py` require the PyTorch `moshi` and `sphn` packages in that
environment. They intentionally fail before doing work if those imports are missing.

See `docs/finetune.md` for the full design, the schedule/selection flags, the
AutoResearch protocol, and recommended 128 / 512 / 1449 / CUDA commands.

## Scripts

- `common.py` is the shared toolkit (device/dtype, cached-shard dataset + loader,
  LoRA insertion + adapter save/load, teacher-forced losses, greedy generation,
  BLEU/chrF/WER + loop metrics, and the `value@fraction` schedule primitives). The
  three scripts below are thin wrappers over it.
- `build_pairs.py` reads `remote_dataset/fleurs_vi_en/{train,validation,test}/manifest.csv`,
  validates audio files, and writes deterministic `finetune/pairs/{split}.jsonl` files
  plus `val16.jsonl` / `val128.jsonl` held-out gate subsets (first-N of validation).
- `cache_codes.py` loads Mimi and the text tokenizer, then writes resumable `shard_*.pt`
  caches. Each sample contains `codes[33, T]`:
  - `0`: English text tokens, padded with Hibiki text pad id.
  - `1..16`: English target Mimi codes.
  - `17..32`: Vietnamese source Mimi codes plus a source EOS frame.
  By default it applies coarse target alignment from the Hibiki-Zero paper at FLEURS scale:
  English audio is left-padded by a deterministic delay sampled from
  `[0, 0.5 * vi_duration_s]`, and English text starts after the same delay. Use
  `--target-delay-ratio 0` to disable this.
- `cache_vivos_full.py` builds `hibiki_vn_lora_cache_v2` train/dev shards only
  after the full VIVOS QA report says `go`. It retains row-level source,
  translation, TTS, QA, alignment, speaker/stratum, and Mimi provenance, then
  audits exact accepted/cache coverage, tensor/code ranges, source EOS,
  degenerate codebooks, and supervision-token accounting. `common.py` accepts
  both v1 and v2 shards; legacy v1 rows are labeled `legacy_unspecified`.
- `release_vivos_cache.py` prepares the one immutable VIVOS cache release under
  `releases/v2/vivos_qwen3_tts_mlx_v3_full_v1`, publishes that exact bundle in
  one optimistic-concurrency dataset commit, and records success only after a
  clean snapshot/extraction reruns the cache audit. It never deletes or squashes
  Hub history and refuses an existing local release or remote prefix.
- `train_lora.py` loads the PyTorch LM, freezes everything, applies LoRA to selected
  targets, trains CE on `LMModel.forward` masks, and saves adapter `.safetensors`
  plus optimizer checkpoints. It appends scalar logs to `train_log.jsonl`, can log
  teacher-forced cache validation to `val_log.jsonl` with `--val-cache-dir`, and
  supports piecewise `value@fraction` schedules for loss weights
  (`--text-weight-schedule` / `--audio-weight-schedule`), replay weight
  (`--replay-weight-schedule`), and per-group LR (`--lr-schedule` /
  `--text-head-lr-schedule` / `--audio-head-lr-schedule` + `--warmup-steps`) — each
  degrading to the matching static flag. `--eval-every N` runs an in-training batched
  greedy val eval and saves `adapter_best.safetensors` on chrF improvement. Default
  dtype is `bfloat16`; MPS `float16` can go non-finite after the first optimizer step.
  `--seed` controls model and data-order randomness. For long CUDA batches,
  `--frame-batch-schedule` filters at `--max-frames`, builds length-sorted batches in
  cumulative frame buckets, and shuffles only whole batches per seeded epoch. It replaces
  `--batch-size` and is intentionally incompatible with the row-level replay sampler.
  `--val-batch-size` independently controls cached teacher-forced validation. The
  `288:10,384:8,512:5` schedule is only an unvalidated candidate estimate: benchmark
  max-length forward/backward on the target H100 and require at least 5 GB VRAM headroom
  before selecting any bucket sizes.
- `validate_lora.py` computes teacher-forced CE on cached rows for the base model or
  an adapter. It tracks adequacy/overfit but cannot detect free-running collapse;
  greedy evaluation is mandatory for that failure mode.
- `eval_lora.py` loads an adapter, runs PyTorch generation on a pair file, writes
  references/predictions in `predictions.csv`, and writes `metrics.json` with
  nonempty/EOS/exact plus BLEU/chrF/WER when `sacrebleu` is installed. Use
  `--text-only` for fast gates, or omit it to also decode/write wav outputs.

## Defaults

Weights are expected in `weights/`:

- `config.json`
- `hibiki-pytorch-77f82164@110.safetensors`
- `mimi-pytorch-e351c8d8@125.safetensors`
- `tokenizer_spm_48k_multi6_2.model`

Outputs stay under `finetune/`: `pairs/`, `cache/`, and `runs/`.

## Limitations

This is a mechanics scaffold, not a quality recipe. FLEURS is small, target/source recordings are
only coarsely aligned by frame padding, and there is no RL, old-language replay, LoRA merge, or MLX
conversion here. Missing weights, data, or Python deps fail loudly.

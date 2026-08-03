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

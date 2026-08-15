# Vietnamese SFT mechanics

The training stack uses the PyTorch `moshi` package and is separate from the MLX
inference runtime. It supports one path: full-model SFT from the upstream
Hibiki-Zero 3B base checkpoint on CUDA.

## Inputs

Place these files in `weights/`:

- `config.json`
- `hibiki-pytorch-77f82164@110.safetensors`
- `mimi-pytorch-e351c8d8@125.safetensors`
- `tokenizer_spm_48k_multi6_2.model`

Training consumes cached `shard_*.pt` files. Each sample is `codes[1+n_q, T]`:
English text, English target Mimi codes, then Vietnamese source Mimi codes with
a source-EOS frame. `cache_codes.py` builds FLEURS caches;
`cache_phomt_stream.py` builds the large published PhoMT cache one parquet shard
at a time.

```bash
python remote_dataset/download_fleurs_vi_en.py --split train
python remote_dataset/download_fleurs_vi_en.py --split validation
python finetune/build_pairs.py --splits train validation
python finetune/cache_codes.py --pairs finetune/pairs/train.jsonl
python finetune/cache_codes.py \
  --pairs finetune/pairs/validation.jsonl \
  --out-dir finetune/cache/validation
```

## Trainer invariants

- Every model parameter is trainable; there is no adapter or freeze map.
- CUDA uses fp32 master weights and bf16 autocast.
- `--max-frames 280` bounds memory; fixed length-sorted batches minimize padding.
- Text prefix PAD has weight 0.5 by default; content and EOS remain weight 1.
- Learning-rate, text-loss, and audio-loss schedules use `value@fraction` syntax.
- Checkpoints contain the exact full model. Loading rejects missing or extra keys.
- `--resume-checkpoint` is only for interruption recovery in the same run.

Teacher-forced validation is diagnostic. `--eval-every` performs deterministic
free-running evaluation and writes predictions plus generation-health metrics.
Final selection follows [validation_plan.md](validation_plan.md).

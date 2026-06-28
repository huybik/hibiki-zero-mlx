# Vietnamese LoRA Scaffold

Minimal local scaffold for the Vietnamese -> English LoRA water test. It uses the existing
FLEURS vi/en manifests and Hibiki/Mimi weights; it does not touch inference/runtime files.

Run from the repo root with the project Python:

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python
$PY finetune/build_pairs.py --splits train validation test
$PY finetune/cache_codes.py --pairs finetune/pairs/train.jsonl
$PY finetune/train_lora.py --cache-dir finetune/cache/train --max-steps 10
```

`cache_codes.py` and `train_lora.py` require the PyTorch `moshi` and `sphn` packages in that
environment. They intentionally fail before doing work if those imports are missing.

## Scripts

- `build_pairs.py` reads `remote_dataset/fleurs_vi_en/{train,validation,test}/manifest.csv`,
  validates audio files, and writes deterministic `finetune/pairs/{split}.jsonl` files.
- `cache_codes.py` loads Mimi and the text tokenizer, then writes resumable `shard_*.pt`
  caches. Each sample contains `codes[33, T]`:
  - `0`: English text tokens, padded with Hibiki text pad id.
  - `1..16`: English target Mimi codes.
  - `17..32`: Vietnamese source Mimi codes plus a source EOS frame.
  By default it applies coarse target alignment from the Hibiki-Zero paper at FLEURS scale:
  English audio is left-padded by a deterministic delay sampled from
  `[0, 0.5 * vi_duration_s]`, and English text starts after the same delay. Use
  `--target-delay-ratio 0` to disable this.
- `train_lora.py` loads the PyTorch LM, freezes everything, applies LoRA only to
  `LMModel.transformer`, trains CE on `LMModel.forward` masks, and saves adapter
  `.safetensors` plus optimizer checkpoints. It also appends scalar logs to
  `finetune/runs/vn_lora/train_log.jsonl`.

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

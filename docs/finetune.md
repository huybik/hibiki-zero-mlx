# Finetune (Track A): Vietnamese LoRA on the 3B main transformer

PyTorch/MPS (CUDA-portable) LoRA stack that adapts Hibiki-Zero to a **new source
language (Vietnamese → EN)** by fine-tuning only the main transformer (depformer /
Mimi / embeddings frozen). Separate from the MLX inference runtime; uses the
`moshi` pip package + `sphn` + `safetensors` in the conda base env. Phase 4 of the
refactor plan is **done** (consolidation + schedules + selection + val128).

## Design (locked)

- **Freeze map:** everything `requires_grad=False`, then LoRA on `LMModel.transformer`;
  optional full `text_linear` (`--train-text-head`) and audio-head LoRA on
  `depformer_in`+`linears` (`--train-audio-heads`). New LoRA `B` is **zero-init**
  (random-init spikes loss). Audio loss still backprops through the frozen depformer.
- **Cache codes once:** `cache_codes.py` writes `codes[1+n_q, T]` shards — row 0 EN
  text tokens (prefix-pad supervised, tail-pad masked), rows `1..dep_q` EN target Mimi
  codes, rows `dep_q+1..` VI source codes + source-EOS. Mimi is not in the training loop.
- **MPS memory:** `--dtype bfloat16 --batch-size 2 --grad-accum-steps 2` (batch 4 spikes
  the 48 GB driver). Eval/validate can use `--batch-size 8`. `float16` goes non-finite
  after the first step — bf16 is the default.
- **Device portability:** all `torch.mps.*` calls (empty_cache/synchronize, memory
  stats) are gated behind `common.is_mps(device)`, so `--device cuda` runs clean.

## Layout

- `common.py` — the shared toolkit: device/dtype/seed, cached-shard dataset + loader +
  replay sampler, LoRA insertion + adapter save/load (metadata-driven), teacher-forced
  losses, greedy generation, BLEU/chrF/WER + loop metrics, and the schedule primitives.
  `train_lora.py` / `eval_lora.py` / `validate_lora.py` are thin wrappers over it.
- `build_pairs.py` FLEURS manifests → `pairs/{split}.jsonl` (+ deterministic
  `val16.jsonl` / `val128.jsonl` held-out gate subsets, first-N of validation).
- `fetch_phomt.py` HF `anquachdev/PhoMT-en-vi-speech` (real EN+VI speech) →
  `remote_dataset/phomt_en_vi/{en,vi}/*.wav` + `pairs/phomt_train.jsonl` (989 pairs,
  drops vi>25 s). Reads `HF_TOKEN` from `.env`; same 8-field pair schema as `build_pairs`.
- `cache_codes.py` → `cache/{train,validation,phomt_train,...}/shard_*.pt`.
  `train_lora.py --cache-dir` accepts **multiple dirs** — shards from all are pooled, so
  FLEURS + PhoMT train together without re-encoding FLEURS.
- `autoresearch.py` — fixed trial runner (subprocess), TSV protocol, primary = val chrF.

## Schedules (all CLI, piecewise-constant `value@fraction`)

Every static flag is the degenerate single-point schedule, so old commands run unchanged.
`fraction` is a fraction of total optimizer steps (`--max-steps`, else epochs×steps/epoch).

- **Loss weights:** `--text-weight-schedule "5@0,2@0.6"`, `--audio-weight-schedule`.
  Fall back to `--text-loss-weight` / `--audio-loss-weight`.
- **Replay:** `--replay-weight-schedule "300@0,100@0.5"` (needs `--replay-ids`);
  reuses the WeightedRandomSampler, rebuilt at each boundary. Falls back to `--replay-weight`.
- **Per-group LR:** `--lr-schedule` (transformer LoRA), `--text-head-lr-schedule`,
  `--audio-head-lr-schedule`, plus `--warmup-steps N` (linear warmup, all groups). Fall
  back to `--lr` / `--text-head-lr` / `--audio-head-lr`. Groups collapse to one when
  their schedules match.

## Selection & speed

- `--eval-every N` runs a **batched greedy val eval** (`--eval-pairs`, `--eval-limit`
  128, `--eval-batch-size` 8, `--eval-text-temp 0.0`), logs chrF to
  `greedy_eval_log.jsonl`, and saves `adapter_best.safetensors` + `best.json` on chrF
  improvement. Mimi is loaded only when `--eval-every>0`.
- Teacher-forced CE validation stays available via `--val-cache-dir` + `--val-every`.

## AutoResearch protocol (keep it)

`autoresearch.py {run-trial | run-staged-trial | record-existing}` drives train →
validate (seen3 + val CE) → greedy eval (seen3 anchors + val16/val128) → append one
row to `finetune/autoresearch/results.tsv` (gitignored). Primary metric = val chrF;
secondary gates = nonempty rate, EOS rate, overlong / repeated-4gram loop metrics.
One hypothesis per commit; keep only if the primary metric improves. `run-staged-trial`
expresses replay/text-weight staging as two chained train stages (now also expressible
in a single run via the schedule flags above).

Gate discipline: `seen_first3` / `short16` for smoke mechanics only; **val128 for real
decisions**; full 1449 last. 5/16-row deltas are noise.

## Recommended next commands

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/bin/python
export HF_HOME="$(pwd)/.hf_cache"                                    # HF_HOME may point at a dead mount
# One-time: build pair files + val128 gate, cache codes.
$PY finetune/build_pairs.py --splits train validation test          # writes val16/val128 too
$PY finetune/cache_codes.py --pairs finetune/pairs/train.jsonl
$PY finetune/cache_codes.py --pairs finetune/pairs/validation.jsonl --out-dir finetune/cache/validation
# Add PhoMT real-speech data (fetch + cache into its own dir).
$PY finetune/fetch_phomt.py
$PY finetune/cache_codes.py --pairs finetune/pairs/phomt_train.jsonl --out-dir finetune/cache/phomt_train

# Combined FLEURS+PhoMT vi->en run (pool both caches; --eval-every 0 avoids the slow
# in-training val128 greedy, then eval audio at the end with eval_lora --stop-on-eos).
$PY finetune/train_lora.py --cache-dir finetune/cache/train finetune/cache/phomt_train \
  --out-dir finetune/runs/vn_phomt_combined --dtype bfloat16 --batch-size 2 --grad-accum-steps 2 \
  --max-steps 1000 --train-text-head --train-audio-heads --lora-rank 32 \
  --text-weight-schedule "5@0,2@0.6" --lr-schedule "1e-4@0,3e-5@0.6" --warmup-steps 20 --eval-every 0
$PY finetune/eval_lora.py --adapter finetune/runs/vn_phomt_combined/adapter_step001000.safetensors \
  --pairs finetune/pairs/val16.jsonl --limit 16 --out-dir finetune/runs/vn_phomt_combined/eval_audio

# 128-row decision run on MPS with schedules + in-training best-on-chrF selection.
$PY finetune/train_lora.py --cache-dir finetune/cache/train --out-dir finetune/runs/vn_sched \
  --dtype bfloat16 --batch-size 2 --grad-accum-steps 2 --max-steps 512 \
  --train-text-head --train-audio-heads --lora-rank 32 \
  --text-weight-schedule "5@0,2@0.6" --replay-ids 213,211,245 \
  --replay-weight-schedule "300@0,100@0.5" --lr-schedule "1e-4@0,3e-5@0.6" --warmup-steps 20 \
  --eval-every 128 --eval-pairs finetune/pairs/val128.jsonl --eval-limit 128 --eval-batch-size 8

# Larger CUDA run (same command; MPS-only calls auto-disable off MPS).
$PY finetune/train_lora.py --device cuda --dtype bfloat16 --batch-size 16 --grad-accum-steps 1 \
  --cache-dir finetune/cache/train --max-steps 4000 --train-text-head --train-audio-heads \
  --lora-rank 32 --eval-every 500 --eval-pairs finetune/pairs/val128.jsonl --eval-limit 128
```

## Limitations

Mechanics/optimization scaffold, not a finished quality recipe: FLEURS is small,
source/target are only coarsely aligned by frame padding, and there is no LoRA merge or
MLX conversion here. The depformer-frozen LoRA-on-main bet may still cap quality (per
`distill_plan` Track A) — that is exactly the cheap hypothesis these schedules probe.
Missing weights/data/deps fail loudly.

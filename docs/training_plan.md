# Direct Vietnamese training recipe

There is one training recipe: full-model, direct voice-preserving simultaneous
translation from Vietnamese speech to English text and speech.

## Model and stream contract

Initialize from the exact upstream Hibiki-Zero artifacts:

- `weights/hibiki-pytorch-77f82164@110.safetensors`;
- `weights/mimi-pytorch-e351c8d8@125.safetensors`;
- `weights/config.json`;
- `weights/tokenizer_spm_48k_multi6_2.model`.

Do not initialize from an ASR or prior SFT checkpoint. A trainer checkpoint is
valid only to resume the same interrupted run.

Each cached row remains untransformed:

- target audio: cached English Mimi codes with their native `-1` validity mask;
- target text: cached CTC-timed English tokens ending in tokenizer EOS;
- source audio: cached Vietnamese Mimi codes ending in explicit codec-card EOS.

Target audio is present as teacher forcing. Audio CE and text CE both have
weight `1.0`; prefix text PAD has weight `0.05`. There is no post-source
transform, ASR replay, contrastive or anti-repetition loss.

## Frozen data receipt

Training uses the published grounded-v2 PhoMT and FLEURS caches exactly as
verified by `finetune/freeze_full_data_receipt.py`:

| Property | Value |
| --- | ---: |
| Frozen rows | 719,120 |
| PhoMT rows | 683,164 |
| FLEURS rows | 35,956 |
| Selection weights | 0.95 / 0.05 |
| Selection seed | 42 |
| Raw train max frames | 280 |
| Physical batch | 16 |
| Steps per epoch | 44,945 |

Membership and order are frozen in
`finetune/runs/grounded_v2_full_direct_receipt/sample_manifest.jsonl`. The
training loader follows that manifest without reshuffling.

Validation uses the raw grounded-v2 FLEURS validation cache:

| Property | Value |
| --- | ---: |
| Eligible rows | 138 |
| Raw max frames | 280 |
| Observed max frames | 277 |
| Batch | 4 |
| Shuffle | false |

Validation is deterministic and length-sorted. The smoke reverses it only to
exercise the longest row; production never shuffles it.

## Optimizer and runtime

The fixed production arguments are:

```text
epochs=5, total_steps=224725
batch_size=16, grad_accum_steps=1
max_frames=280, val_max_frames=280, val_batch_size=4
lr=1e-6
optimizer=fused AdamW, betas=(0.9,0.95), weight_decay=0.1
grad_clip=1.0
audio_loss_weight=1.0, text_loss_weight=1.0, text_pad_loss_weight=0.05
seed=42, frame_bucket=16, torch_compile=true
val_every=9000, save_every=9000, keep_checkpoints=2, log_every=10
```

All parameters train. CUDA holds fp32 master weights and uses bf16 autocast.
There is no LR schedule, warmup, accumulation, gradient checkpointing, or sixth
epoch.

## Preflight and launch

The pod must have one compute-capability-9.0 H100 with at least 90 GiB VRAM,
driver 570+, Torch `2.8.0+cu128`, Moshi `0.2.13`, at least 110 GiB host RAM, and
at least 190 GiB free disk.

```bash
export HIBIKI_RECIPE=grounded-v2
export HIBIKI_HF_REPO=huybik/hibiki-zero-vi-full-sft
./finetune/h100.sh preflight
./finetune/h100.sh smoke
./finetune/h100.sh train
```

`train` is allowed only after a commit-matched smoke and only when the remote
`grounded_v2_full_direct_voice_5epoch/` prefix is empty. The full run directory is
`finetune/runs/vi_grounded_v2_full_direct_voice_5epoch`.

## Validation and checkpoints

Teacher-forced validation runs every 9,000 steps and at the final step when it
is not already on cadence. It covers all 138 rows and logs total, audio, text,
content, and PAD loss/accuracy. The lowest total teacher-forced validation loss
is promoted and uploaded as the current best model.

A complete recovery point is a matching
`model_stepNNNNNN.safetensors` / `trainer_stepNNNNNN.pt` pair. The trainer is
published after the model and acts as the remote completion marker. The newest
two complete pairs are retained. The final step 224,725 is always saved and
synced even though it is off the 3,000-step cadence.

The remote prefix also stores:

- `run.json` for the immutable run identity and Git commit;
- `metadata/run_config.json`;
- `metadata/sample_manifest.jsonl`;
- `metadata/full_data_receipt.json`;
- compact train and validation logs under `artifacts/` at final sync.
- the latest complete `best/best_stepNNNNNN.safetensors` plus its JSON marker.

Resume requires the newest complete pair in the original run directory, exact
metadata, exact optimizer state, the current promoted best model/marker after
step 9,000, and the commit recorded by `run.json`:

```bash
./finetune/h100.sh resume \
  finetune/runs/vi_grounded_v2_full_direct_voice_5epoch/trainer_stepNNNNNN.pt
```

See [finetune.md](finetune.md) for complete restore commands.

## Stop conditions

Stop on non-finite loss, artifact/receipt/manifest mismatch, missing target-audio
tokens, changed fixed LR or optimizer values, an incomplete resume pair, or
three consecutive recovery-sync failures. Do not change the receipt or extend
the run; recover the exact run from its newest complete pair.

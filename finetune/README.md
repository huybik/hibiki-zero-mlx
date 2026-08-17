# Vietnamese full-model SFT

The only supported CUDA path is direct full-data Vietnamese-to-English
voice-preserving simultaneous translation from upstream Hibiki-Zero.

The cached English target Mimi audio, CTC-timed English text and EOS, Vietnamese
source Mimi audio and explicit EOS are consumed unchanged. Target audio is
teacher-forced and trained with audio CE. Pilot, ASR warm-start/replay,
post-source, contrastive, anti-repetition, and multi-epoch modes are obsolete.

Core files:

- `h100.sh`: pinned H100 setup, cache build, preflight, smoke, train, and resume.
- `train.py`: one-epoch full-model training and exact recovery checkpoints.
- `freeze_full_data_receipt.py`: immutable membership, stream, and validation contract.
- `validate.py`: teacher-forced diagnostics.
- `eval.py`: optional correct-source free-running text/audio evaluation.
- `common.py`: shared cache, loss, checkpoint, generation, and metric logic.
- `hf_sync.py`: two-pair recovery under `grounded_v2_full_direct_voice`.

See [pod handoff](../docs/finetune.md),
[training recipe](../docs/training_plan.md), and
[validation](../docs/validation_plan.md).

# Vietnamese full-model SFT

The only supported CUDA path is direct full-data Vietnamese-to-English
voice-preserving simultaneous translation from upstream Hibiki-Zero, launched by
`training/h100.sh`.

The cached English target Mimi audio, CTC-timed English text and EOS, Vietnamese
source Mimi audio and explicit EOS are consumed unchanged. Target audio is
teacher-forced and trained with audio CE.

Modules, run from the repository root as `python -m training.finetune.<module>`:

- `train`: five-epoch full-model training and exact recovery checkpoints.
- `freeze_full_data_receipt`: immutable membership, stream, and validation contract.
- `validate`: teacher-forced diagnostics.
- `eval`: optional correct-source free-running text/audio evaluation.
- `common`: shared cache, loss, checkpoint, generation, and metric logic.
- `hf_sync`: two-pair recovery plus promoted-best upload under
  `grounded_v2_full_direct_voice_5epoch`.

See [pod handoff](../../docs/finetune.md),
[training recipe](../../docs/training_plan.md), and
[validation](../../docs/validation_plan.md).

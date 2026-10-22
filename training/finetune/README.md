# Full-model SFT

`training/h100.sh train` runs `training.finetune.train` on one H100 with HF disaster
recovery. Training starts from upstream Hibiki-Zero, updates every parameter, and
consumes grounded caches unchanged: English target Mimi audio is teacher-forced and
trained with audio CE alongside the CTC-timed English text.

Modules, run from the repository root as `python -m training.finetune.<module>`:

- `train`: full-model training, best-model promotion, and exact recovery checkpoints.
- `validate`: teacher-forced diagnostics on a cache.
- `eval`: correct-source free-running text/audio evaluation.
- `common`: shared cache dataset, loss, checkpoint, generation, and metric logic.
- `hf_sync`: two-pair recovery plus promoted-best upload under the run's HF prefix.

See the [pod guide](../../docs/finetune.md).

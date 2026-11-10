# Full-model SFT and GRPO

`training/h100.sh run` starts or continues `training.finetune.train` on one H100 with HF
disaster recovery; `training/h100.sh grpo` does the same for `training.finetune.grpo`, the stage
after SFT. Training starts from upstream Hibiki-Zero, updates every parameter, and trains
on multi-sentence streams joined from grounded cache rows (the cache itself is never
changed): English target Mimi audio is teacher-forced and trained with audio CE alongside
the CTC-timed English text.

Modules, run from the repository root as `python -m training.finetune.<module>`:

- `train`: full-model training, checkpoint promotion on free-running chrF of real Vietnamese
  speech (FLEURS dev) once every evaluation set passes the health gates, and exact recovery
  checkpoints written in the background; it resumes the newest one
  in its out-dir.
- `grpo`: GRPO from an SFT checkpoint (rollouts, BLEU process rewards in `rewards`, the clipped
  update), promoting as `train` does; `grpo_profile` times its stages before a run.
- `validate`: teacher-forced diagnostics on a cache's validation streams.
- `eval`: correct-source free-running text/audio evaluation.
- `streams`: memory-mapped cache rows (from per-shard packs), stream planning and assembly,
  frame-budget batches.
- `lm_training`: moshi's LM made faster for training and generation, with the same math.
- `common`: what the scripts share: the model and optimizer, the loss kernel and losses,
  free-running translation, metrics and the health gates.
- `checkpoints`: a run directory: checkpoint pairs, resume and its contract, promotion, logs.
- `hf_sync`: two-pair recovery plus promoted-best upload under the run's HF prefix.

See the [pod guide](../../docs/finetune.md).

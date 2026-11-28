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
- `grpo`: GRPO from an SFT checkpoint, promoting as `train` does: its rollouts and their BLEU
  process rewards (`grpo_rollout`, `rewards`) and the clipped update (`grpo_objective`);
  `grpo_profile` times its stages before a run.
- `validate`: teacher-forced diagnostics on a cache's validation streams.
- `eval`: correct-source free-running text/audio evaluation.
- `packs`: per-shard packs of a cache, the only form a run reads (`--drop-shards` deletes the
  packed shards); `rows`: the rows a run loads from them, memory-mapped, and where each target
  may start; `streams`: stream planning and assembly, frame-budget batches; `rooms`: room noise
  in the source between sentences.
- `lm_training`: moshi's LM made faster for training and generation, with the same math.
- `model`: the model, tokenizer, policy, teacher and optimizer as every script loads them
  (importing it installs `lm_training`); `losses`: the loss kernel and teacher-forced losses;
  `distill`: the chunked distillation that skips the full logits.
- `eval_rows`: evaluation sets and their source audio; `generation`: free-running translation,
  its outputs and the trainers' validation; `metrics`: its scores and the health gates.
- `checkpoints`: a run directory: resume and its contract, promotion, logs; `run_files`: its
  checkpoint pairs and `best.json`, as `checkpoints` and `hf_sync` both read them (no torch).
- `hf_sync`: two-pair recovery plus promoted-best upload under the run's HF prefix, freeing what it prunes.

See the [pod guide](../../docs/finetune.md).

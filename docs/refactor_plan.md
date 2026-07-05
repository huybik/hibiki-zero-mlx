# Aggressive Refactor Plan — iPhone-Realtime MLX Hibiki

**Goal:** a clean, minimal repo whose single product is a **realtime MLX speech-translation runtime
that fits the iPhone 80 ms/frame budget with decent quality**, plus the training code needed to get
there (Track B parallel head + Track A finetune stack).

Execution rules: one agent per phase, run sequentially, commit after each phase, update
`CONTEXT.md` at the end of every phase (rewrite stale sections, don't append). Verify before every
deletion (`grep -rn` for imports/references). Gates that must pass before any phase's commit that
touches the runtime: `python scripts/verify_mlx_q4.py` produces coherent `leon` output, and the
**silence-in test** (zeros in → rms < 0.10, peak < 1.1).

Known facts that drive the plan (measured, do not re-derive):
- Per frame on M4: main transformer ~8.6 ms, depformer ~15.7 ms (64%, launch-bound at 0.72 ms/slice
  × 16), codec fully hidden by the 3-thread pipeline. Quantization gives the depformer **zero**
  speedup; only the parallel head (Track B) removes the 16 sequential passes.
- q4 gs32 is mandatory for moshi-swift compatibility. Depformer q3 is quality-safe, q2 is not.
- Hibiki-M 1B q4 (1.125 GB) is already staged/published — the natural phone-size base.
- The PyTorch `hibiki_zero` serve/generate stack (~0.7× RT on MPS) is superseded by the MLX path
  (3× RT) and is not on the iPhone path.

---

## Phase 1 — Repo restructure & dead-code purge

**Owner-agent deliverable:** a repo where every remaining file is on the iPhone path (runtime,
conversion, training, eval) and imports work without `sys.path` hacks.

Target layout:

```
code/
├── pyproject.toml        # installs hibiki_mlx as a package (editable)
├── main.py               # thin CLI: file + --mic, delegates to hibiki_mlx
├── hibiki_mlx/           # THE runtime package (renamed from src/)
│   ├── __init__.py
│   └── pipeline.py       # from src/infer_mlx_fast.py: load()/run()/mic pipeline
├── moshi-mlx/            # vendored fork, unchanged location (own editable install)
├── scripts/              # conversion + verification + profiling ONLY
│   ├── convert_mlx_q4.py, convert_hibiki_m_mlx_q4.py
│   ├── verify_mlx_q4.py, profile_mlx.py
│   └── push_mlx_q4.py, push_hibiki_m_mlx_q4.py
├── finetune/             # Track A training stack (kept, refactored in Phase 4)
├── remote_dataset/       # benchmark/eval datasets (CoVoST2, FLEURS)
├── training-data/        # PhoMT TTS pipeline (kept)
├── docs/  vision/  reports/
└── weights/              # gitignored artifacts (paths unchanged)
```

Delete (verify each with `grep -rn` across the repo for imports/CLI references first):
- `frontend/` (465 MB), `build_frontend.sh`, `hibiki_zero/static/` — web UI of the dead PyTorch path.
- `hibiki_zero/` — the whole PyTorch inference stack (confirm `finetune/` imports `moshi`, not
  `hibiki_zero`, before deleting). Remove its entry from `pyproject.toml`.
- `src/mlx_hibiki_patch.py` — shim already published on HF (`huybik/hibiki-zero-3b-mlx-q4`); dead locally.
- Concluded one-shot experiments: `scripts/exp_quant.py`, `exp_sampler.py`, `exp_warmup_probe.py`,
  `exp_text_temp_bench.py`, `scripts/convert_mlx_bf16.py`, `scripts/infer_mlx_bf16.py`
  (findings already recorded in CONTEXT.md).
- `remote_dataset/download_french_english_unprocessed.py` + its `reports/benchmarks/french_english_unprocessed/`.
- Empty/stale dirs: `report/`, `datasets/` (inspect first), `__pycache__/`, `.DS_Store` files.
- `frontend`-related config anywhere (README, workflows). Check `.github/workflows/publish-package.yml`
  — if it publishes the deleted PyTorch package, delete or repoint it.

Restructure:
- `src/infer_mlx_fast.py` → `hibiki_mlx/pipeline.py`; make `hibiki_mlx` a real installable package
  (pyproject `[project]` + editable install into the project venv); kill every `sys.path.insert`.
- `main.py` becomes a thin argparse CLI importing `hibiki_mlx`.
- Fix all script imports; scripts keep anchoring weight paths on repo root.
- Rewrite `README.md` around the MLX runtime (what it is, how to run file/mic, how to convert,
  link to docs). Rewrite `CONTEXT.md`: drop the MPS-patch section and PyTorch run instructions,
  compress history into current-state facts.

Gate: `python main.py hibiki_zero/samples/leon.wav`-equivalent still works after the move (samples
must be relocated, e.g. `assets/samples/`), `verify_mlx_q4.py` passes, silence-in passes.
Commit: `Phase 1: restructure repo around MLX runtime, purge PyTorch stack`.

---

## Phase 2 — Inference optimization sweep (no-training strategies)

**Deliverable:** a single benchmark harness + a decision table that picks the phone configuration,
and every cheap runtime win landed. All measured on M4; iPhone numbers are projections until Phase 5.

Strategies (each = measure → keep-if-wins):
1. **Hibiki-M 1B as the phone base.** Benchmark the staged `weights/hibiki-m-mlx-q4` end-to-end:
   per-stage profile (main/depformer/codec) + CoVoST2 fr→en n=30 BLEU/chrF via the existing
   `remote_dataset/run_batch.py` path. Expected: main ≪ 8.6 ms; depformer still 16-slice-bound.
   This is the primary phone candidate — the 3B stays the Mac/teacher model.
2. **Depformer launch-count reduction.** The 16 slices are sequential by design, but each slice is
   6 layers ≈ 23 launches. Try `mx.compile` of the whole per-slice step (fixed single-token shapes,
   cache offset as runtime array, same trick as the main-transformer fast path), and lazy chaining
   with a single `mx.eval` per frame. Target: cut ~370 launches/frame meaningfully. Keep only if
   >5% frame-time win; record either way.
3. **Quant matrix for the phone.** q4-gs32 baseline; + depformer-q3 variant (bandwidth win on
   phone, none on M4 — still worth shipping smaller). Produce artifact sizes + M4 speed + silence-in
   + short BLEU for each. Keep gs32 everywhere.
4. **KV-cache discipline.** Cap `RotatingKVCache` to a realistic live window (e.g. 2–4 min) for the
   mic path; measure memory. Document int8-KV as post-5a work (bandwidth-bound only after the head
   is fixed) — do not implement.
5. **Unified benchmark harness.** Fold `profile_mlx.py` + ad-hoc timing into one
   `scripts/bench.py --model {3b,1b} --quant {q4,q4-depq3}` that emits the per-stage table +
   RT factor + a projected iPhone frame time (scale factors documented in the output).

Output: `reports/inference_matrix.md` — model × quant × strategy table, chosen phone config, and
the measured gap that Track B (Phase 3) must close. Update CONTEXT.md speed section. Commit.

---

## Phase 3 — Track B: parallel codebook head via self-distillation (the architectural fix)

The only change that removes the 64% depformer cost. Per `docs/distill_plan.md` (P1–P5), scoped to
a working end-to-end scaffold + smoke-scale distill on the M4; full-hours training is a later run,
not this phase.

1. **Teacher dump** — `distill/dump_teacher.py`: run the frozen q4 (or bf16) model over source
   audio (reuse downloaded CoVoST2 fr wavs), cache per frame `(transformer_out, text_token,
   16 cb teacher logits)` in resumable shards. Start with ~2–5 h of audio.
2. **Parallel head module** — `distill/parallel_head.py` (MLX `nn.Module`): delay-pattern design
   first (MusicGen-style), `num_passes` knob 1–4; init from the AR depformer's embeddings/linears
   where shapes allow.
3. **Trainer** — `distill/train_head.py`: KL(student‖teacher logits) + CE on teacher tokens,
   `mlx.nn.value_and_grad` on head params only, Adam, val split, checkpointing. Smoke: loss falls,
   overfit a tiny shard to near-zero.
4. **Integration** — swap the head into the vendored `moshi_mlx` generate step behind a config
   flag (`depformer: ar | parallel`), re-quantize, run gates: silence-in, `verify_mlx_q4`,
   per-frame speed (expect depformer 15.7 → ~1–4 ms), short BLEU vs teacher.
5. Record the passes-vs-quality curve at smoke scale in `reports/parallel_head_smoke.md` and the
   exact command list to scale hours later.

Keep cb16 (frozen-main invariant — see distill_plan §6). Commit.

---

## Phase 4 — Training/finetune stack refactor & optimization strategies

**Deliverable:** `finetune/` cleaned into a small, device-portable (MPS + CUDA) training toolkit
implementing the queued optimization hypotheses from CONTEXT.md, ready for 128/512/full-1449
FLEURS runs and larger CUDA runs.

1. **Consolidate.** Dedupe shared logic across `train_lora.py` / `eval_lora.py` / `validate_lora.py`
   / `autoresearch.py` into `finetune/common.py` (model build, adapter load/save, cache dataset,
   metrics). Delete dead experiment branches; keep the autoresearch TSV protocol.
2. **Schedules** (the current best hypotheses, all CLI-driven, no new frameworks):
   - text/audio loss-weight **schedules** (e.g. text-w 5→2 late) instead of static weights;
   - replay weight/anchor-count **schedules** (staged 300→100 generalization of `--replay-*`);
   - per-group LR schedules: transformer-LoRA vs `text_linear` vs audio-head LoRA.
3. **Selection & speed:** periodic eval during training + best-checkpoint selection on val chrF;
   batched greedy eval (`--batch-size 8` path) for val128; make MPS-specific cache-clearing/sync
   conditional on device so CUDA runs are clean.
4. **Gate discipline:** stage gates as CONTEXT prescribes — seen3/short16 for smoke only,
   val128 for real decisions. Add `val128` manifest.
5. Smoke-verify: one short run per new flag on MPS (`--dtype bfloat16 --batch-size 2
   --grad-accum-steps 2`), assert resume + adapter round-trip still work.

Update `docs/finetune.md` + CONTEXT.md finetune section (compress the run-history narrative into
current-state + protocol). Commit.

---

## Phase 5 — iPhone deployment readiness

**Deliverable:** exportable phone artifact + compatibility proof + honest budget report.

1. **Artifact:** the chosen phone config from Phase 2/3 (expected: Hibiki-M q4-gs32 [+ depformer-q3,
   + parallel head when trained at scale]) staged as a standalone HF-style repo dir with config,
   weights, tokenizer, model card.
2. **moshi-swift compatibility check:** a script that validates the artifact the way moshi-swift's
   loader does (gs32 q4 naming/shapes, config keys, tokenizer presence) without needing a device.
3. **Budget report** `reports/iphone_budget.md`: per-stage M4 numbers × documented M4→A18 scale
   assumptions → projected frame time vs the 80 ms budget, for AR head vs parallel head; memory
   footprint; what remains blocking realtime (if anything).
4. Final `README.md` + `CONTEXT.md` refresh. Commit.

---

## Risks

- Deleting `hibiki_zero/` removes the PyTorch serve demo — accepted; MLX path is the product.
  (Finetune uses the `moshi` pip package, verify before delete.)
- Parallel-head quality at smoke scale will be rough — the phase gate is *mechanism works +
  speed measured*, not final BLEU; scaling hours is a follow-up run.
- `mx.compile` on the depformer may win little (launch-bound may persist inside MLX dispatch) —
  time-boxed, keep only if >5%.
- iPhone numbers stay projections until an actual device build (out of scope: no device in loop).

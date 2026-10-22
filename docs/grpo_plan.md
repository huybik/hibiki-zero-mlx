# GRPO preparation plan (Phase 4 analog)

Preparation for Hibiki-Zero style GRPO reinforcement learning on one H100,
following arXiv 2602.11072v2 §3.3 (process rewards) and §4.2.5 (RL protocol),
adapted to this repo's grounded-v2 caches and direct full-model trainer.

Parent plan: `docs/vi_en_parity_plan.md` Phase 4. This doc defines the WHAT;
run phases strictly sequentially, one implementer per phase, commit per phase.

## 1. What the paper does (verified against the v2 HTML)

- Start from the fine-tuned supervised model. RL data = ST training inputs
  (speech X + known **sentence-level** reference translation y). No KL term.
- For each input, π_old generates **G=4** free-running translations, each to a
  **fixed length T** frames (paper: T=1500 @ 12.5 Hz ≈ 120 s). Decoding:
  temperature 0.8, top-k 250, both text and audio streams. No early stop.
- **Process rewards**: split source sentences at frame indexes t_i; at reward
  frame t let y_t be the gold translation of source sentences fully spoken up
  to t. r_t(i) = (1−α)·BLEU(ŷ_t, y_t) + α·BLEU(ŷ_T, y_T), α=0.4.
- Reward frames (t'_j) sit at the **end timestamp of every n_w=8 input words**
  (regular pattern tied to input content beat ad-hoc placement).
- Normalize rewards per reward frame across the group ((r−mean)/std over the
  G samples); advantage at step t = sum of normalized rewards at later reward
  frames: R_t = Σ_{t'_j>t} r̄_(t'_j)  — later steps see more evidence.
- Clipped PPO-style ratio π_θ/π_old per codebook q and frame t, ε=0.2,
  min-form, times R_t. Codebook weights **c_0=100, c_q=1 for q≥1** (text
  stream upweighted 100× vs each audio codebook). Maximize (1/G)ΣΣ c_q L_q.
- π_old is refreshed from π_θ every **τ=20** updates. AdamW LR **2e-7**,
  batch **32**, **2000** updates. Eval every 10·τ; ship best quality/latency.
- Ablations: α is the latency↔quality knob; n_w<8 adds nothing; using the
  FULL reference instead of sentence prefixes improves quality but hurts
  latency (noisy optimistic intermediate scores); **RL cannot teach
  simultaneous behavior to a model supervised with full-sentence delay** —
  it only sharpens what SFT left room for.

## 2. Mapping to this repo

| Paper concept | Repo reality |
| --- | --- |
| Input sentence boundaries t_i | No VI forced alignment. Split `text_vi` on punctuation; map each sentence end to a source frame proportionally by word count over `vi_frames`. Documented approximation. |
| Reward frames every 8 input words | Every 8 Vietnamese words, same proportional mapping onto source frames. |
| Gold sentence prefixes y_t | Progressive sentence-prefixes of `text_en` (PhoMT rows are sentence-aligned pairs). Prefix noise is a known risk (paper Exp. A). |
| Fixed-T rollouts | T = bucketed cap derived from source length + tail (reuse the 280-frame SFT cap logic scaled up; measure VRAM before fixing). `stop_on_eos=False`; PAD/EOS handling exactly as free-running eval. |
| Rollout decoding | Reuse the `common.generate_batch` streaming loop (`training/finetune/common.py`) minus output saving, with `lm_gen` config identical to `training/finetune/eval.py` (temp/top-k flags). Resolve the eval.py text_temp=0.4 default vs paper 0.8 before the run; parity plan §6 pins 0.8/250. |
| Token logprobs for ratios | Rollouts stored as grounded-v2 style code tensors `[1+n_q, T]`; each update re-forwards the batch under grad (bf16 autocast + gradient checkpointing like SFT) to get text + per-codebook audio logprobs. Never backprop through generation. |
| Batch 32 | 32 groups × G=4 = 128 sequences per update is likely beyond one H100 for a 3B multistream model. Micro-batch the ratio/loss computation with grad accumulation to an effective 32; measure real throughput in Phase B before freezing the horizon. |
| Data | Sentence-level rows with reliable references. Start from the frozen 719,120-row train receipt (PhoMT-dominant); FLEURS excluded from RL (reuse already rejected there). A dedicated GRPO manifest freezes membership + order, SHA-pinned like the SFT input manifest (`CachedCodeDataset(sample_manifest=...)`). |
| Init checkpoint | `--init-checkpoint` pointing at a promoted SFT best model (today: `grounded_v2_full_direct_voice_5epoch` best; later: Phase 2/3 checkpoints). Loading path mirrors `common.load_model`. |

Expectation setting: our SFT is fully teacher-forced with no delay curriculum,
so per paper Exp. B this stage mainly sharpens EOS, length control, and
repetition recovery — exactly the step-135k failure modes — not simultaneity.

## 3. Phases

### Phase A — Rollout + reward engine (`training/finetune/grpo_rollout.py`)

1. Batched free-running generation of G samples per input from any checkpoint,
   writing a rollout shard per batch: ids, sampled code tensors `[1+n_q,T]`,
   decoded text per sample, EOS flags, lengths. Deterministic seeding per
   (manifest position, group member).
2. Reward computer: from `text_vi`/`text_en` build source-sentence frame map,
   reward frames every 8 VI words, prefix references from `text_en`;
   r_t(i) per Eq. 3 scored separately for each group member (sentence-level
   sacrebleu, already a dependency via `sacrebleu==2.6.0` in `training/h100.sh`
   setup). One corpus BLEU over the whole group gives every member the same
   reward, so the group-normalized advantages are all zero.
3. Group statistics + advantages per Eq. 5–6 saved alongside.
4. Offline unit checks runnable locally (CPU, tiny fixtures): reward math
   on synthetic texts, frame-map monotonicity, advantage sum ordering.

Exit gate: given a fixture rollout, rewards/advantages match hand-computed
values; generation on a few validation rows reproduces eval.py behavior
(same texts modulo seed) with `stop_on_eos` disabled.

### Phase B — GRPO trainer core (`training/finetune/grpo.py`)

1. Load init checkpoint + rollout shards; recompute logprobs under grad;
   clipped ratio loss with c_0=100/c_q=1 weights; bf16 autocast, fp32 master
   optimizer state, fused AdamW(0.9,0.95), wd 0.1, grad clip 1.0 — matching
   SFT conventions in `training/finetune/train.py`.
2. π_old refresh every τ=20 (deepcopy of weights or stale-parameter buffer);
   assert ratio == 1.0 exactly on the first update after each refresh.
3. Same-run resume contract in the spirit of `train.py`: run_config.json,
   trainer/model checkpoint pairs, atomic saves, keep-N pruning.
4. Receipt + JSONL logs (reward mean/std per frame index, KL-free ratio
   stats, entropy proxy, grad norm, s/step).
5. Smoke mode: tiny manifest, 3 updates, τ=2, asserting refresh semantics and
   finite losses.

Exit gate: smoke passes on H100 (and CPU micro-smoke locally); loss decreases
or stays flat with sane ratio ranges over the smoke window; resume from
step-2 checkpoint reproduces step-3 loss within tolerance.

### Phase C — Launcher, eval, recovery (`training/grpo.sh`, `training/finetune/hf_sync.py`)

1. New dedicated `training/grpo.sh` launcher (not new subcommands inside
   `h100.sh`, which is hard-gated to the frozen grounded-v2 SFT contract):
   same environment conventions (reuse the venv built by
   `./training/h100.sh setup`, identical Torch/moshi pins), plus
   `preflight-grpo` (artifact hashes + init-checkpoint + GRPO manifest pin),
   `rollout`, `train-grpo`, `resume-grpo`; VRAM monitor as in SFT smoke.
   Its smoke marker and commit gate are tracked separately from the SFT
   smoke so the two launchers never invalidate each other.
2. Periodic free-running evaluation every 10·τ updates on the three holdout
   manifests (FLEURS, PhoMT holdout, val128 subset) via the Phase A/B stack:
   BLEU/chrF + the existing health gates in
   `common.generation_health` (`training/finetune/common.py`) as hard promotion
   filters; promote best quality/latency trade-off, never the newest.
3. HF sync under a distinct public prefix (e.g. `grounded_v2_grpo_v1`),
   keeping two complete recovery pairs like the SFT run.

Exit gate: end-to-end dry run — rollout → N updates → eval → promote →
resume — passes on the pod; promotion refuses a checkpoint failing any gate.

### Phase D — Run bookkeeping

Freeze the GRPO data manifest (row-disjoint from all validation manifests),
receipt, and run docs (`docs/finetune.md` section); decide final T, effective
batch, and update budget (start 500–1000) from Phase B/C measurements.

## 4. Risks

| Risk | Mitigation |
| --- | --- |
| VRAM: 128 live sequences × T frames | Grad accumulation + gradient checkpointing; fix T after measuring; shrink group-batch if needed |
| Prefix-reference noise (paper Exp. A) | Keep α=0.4 initially; ablate α only after a stable baseline |
| Proportional VI sentence/word timing is approximate | Unit fixtures bound the error; ablate n_w and prefix scheme later |
| RL destabilizes SFT gains | LR 2e-7, ε=0.2 clip, τ=20, hard health-gate promotion, keep best pre-RL checkpoint as fallback artifact |
| Rollout cost dominates wall-clock | Measure gen RT (eval logs show ~xRT); reuse rollouts across τ-window updates as the paper does |

# GRPO plan

The reinforcement learning stage of [vi_en_parity_plan.md](vi_en_parity_plan.md#4-grpo), on one
H100. It follows the Hibiki-Zero paper ([arXiv 2602.11072v2](https://arxiv.org/abs/2602.11072),
§3.3 and §4.2.5, checked against the HTML of 22 Jul 2026), applied to this repo's streams,
trainer and launcher. Rewritten 2026-11-04. The first plan named a separate launcher and
committed tests, and its prefix reference left out the sentence being spoken.

```text
 every τ=20 updates (π_old = π_θ):
   cached streams ─► bf16 copy of π_θ samples G=4 rollouts each, T frames, no early stop
                 ─► BLEU process rewards at reward frames ─► group-normalized advantages
                 ─► π_θ logprobs of the rollouts via the update's forward (π_old's, exact)
 each update:    32 streams × 4 rollouts ─► clipped ratio loss × advantage ─► AdamW
 every 200:      the SFT validation's free-running sets ─► chrF and end offset ─► promote
```

## 1. The paper's method

- It starts from the supervised model. Its RL data are the inputs of its coarse speech
  translation data: 30–75 s utterances with sentence-level reference translations. There is no
  KL term.
- For each input, π_old samples G = 4 translations, each a fixed T = 1,500 frames (120 s), at
  temperature 0.8 and top-k 250 on text and audio. It does not stop at EOS.
- Process reward of rollout i at frame t (Eq. 3):
  r = (1 − α)·BLEU(ŷ_t, y_t) + α·BLEU(ŷ_T, y_T), with α = 0.4.
  - ŷ_t is the text the model emitted up to t.
  - y_t joins the reference translations of the input sentences up to S(t), the sentence being
    spoken at t, that sentence included. Rewarding the unfinished sentence's whole translation
    is what pushes latency down.
- Reward frames t′_j sit at the end of every n_w = 8 input words. A regular pattern along the
  input beat other placements, and n_w < 8 gained nothing.
- Each reward frame's rewards are normalized over the group: (r − mean) / std over the G
  rollouts.
- The advantage of every token at frame t is R_t = Σ over reward frames t′_j > t of the
  normalized rewards. Tokens after the last reward frame get none.
- The loss is the clipped PPO ratio π_θ / π_old per output stream q and frame t (ε = 0.2), in
  min form, times R_t, summed over t. The streams are text (q = 0) and the 16 audio codebooks,
  weighted c_0 = 100 and c_q = 1. It maximizes (1/G) Σ_i Σ_q c_q L_q^(i).
- π_old is replaced by π_θ every τ = 20 updates.
- Optimization: learning rate 2e-7 (the optimizer is not named), batch 32 inputs (128
  rollouts), 2,000 updates. It evaluates every 10·τ updates and keeps the best quality/latency
  checkpoint; its ablations track BLEU and text LAAL.
- Ablations:
  - α is the latency/quality knob: a higher α gives better quality and more lag.
  - The full reference in place of prefixes improves quality but costs latency.
  - RL cannot teach a model supervised at full-sentence delay to start a sentence early; it only
    sharpens what the SFT left room for. Our SFT delays each row's English by U[0, 0.5 × its
    Vietnamese duration], as the paper's δ = 0.5, so there is that room.

## 2. In this repo

| Paper | Here |
|---|---|
| Inputs, sentences | Streams planned by `streams.StreamDataset` from the noisy `en_speech` train cache, with their own `--seed` and 30–75 s, the paper's lengths (a tenth hold one 5–22 s row, `streams.SINGLE_ROW_STREAMS`, as in SFT). One cache row is one sentence. The stream plan gives each row's `source_start` and `source_frames`. |
| References y, ŷ | Each row's English is its own cache text stream. Both it and a rollout's text are decoded as evaluation decodes (tokens above PAD; a rollout's text ends at its first EOS, so an early EOS loses the rest). No external metadata is needed. |
| S(t), y_t | The rows whose Vietnamese has started by frame t, in order. A gap after a row belongs to that row, as the paper's sentence i runs from t_i to t_{i+1}. |
| Reward frames | Frame boundaries: a reward at boundary b counts the text of the frames before b, and credits exactly those tokens (R_t sums the boundaries after t). The cache has no Vietnamese words or timings, so a boundary follows every 2.5 s of Vietnamese speech, counted inside rows only (≈ 8 syllables at the TTS's pace; check the rate on 1,000 rows' `text_vi` once), plus one at T. The paper has no frame after the last input word. We add T so the tail, where the translation finishes, is trained by the final BLEU. |
| BLEU | `sacrebleu.sentence_bleu` (already pinned) per rollout and reward frame, in CPU worker processes while the GPU generates the next batch. A frame whose group has zero std gets advantage 0. |
| Rollouts | Fed from cached Vietnamese codes, so no Mimi encode or decode: the stream's own lead of silence, then its source with its EOS, then silence to T, through `LMGen` as `common.generate_batch` drives it (sized KV cache, `stop_on_eos` off). Store each rollout in the cache layout `[33, T]`: sampled text and English codes, plus the source. |
| π_old logprobs | At each refresh, one pass of π_θ over the window's rollouts through the update's own forward (fp32 masters under bf16 autocast, gradients enabled, the result detached), cut into the update's own micro-batches. The ratio is then exactly 1 at the first update after a refresh. A no-grad pass, or another batch shape, picks other bf16 kernels: on MPS a no-grad pass gave 1.0012 and clipped 6 tokens. Logprobs at the sampling temperature 0.8, top-k ignored. |
| Generation model | A bf16 copy of π_θ made at each refresh, as SFT validation builds its generator. |
| Update | The SFT training forward (`lm_training`'s compiled transformer, step-major depformer) returns per-token logprobs of the 17 output streams. Micro-batches fit the 4,608-frame budget, accumulated to 128 rollouts. AdamW with 8-bit moments (0.9, 0.95), weight decay 0.1, grad clip 1.0, the gradient norm checked as in SFT. |
| Evaluation | Every 10·τ = 200 updates, the SFT validation's free-running pass over its five sets, unchanged. Quality is FLEURS dev chrF. Latency is the median `end_offset_s`, the stand-in for the paper's text LAAL. |
| Promotion | The best FLEURS dev chrF among checkpoints that pass the health gates and whose FLEURS end offset is at most 0.2 s worse than the starting model's (`LATENCY_SLACK_S`, [grpo](agent-memory/grpo.md)). |
| Start | The final SFT model as `--model-weight`, after the real-Vietnamese tranche and the two-way SFT (GRPO is the last stage, user 2026-11-04; [parity plan](vi_en_parity_plan.md#order)): its best validation by FLEURS dev chrF or its final model, whichever scores better on the test sets. The box profile and dry run may use run 2's final model. |

Out of the first version:

- the parity plan's penalty for speech while the source is silent;
- an α sweep;
- GRPO on a distilled student.

Add one only if the rollouts show a need for it.

## 3. Code

- `training/finetune/rewards.py`: pure functions on text and frames:
  - reward frames from a stream plan;
  - prefix references;
  - process rewards;
  - group normalization;
  - advantages per frame.
- `training/finetune/grpo.py`: the loop in the diagram (stream planning, rollouts, old logprobs,
  updates, evaluation, checkpoints, logs).
- `training/finetune/checkpoints.py`: the run directory `train.py` and `grpo.py` share:
  - `CheckpointWriter`;
  - checkpoint pairs, pruning and resume (`open_run`, `load_newest_pair`), and the resume
    contract (`claim_run_config`);
  - `promote_best`;
  - `keep_log_lines`, `append_log`;
  - the atomic writes.
- `training/finetune/common.py`: what else they share: the model and optimizer
  (`load_policy`, `adamw`), the loss kernel (`token_ce`) and the free-running validation
  (`load_eval_rows`, `validate_free_running`).
- `training/finetune/grpo_profile.py`: phases 1 and 2's profiler (generation, rewards,
  old logprobs, update), with the policy and AdamW states resident.
- `training/h100.sh grpo RUN_NAME [ARGS]`: `run`'s claim, preflight, lock, saved arguments and
  `hf_sync`, starting `training.finetune.grpo`. `hf_sync` needs no change, since `grpo.py`
  writes the SFT run's files:
  - `model_stepN`/`trainer_stepN` pairs;
  - `run_config.json` with `val_every` and `save_every`;
  - `train_log.jsonl`, `val_log.jsonl` and `best.json`.
- Saves land on refresh boundaries (`--save-every` a multiple of τ). A resume therefore starts a
  window: it rebuilds π_old from the saved π_θ and draws that window's rollouts again, seeded by
  update index.
- No test files in the repo: rewards are checked by scratch scripts on the Mac's CPU, the rest on
  the box.

## 4. Cost, to be measured

Measured numbers (from [sft-speed](agent-memory/sft-speed.md)):

- **SFT forward and backward:** 14,400 frames/s on the 700 W H100 SXM.
- **Training memory:** the fp32 weights and gradients and AdamW's 8-bit moments take ~31 GB.
- **KV cache:** ~115 KB per row and frame.

Estimates from those, at T = 1,200 frames (75 s of input plus a 21 s tail):

| Per update (128 rollouts) | Estimate |
|---|---|
| Generation, batches of 128 at ~30 ms/step | ~36 s |
| Old logprobs, amortized over τ | ~4 s |
| Update, 153,600 frames forward and backward | ~11 s |
| Rewards on the CPU, overlapped with generation | ~0 s |
| **Total** | **~50 s; 1,000 updates ≈ 14 h ≈ $20; the paper's 2,000 ≈ 28 h** |

- **Generation memory:** the fp32 weights and AdamW's 8-bit moments (18.9 GB), plus the bf16
  copy (5.8 GB), plus a KV cache of 128 × 1,200 frames (~18 GB). That fits; 256 rows would not.
- **Gradients:** freed before each refresh, as validation does now.

## 5. Phases (each profiled on the box before the next)

1. **Rewards and rollouts.**
   - Build `rewards.py`, the rollout engine and a profiling entry point.
   - On the box:
     - sample G = 4 rollouts for 32 streams from a start checkpoint;
     - time generation at batch 64, 128 and 192 and T = 1,200 or 1,500, recording milliseconds
       per step, the peak memory and GPU utilization sampled every 0.5 s;
     - time the CPU rewards.
   - Exit: the reward mean rises with the rollouts' BLEU, the advantages are zero-mean per frame,
     a hand-checked stream matches, and the generation batch and T are chosen from the numbers.
2. **Updates.**
   - Build the old-logprob pass, the clipped loss and the τ refresh, and run a smoke test of a few
     updates with τ = 2 on the box.
   - Exit:
     - the ratio is exactly 1 on each refresh's first update;
     - the losses and gradient norm are finite;
     - update throughput is measured per micro-batch size;
     - the GPU stays busy, or the gap is explained.
3. **Run bookkeeping.**
   - Build `checkpoints.py`, `h100.sh grpo`, evaluation and promotion.
   - Dry run on the box: rollouts, 2 windows, evaluation, save, kill, resume, sync.
   - Exit: the resume repeats the window and the Hub holds the pair and logs.
4. **The run.**
   - α = 0.4 and the paper's settings, 2,000 updates. Every evaluation (each 200 updates)
     promotes, so stop the run once two evaluations in a row improve neither FLEURS chrF nor the
     end offset. A run cannot be extended: its stop is part of its saved arguments.
   - Then the real-Vietnamese tranche (parity plan, stage 1.3), which this cannot replace: the
     rollouts are TTS speech.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Memory at T × batch | Measure in phase 1; shrink the generation batch, not T |
| Generation dominates the time | Batch size and the sized KV cache from phase 1; rewards overlap on the CPU |
| Reward frames by speech time are approximate | The paper finds n_w insensitive below 8; check the syllable rate once |
| RL undoes SFT gains | LR 2e-7, ε = 0.2, τ = 20, promotion on gates and FLEURS, the start checkpoint kept |
| Rewards fit the TTS references' style (the paper's train BLEU ~60 against ~30 in evaluation) | Select on FLEURS, real speech, never on rollout BLEU |

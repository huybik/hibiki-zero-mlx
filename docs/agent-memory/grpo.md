# GRPO

Read when touching `finetune/grpo_rollout.py`, `finetune/tests/test_grpo_rollout.py`,
`docs/grpo_plan.md`, or adding `finetune/grpo.py` or `finetune/grpo.sh`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Plan](#plan) · [Phase A](#phase-a) ·
[Known defect](#known-defect)

## Plan

- [docs/grpo_plan.md](../grpo_plan.md) is Phase 4 of
  [docs/vi_en_parity_plan.md](../vi_en_parity_plan.md): paper-style GRPO (arXiv 2602.11072v2)
  on one H100, starting from an SFT checkpoint.
- Paper values used: G=4, α=0.4, reward every 8 source words, ε=0.2, text weight 100,
  π_old refresh every 20 updates, LR 2e-7, batch 32, 2000 updates.
- Phases run strictly in order with one commit each: A is the rollout and reward engine (done);
  B is the trainer core `finetune/grpo.py`; C is the launcher `finetune/grpo.sh`, eval and
  `hf_sync` recovery; D is run bookkeeping. B to D do not exist yet.

## Phase A

`finetune/grpo_rollout.py`:

- Reads raw pair rows (`vi_audio`, `vi_duration_s`, `text_vi`, `text_en`), by default
  `finetune/pairs/validation.jsonl` with `--limit 128`, not the Mimi caches.
- Generates `--group-size` (4) free-running samples per row from the upstream weight or
  `--checkpoint`, using temperature 0.8 and top-k 250 for both text and audio. Generation length
  is the longest input plus `--tail-s` 8 unless `--gen-duration` is set.
- Writes one `rollout_batchNNNNNN.pt` shard per batch (codes, text token ids, texts, EOS flags,
  rewards, advantages) plus `rollouts_manifest.jsonl` with each shard's SHA-256.
- Batch i is seeded with `common.seed_all(seed + i * 1000003)`; the members of one row are
  adjacent.
- No Vietnamese forced alignment exists, so sentence ends and reward frames map to source
  frames proportionally by word count over `vi_frames`. Hypotheses are truncated at frame t by
  the same rule, also over `vi_frames`; the module docstring says generated length instead.
- `finetune/tests/test_grpo_rollout.py` holds the offline CPU tests of the reward maths.

## Known defect

`compute_rewards` scores the whole group with one corpus BLEU per reward frame and gives every
member that same value. Per-frame group normalisation then yields zeros, so every advantage is 0
and GRPO would get no learning signal. The test `test_rewards_match_independent_sacrebleu_expression`
encodes this behaviour. The paper's r_t(i) is per sample, so Phase B needs per-member BLEU first.

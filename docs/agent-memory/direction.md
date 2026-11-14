# Direction: the translation direction as a trained condition

Read when touching `training/finetune/direction.py`, `hibiki_helpers.direction_conditions`,
`train.py --condition-lr` and `--lr 0`, `eval.py --direction`, the `direction` and `direction-table`
box jobs, or a runtime's `--direction` (MLX, Swift, the apps' picker).
Back to [CONTEXT.md](../../CONTEXT.md) · [two-way](two-way.md) · [Runs and tests](#runs-and-tests)

## Runs and tests

The shipping model (GRPO update 1,400) guesses the direction from the voice, and outside speech
fools it. On OneVoice's 100 EN→VI clips ([mlx-inference](mlx-inference.md#model-directories), 6
bits) 18 came out English, 12 of them on all of three more seeds; those 18 average −26 dBFS against
the rest's −21 (before the level stage) and are conversational or voice-assistant English ("no
that's all I needed help with, thank you" → "But Datta Niddedtowwess previously was well related to
Heave."); VI→EN wrote Vietnamese on 1 of 100. The user's own Vietnamese (15 s from the MacBook Pro
mic, file mode) came out Vietnamese on 5 of 5 seeds, heard right ("… bờ hồ … bình minh …"), while
OneVoice VI clips played through the laptop's speakers into the same mic came out English. Banning
non-ASCII text pieces gave English on 1 of 3 seeds, so the model's guess of the source wins over a
text mask. So the direction is a condition the model is trained on and the runtimes set (user,
2026-11-14, "go with A"):

- `training.finetune.direction` gives a checkpoint moshi's `lut` conditioner `direction` (entries
  `vi-en` and `en-vi`, 16 wide, projected to the model's 2,048 by a matrix that starts at zero),
  summed by the fuser into every frame's input, in its config.json, so every loader builds it. At
  zero the model's teacher-forced losses equal the parent's exactly (both validation caches, on the
  box); a projection drawn at N(0, 0.05) takes them from 2.7 to 13: the input is that sensitive.
- `hibiki_helpers.direction_conditions` hands it each row's direction, inside the model's graph:
  a stream's (`Stream.direction`, a batch's `directions`) in training, teacher-forced validation
  and GRPO's updates; an evaluation row's (`common.DIRECTION`, from its audio column) through
  `get_lmgen` in free-running evaluation and GRPO's rollouts. A model without conditioners (the
  upstream model, older checkpoints) gets none.
- `train.py --condition-lr` (1e-4) is the table's peak rate on the same warmup and cosine: at a
  fine-tune's rate it would barely move. GRPO keeps one group, so there the table moves at GRPO's
  rate, in effect frozen. `eval.py --direction` conditions every row on one direction (a model
  without the condition refuses it): the obedience test (flipped, the output language should
  follow it). `cut.py` refuses a config that would drop the parent's table.
- The run, `twoway_direction` (`restart.sh direction`, started 2026-11-14 04:59 UTC): update 1,400
  with the table, both directions' rows, 4,608-frame batches, the model at peak 1e-6 to keep GRPO's
  gains, 12,000 updates validated every 4,000. At 4,000 it was under both GRPO's start and update
  1,400 on every set (FLEURS dev VI→EN 38.78 against 40.74 and 42.74, EN→VI 32.34 against 34.16 and
  38.79; median end offset VI→EN 4.5 s against 2.9 and 1.1; 7-51 empty rows a set against 0-3):
  the SFT rows pull back what GRPO gained. On the CPU (one seed) it turned the user's `vi2` to
  English and, told `en-vi`, to Vietnamese, while clear English told `vi-en` stayed Vietnamese: the
  table tips an ambiguous voice, the audio wins a clear one. The run stopped at its 8,000 save (a
  full disk, [sft-recovery](sft-recovery.md#launcher)); its pair 6,000 resumes it.
- `twoway_direction_table` (`restart.sh direction-table`): the same with `--lr 0`, which freezes the
  model (no gradients or AdamW states for it) and trains the table alone, so GRPO's weights stay
  exactly (0.17 s an update; validation 6,000 and 12,000 tied, 40.15 and 40.08). Against update
  1,400 (2026-11-14, `eval.py` seed 43; the box's `/workspace/evals/`):
  - the 11 test sets: mean chrF 45.48 → 44.62 (VI→EN sets within ±0.7, EN→VI −1.2 to −2.9),
    VI→EN ending 1-2 s later (FLEURS-vi median end offset 0.9 → 2.2 s): the table's shared part
    pulls toward the SFT rows' timing;
  - OneVoice's 100 EN→VI clips written in English: 22 → 6; VI→EN 0 → 0; the user's `vi2` written
    in Vietnamese on 4 → 3 of 5 samples, `vi1` 3 → 2;
  - told the other direction, clear speech still follows the audio (Vietnamese told `en-vi`: 1%
    Vietnamese; English told `vi-en`: 1% English on FLEURS, 17% on OneVoice).
  The table tips an ambiguous voice and never overrides a clear one: in every training row the
  direction agrees with the audio, so nothing teaches the model to trust it over what it hears.
- Guidance at decoding (moshi's `cfg_coef`, the other direction as the negative) does not help:
  at 2 the user's `vi2` came out English on 3 of 5, but OneVoice EN→VI wrote English on 17 (4: 21),
  and flipped clear speech still followed the audio (at 4, invented Vietnamese).
- The before/after test, on the box: OneVoice's 200 clips (`data/eval/onevoice-{en,vi}/test.jsonl`)
  and `data/eval/user/test.jsonl` (`vi1`, `vi2`: the user's voice on the MacBook mic; `clean3`,
  `loop3`: OneVoice VI clips played through the laptop's speakers into it).

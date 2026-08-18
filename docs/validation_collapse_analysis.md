# Validation-collapse analysis

This document analyzes the direct VI-EN run
`grounded_v2_full_direct_voice_5epoch`. The last benchmarked checkpoint is step
135,000; the best deterministic FLEURS teacher-forced validation checkpoint is
step 18,000.

## Conclusion

The run did not fail because of numerical instability or a sudden optimizer
explosion. It entered a clear generalization-divergence regime after step
18,000: rolling training loss continued to fall while loss on the same fixed
FLEURS validation rows rose for thirteen consecutive checkpoints.

An additional row-disjoint PhoMT evaluation confirms that the collapse is
domain-specific. From step 18,000 to step 135,000, loss improved on unseen
PhoMT from 3.6814 to 3.1285 while FLEURS validation worsened from 5.8426 to
7.2782. The underlying cause is a conflict between the training mixture and the
FLEURS selection domain, amplified by the data-reuse structure:

- the optimization mixture is 95% PhoMT and 5% FLEURS;
- validation is 100% held-out FLEURS;
- only 1,392 unique FLEURS training rows are eligible, but 35,956 FLEURS rows
  are selected into each frozen epoch, or about 25.8 appearances per unique row;
- the same frozen training manifest is replayed for five planned epochs.

By step 135,000, approximately three full epochs had completed, so an average
FLEURS training example had been presented about 77 times. The model kept
improving both seen and unseen PhoMT while losing generalization to held-out
FLEURS.

The fixed `1e-6` learning rate contributed by continuing to move the model after
the validation optimum, but the evidence does not support calling it globally
too high: the run was stable and useful through step 18,000. The failure was
training for too long at an undiminished update size against a mismatched,
heavily reused mixture.

Teacher forcing is a second, distinct issue. It does not explain the rising
teacher-forced validation loss, because both training and validation use gold
target history. It does explain why free-running generation is substantially
worse than teacher-forced metrics suggest and why repetition, excessive length,
and missing EOS become more severe.

## Evidence

The evidence comes from the stopped pod's run artifacts:

- `finetune/runs/vi_grounded_v2_full_direct_voice_5epoch/train_log.jsonl`;
- `finetune/runs/vi_grounded_v2_full_direct_voice_5epoch/val_log.jsonl`;
- `finetune/runs/eval_direct_voice/upstream_full128/metrics.json`;
- `finetune/runs/eval_direct_voice/best_step018000_full128/metrics.json`;
- `finetune/runs/eval_direct_voice/terminal_step135000_full128/metrics.json`.

### Training and validation diverged

Training loss is a trailing ten-step mean, while validation loss is a
full-corpus mean, so their absolute levels should not be compared directly.
Their opposing trends are nevertheless unambiguous:

| Step | Epoch position | Rolling train loss | Fixed validation loss |
| ---: | ---: | ---: | ---: |
| 18,000 | 0.40 | 3.6674 | **5.8426** |
| 36,000 | 0.80 | 3.5560 | 6.2577 |
| 54,000 | 1.20 | 3.1687 | 6.3917 |
| 90,000 | 2.00 | 3.0826 | 6.8838 |
| 126,000 | 2.80 | 2.7382 | 7.2530 |
| 135,000 | 3.00 | 2.7357 | 7.2782 |

From step 18,000 to step 135,000, rolling training loss fell about 25%, while
validation loss rose about 25%. The train-validation gap grew from roughly 2.18
to 4.54. That is the signature of memorization, domain overfitting, or negative
transfer—not a noisy validation fluctuation.

The validation measurement itself remained invariant at every checkpoint:

- the same 138 examples;
- 324,706 valid audio tokens;
- 9,903 supervised text tokens;
- maximum raw length 277 frames;
- deterministic length order with `shuffle=False`.

Validation shuffling, row-count changes, and varying sequence lengths therefore
cannot explain the curve.

### Row-disjoint PhoMT validation improved

The eligible PhoMT cache contains 684,232 rows at `max_frames=280`. The frozen
training manifest selects 683,164 unique PhoMT IDs, leaving 1,068 IDs unused by
training. Those 1,068 rows form a deterministic, zero-ID-overlap PhoMT holdout.
A deterministic sample of 1,068 selected training IDs provides a same-sized
seen-data control.

| Model | Unseen PhoMT holdout | Seen PhoMT control | Seen/unseen gap | Unseen content accuracy |
| --- | ---: | ---: | ---: | ---: |
| Upstream | 16.9094 | 16.9230 | -0.0135 | 5.26% |
| Step 18,000 | 3.6814 | 3.6526 | 0.0289 | 73.76% |
| Step 135,000 | **3.1285** | **2.7202** | 0.4083 | **79.06%** |

The step-135,000 unseen PhoMT components were audio 1.9676, text 1.1609, and
content CE 1.2125. All improved from step 18,000, whose corresponding values
were 2.3446, 1.3369, and 1.3963.

The seen-data control improved more quickly than the holdout, and its gap grew
to 0.4083 by step 135,000, so some PhoMT memorization is present. Crucially,
unseen PhoMT still improved by 15.0%. The model therefore did not suffer a
universal validation collapse: it specialized toward PhoMT while regressing on
FLEURS.

The reproducible evaluation artifact is
`finetune/runs/eval_phomt_teacher_forced/metrics.json` on the stopped pod. It
records the training-manifest hash, split ID hashes, batch 8, shuffle false, and
all loss components. This holdout is row-ID-disjoint but has not yet been audited
for duplicate or near-duplicate text/audio across different IDs, so it is a
same-domain diagnostic rather than a sealed final test.

### Content and audio collapsed; padding did not

| Validation component | Step 18,000 | Step 135,000 | Change |
| --- | ---: | ---: | ---: |
| Audio CE | 2.2355 | 2.6765 | +0.4411 |
| Text content CE | 3.7820 | 4.8302 | +1.0482 |
| Prefix-PAD CE | 0.1087 | 0.0305 | -0.0782 |
| Combined text loss | 3.6071 | 4.6017 | +0.9946 |
| Total loss | 5.8426 | 7.2782 | +1.4357 |
| Content accuracy | 42.83% | 40.11% | -2.72 points |
| PAD accuracy | 97.30% | 99.41% | +2.11 points |

About 69% of the total loss increase came from the combined text branch and 31%
from target audio. Prefix-PAD loss improved and has only a 0.05 branch weight,
so the padding objective did not cause the collapse.

The content CE rose much more than content accuracy fell. This means the model
became increasingly confident on at least some wrong validation tokens, which
is consistent with overfitting and calibration drift. See
[loss_function.md](loss_function.md) for the exact loss equations and masks.

### Free-running behavior degraded

All three free-running evaluations used the same 128 correct-source FLEURS rows
and decoding configuration. No shuffled-source condition was used.

| Metric | Upstream | Step 18,000 | Step 135,000 |
| --- | ---: | ---: | ---: |
| BLEU | 0.213 | **0.880** | 0.633 |
| chrF | 14.269 | **17.117** | 16.311 |
| WER | **1.130** | 1.329 | 1.951 |
| EOS found | **125/128** | 124/128 | 115/128 |
| Mean length ratio | **1.056** | 1.283 | 2.049 |
| Repeated-4-gram outputs | **16/128** | 22/128 | 32/128 |
| Overlong outputs | **2/128** | 11/128 | 45/128 |

Step 18,000 improved BLEU and chrF over upstream but already failed the
generation-health gate because of repetition. Step 135,000 was worse on BLEU,
chrF, WER, EOS, repetition, and output length. Early stopping at step 18,000
would have prevented further damage, but it would not by itself have produced a
healthy simultaneous translator.

## Root-cause ranking

### 1. Confirmed: FLEURS generalization divergence after step 18,000

The falling training curve and monotonically rising deterministic validation
curve prove that continued optimization improved the training objective while
damaging held-out performance. Five epochs was not supported by the data. The
useful FLEURS region ended before half of the first nominal epoch.

### 2. Confirmed: PhoMT-FLEURS domain conflict

The frozen receipt contains 683,164 selected PhoMT rows and 35,956 selected
FLEURS rows per epoch. Those FLEURS selections come from only 1,392 eligible
unique rows. The optimizer therefore minimizes approximately

\[
L_{train}=0.95L_{PhoMT}+0.05L_{FLEURS},
\]

while checkpoint selection measures only held-out FLEURS. These objectives are
not equivalent. PhoMT dominates the gradient, while repeated FLEURS training
examples can be memorized without improving unseen FLEURS speech.

The row-disjoint experiment directly demonstrates the conflict: between step
18,000 and step 135,000, unseen PhoMT loss improved by 0.5530 while FLEURS loss
worsened by 1.4357. Repeated use of the small FLEURS pool is still a likely
contributor, but the measured conclusion does not depend on that mechanism:
continued training selects for PhoMT at the expense of FLEURS.

### 3. Contributing factor: fixed LR and excessive duration

`1e-6` was not visibly unstable: there were no non-finite losses, spikes,
batch-size changes, or optimizer failures, and training loss decreased smoothly.
It was nevertheless too large to apply unchanged for another 117,000 steps
after the step-18,000 optimum. With no decay or plateau response, every later
update continued pulling the full model away from the best held-out solution.

The practical diagnosis is therefore not “the initial LR was obviously too
high.” It is “the run had no mechanism to stop or reduce updates after
generalization peaked.”

### 4. Confirmed for generation: teacher-forcing exposure gap

During training, gold English text and target-audio history remain in the input.
During generation, the model must consume its own predictions. One early error
therefore changes later context, allowing errors and repeated phrases to
compound. The terminal checkpoint's length ratio, missing EOS, and repeated
4-grams show this failure directly.

Full target-audio teacher forcing can also let the model rely heavily on the
English target history rather than learning a sufficiently robust mapping from
Vietnamese source audio. Correct-source free-running evaluation must therefore
remain a checkpoint gate even when teacher-forced validation improves.

### 5. Structural objective mismatch: audio CE is not always voice preservation

Audio CE asks the model to reproduce the exact English target recording. In the
FLEURS pairing code, Vietnamese and English recordings are independently chosen
speakers joined by sentence ID. Reproducing the English recording's speaker is
not the same objective as preserving the Vietnamese source speaker.

PhoMT cache metadata also distinguishes cross-lingual timbre-matched rows from
unmatched rows. Applying identical audio CE to unmatched pairs can teach target
speaker conversion rather than source-voice preservation. This mismatch does
not by itself explain why validation rose after step 18,000, but it means audio
CE and FLEURS target-audio validation are not sufficient evidence of
voice-preserving translation.

## Recommended recovery

### Preserve the useful artifact

- Keep step 18,000 as the FLEURS-selected artifact from this run.
- Keep step 135,000 as the best measured PhoMT checkpoint, not as a replacement
  for the FLEURS-selected model.
- Do not resume training from step 135,000.
- Label both as experimental checkpoints, not production models: step 18,000
  still fails the free-running repetition gate, and step 135,000 regresses on
  FLEURS semantics and generation health.

### Replace the five-epoch stopping rule

For the next controlled run:

1. initialize from upstream Hibiki-Zero as before;
2. keep batch 16 and the 280-frame cap;
3. cap the first run at 27,000–36,000 steps instead of five epochs;
4. validate at least every 3,000 steps during the first 30,000 steps;
5. stop after two consecutive teacher-forced validation regressions;
6. keep recovery uploads on the desired 9,000-step cadence.

The shorter validation cadence is for locating the optimum; it does not require
uploading a recovery pair every 3,000 steps.

### Test LR separately

Do not change the data mixture and LR in the same causal experiment. First
repeat the short run to verify the early optimum. Then compare one of:

- `1e-6` followed by decay toward `1e-7` before step 27,000; or
- a fixed `5e-7` run with the same short budget.

Select by held-out and free-running behavior, not by the lowest training loss.
If the lower or decayed LR merely delays the same divergence, data composition
rather than LR is dominant.

### Fix the data boundary

- Freeze the measured 1,068-row PhoMT complement as a permanent same-domain
  validation manifest after auditing cross-ID text/audio duplicates.
- Report PhoMT and FLEURS losses separately. Do not replace FLEURS with PhoMT or
  merge them into one number: PhoMT-only validation would have promoted step
  135,000 and hidden the FLEURS regression.
- Select the primary checkpoint domain according to deployment, while requiring
  the other domain not to regress beyond an explicit tolerance.
- Stop treating 35,956 sampled FLEURS entries as 35,956 unique examples. Log
  unique rows and average reuse explicitly.
- Cap repeated use of the 1,392 FLEURS rows. Add more unique real Vietnamese
  speech instead of solving the domain gap only by resampling those rows.
- Randomize training order between epochs if multiple passes remain necessary;
  validation must remain unshuffled.
- Stratify PhoMT validation by timbre-match status, duration, target delay, and
  alignment score to identify which data segment transfers or interferes.

### Align audio supervision with voice preservation

- Apply target-audio CE only to pairs whose English target is verified to match
  the Vietnamese source speaker or timbre.
- Continue using all semantically valid pairs for English text CE.
- For unmatched pairs, either omit audio CE or create a source-voice-matched
  English target; do not call cross-speaker reconstruction voice preservation.
- Evaluate voice preservation with speaker-embedding similarity between source
  Vietnamese and generated English, alongside English ASR content accuracy and
  audio quality.

This fixes the invariant at the supervision boundary: an example receives
voice-preserving audio loss only when its target actually represents the voice
that should be preserved.

### Make free-running quality a promotion gate

Run deterministic correct-source generation on the same validation subset at
each 9,000-step promotion point. A checkpoint should not be called best unless
it satisfies both teacher-forced and generation criteria:

- nonempty predictions at least 122/128;
- EOS found at least 116/128;
- repeated-4-gram outputs at most 12/128;
- mean output/reference length ratio at most 2.0;
- no material chrF regression;
- acceptable generated audio and speaker similarity.

Neither step 18,000 nor step 135,000 passes all these gates. Teacher-forced loss
should remain a useful diagnostic, but it should no longer be the sole best-model
criterion.

After the data and early-stopping fixes, a small target-history dropout or
scheduled-sampling experiment can test the exposure gap. It should be introduced
as an isolated ablation, not bundled with LR and sampling changes. No shuffled
validation, contrastive loss, ASR replay, post-source transform, or
anti-repetition loss is required for this diagnosis.

## Decision rule for the next run

A successful next run must show all three:

1. FLEURS teacher-forced content and audio losses stop diverging from training;
2. correct-source BLEU/chrF improve without worse EOS, repetition, or length;
3. generated English preserves source-speaker identity on a voice-matched set.

If training loss improves without those outcomes, stop. More epochs are not a
solution to this failure mode.

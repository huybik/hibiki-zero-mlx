# The teacher's next round

Read when planning, launching or judging the 3B teacher's next round: the full two-way SFT on the
new EN→VI data, the experiments that gate it, its decay branches and the GRPO after it.
Back to [CONTEXT.md](../../CONTEXT.md) · [Decision](#decision) · [Order](#order) · [Data](#data) ·
[Recipe](#recipe) · [Judging](#judging) · [Time, disk and box](#time-disk-and-box) ·
[Left out](#left-out)

The experiments are in [exposure-bias](exposure-bias.md#experiments), the data in
[envi-data](envi-data.md), GRPO in [grpo](grpo.md#running), the schedules in
[sft-training](sft-training.md#schedules-and-averaging).

## Decision

Planned 2026-11-23; its gates are run ([Order](#order)), the full SFT is not started (user:
"let's finetune the teacher a bit on new data", then "if not a quick sft, what you recommend?",
"keep it at 4,608 frame update on h100", "ok note everything down for future plan"):

- One full pass of two-way SFT on all of `en_speech` both ways, noisy, and the tranche (the data
  the user chose on 2026-11-21, [envi-data](envi-data.md#decision)), then GRPO. The result
  replaces GRPO update 1,400 only if it beats it on the test sets.
- **From the two-way SFT's final model** (`weights/twoway_final.safetensors`, the shipped GRPO's
  update 0), not from update 1,400 (user, 2026-11-23: "the 3b should be the end of sft one, not
  end of grpo one? because we training from it"). SFT undoes GRPO: trained from 1,400 with the direction
  conditioner (peak 1e-6, 4,608 frames), the model had lost GRPO's gains by update 4,000 (FLEURS
  dev VI→EN 42.7 → 38.8, EN→VI 38.8 → 32.3, VI→EN end offset 1.1 → 4.5 s;
  [two-way](two-way.md#next)). So any SFT needs GRPO again, and from GRPO's own start both stages
  compare with the shipped run (selecting 37.45 at update 0, 40.76 at 1,400).
- What it is for: EN→VI on real and noisy English. In the first two-way run ~80% of EN→VI rows
  heard voice-cloned English, and the shipped model writes English for 18 of OneVoice's 100
  outside English clips, mostly quiet and conversational ([two-way](two-way.md#next)). VI→EN's
  next real gain needs more real Vietnamese (the tranche's scale-up), not this mix.

## Order

```text
1. gates     exp 1 on this mix: SFT 10k × 2 (--sampled-text 0 / 0.1), GRPO 400 each   ~6.5 h
             exp 2 probe: GRPO 200 × 2, anchored at β 0.02 / 0.1                     ~2.5 h
2. full SFT  twoway_final, one pass at a constant rate (0.1 if exp 1 won)            ~36–38 h
3. decays    A from the pair at half the pass, B from the end                         ~5.5 h
4. GRPO      from the better branch, anchored if exp 2 won                            ~8–10 h
5. tests     11 test sets, OneVoice, the user's clips, LAAL                           ~1 h
             ≈ 60 h of a 700 W H100 SXM, ~$120 at $2/h
```

- **Gates run 2026-11-23** ([tables](exposure-bias-runs.md)): scheduled sampling 0.1 ties the control
  (not adopted); the anchored probe favours β 0.02; **the data gate failed as written** on VI→EN FLEURS
  dev (−3.85 chrF at seed 44, −3.66 at 61), on its quiet clips, where the new model depends on the
  input's level and `twoway_final` doesn't. Why the SFT lost that is not established. The full SFT
  waits for the user: try one of [two candidate fixes](exposure-bias-runs.md#the-quiet-clips)
  (deeper gain augmentation in its data, or a louder stage target) before spending ~36–38 h.
- **Experiment 1's control arm is the data gate**: plain SFT on this mix from the shipped run's
  start, then GRPO at its settings, compared with the shipped run's as each run's change from its
  own update 0 (planned 400 updates, run 200, against the shipped 38.00). It reads ~670 h, so it
  catches data that hurts rather than proving the full pass helps: if it hurts, stop before step 2
  ([exposure-bias](exposure-bias.md#experiments)).
- The round's winner becomes the students' teacher in place of update 1,400.

## Data

One pass, every row once each way, every source noisy. A noisy copy keeps 40% of its rows clean,
so no clean `en_speech` folder goes beside it, and a tranche folder never goes beside its
`-noisy` copy ([envi-data](envi-data.md#on-the-hub)).

| Direction | Hub folder | Source speech | Hours |
|---|---|---|---:|
| VI→EN | the tranche's `pho`, `vivoice`, `giga` (`anquachdev/mimi-en-vi-speech-20k`) | real Vietnamese | ~5,975 |
| VI→EN | `en-vi-grounded-v2-noisy/train`, all 1,832 shards (seed 0) | TTS Vietnamese | ~8,900 |
| EN→VI | the tranche's `en-vi/{giga,pho,vivoice}-noisy` (seed 1) | voice-cloned English | ~6,000 |
| EN→VI | `en-vi/en-vi-grounded-v2-noisy/train`, all 1,832 shards (seed 1) | real English | ~10,000 |
| | | | **~30,900** |

- The hours are estimates. The tranche's VI→EN is round 4e's 8,456 h less its 500 `en_speech`
  shards' 2,481 h. `en_speech` is its 9,057 h of Vietnamese and 10,486 h of English, less the
  cache's rejections and the OCR and babble drops ([sft-training](sft-training.md#target-timing)).
- About 7,400 h of the real English is new to the teacher, and no English source was noisy before.
- The directions split ~48 / 52, like the first two-way run's even split.
- Validation: both directions' validation caches; the first two-way run's seven sets
  ([two-way](two-way.md#two-way-run)) plus `en-vi-val-noisy` and `en-vi-val-en-noisy`
  ([sft-evaluation](sft-evaluation.md#evaluation-sets)); build what the box lacks first.

## Recipe

- `train.py`'s defaults: whole 4,608-frame updates at peak 2.25e-6 with 5% warmup, on a 700 W
  H100 SXM (user, 2026-11-23: "keep it at 4,608 frame update on h100"). 8,192 frames don't fit
  80 GB whole (~97 GiB), ran 6.5% slower in slices, and `--micro-frames` is gone. On an H200 they
  fit, saving ~5 h for ~25% more money, but at 3e-6 and 1.78× fewer updates an hour of data,
  which changes the recipe ([sft-speed](sft-speed.md#measurements)).
- `--room-share 0.5` (the bank fetched as `grpo_launch.sh` fetches it): half the streams hear a room
  where nobody speaks, as GRPO's do, so the SFT already teaches quiet in a room
  ([noisy-source](noisy-source.md#rooms)).
- `--schedule constant` for one pass (~445,000 updates), validated every 20,000, saved every
  4,000, `--keep-checkpoints 1`, a new `--seed`; no `--sampled-text` (0.1 tied in experiment 1).
- **Two decay branches** (`--schedule decay`, [sft-training](sft-training.md#schedules-and-averaging)),
  each with a new seed, over ~10% of the updates before it:
  - A from the pair at about half the pass (~222,000; ~22,000 updates, ~2 h);
  - B from the final pair (~45,000 updates, ~3.5 h).

  A asks whether the second half of the data (~15,000 h, ~18 h of the box) paid. Every run so
  far peaked before its end (the first two-way run's last 12% gained nothing), and the answer
  sizes later data runs, the tranche's scale-up included. `--keep-checkpoints 1` prunes A's
  pair: hard-link it into `weights/<run>_step<N>/` as soon as it is written (a watcher on the
  box).
- **Averaging**: `average.py` on the winning branch's last checkpoints, scored beside it.
- **GRPO** from the winning branch's final model at the shipped run's settings (`grpo_launch.sh`),
  evaluated every 200 updates and stopped by the user's rule (two evaluations without a new
  best); `--teacher-weight <its start> --kl-coef 0.02` if it is anchored (the probe's better β, its
  gain not isolated from the code's drift). Its `en-vi` folders hold round 4e's 500 `en_speech`
  shards alone, timed by their `vi-en` copy (an extra shard has no pair and raises), while the SFT's
  `en-vi/en-vi-grounded-v2-noisy/train` holds all 1,832: keep GRPO's folders apart on the box, or point its `--pair-cache-dir` at the SFT's full `vi-en` noisy
  folder; settle it at launch ([envi-data](envi-data.md#the-runs-on-it)).
- When the round starts, `restart.sh two-way` is repointed at these folders, this start and the
  schedule. Its current target (`twoway_sft_4608` on the old data) goes.

## Judging

- **The SFT, by its decay branches** (a constant-rate checkpoint reads lower than a decayed one):
  - EN→VI up on `en-vi-val-en` clean and noisy, and on FLEURS-en;
  - VI→EN FLEURS dev within 1 chrF of the start (the pilot's gate; the first two-way run cost
    1.7–2.0);
  - every set healthy.

  Branch A against B: the 11 test sets (paired bootstrap, seed 43) and OneVoice.
- **Ship** only if the round's GRPO best beats update 1,400 on the 11 test sets (mean 45.48,
  seed 43, paired bootstrap) and writes English for fewer OneVoice EN→VI clips.
- **Every evaluation adds**, recorded in full ([sft-evaluation](sft-evaluation.md#recording-results)):
  - OneVoice EN→VI clips written in English (`data/eval/onevoice-en/test.jsonl`): update 1,400
    wrote 18 of 100 at 6 bits in MLX and 22 in PyTorch at seed 43, so compare on one runtime;
  - the user's Vietnamese clips (`data/eval/user/test.jsonl`; `user5/` holds each five times),
    which update 1,400 took for English;
  - `en-vi-val-en-noisy`, which the teacher has never been scored on;
  - LAAL on FLEURS test (`eval.py --laal`; update 1,400: 2.18 s VI→EN, 2.52 s EN→VI).

## Time, disk and box

| Stage | 700 W H100 SXM |
|---|---:|
| Experiments 1 and 2 | ~9 h |
| Full SFT, one pass | ~36–38 h |
| Decay branches A and B | ~5.5 h |
| GRPO, ~1,400–1,800 updates | ~8–10 h |
| Test sets | ~1 h |
| **Total** | **~60 h, ~$120 at $2/h** |

- The SFT's time is from the pass rule ([sft-speed](sft-speed.md#by-gpu)): a row's stream costs
  about the same both ways, so the pass is twice its VI→EN part, 2 × ~14,900 h at round 4e's
  1.21 h of box per 1,000 h of source. Validating 11 sets every 20,000 updates adds ~2 h to that.
- Disk: the packs ~260 GB (VI→EN tranche 57, EN→VI tranche noisy ~52, each `en_speech` noisy
  folder ~75 at 1,832 shards; scaled from GRPO's packs, [envi-data](envi-data.md#the-runs-on-it)),
  the run's pairs ~81 GiB, branch A's pair 37.5 GB, weights, env and evaluation sets ~40: rent
  ~500 GB. GRPO's own folders, kept apart, add ~62 GB; the SFT's pairs leave first, once on the
  Hub ([grpo](grpo.md#running)).
- The card: 80 GB or more. An RTX PRO 6000 takes ~1.7× the time for about the same money; an
  RTX 5090 can't hold the 3B's SFT ([sft-speed](sft-speed.md#by-gpu)).

## Left out

Weighed 2026-11-23 and not run:

- A quick SFT (~40,000 updates, ~2,700 h, ~3.3 h): experiment 1's control arm does its job.
- Experiment 3, centered advantages on the 3B: z-scored won on the student
  ([heads-on-policy-runs](heads-on-policy-runs.md#probe-results)).
- A peak-rate probe: constant-rate checkpoints compare fairly only after a decay, so each arm
  costs one. A rate too high shows at the first validations as the learning-rate test's knock-back
  did (FLEURS dev −2 chrF at 3e-6 and −3.4 at 1e-5 by 2,500 updates; [plan](../vi_en_parity_plan.md#order)
  item 4).
- The direction conditioner and "repeat" rows: dropped 2026-11-14 ([two-way](two-way.md#next)).
- Scheduled sampling of the audio: not built ([exposure-bias](exposure-bias.md#scheduled-sampling)).
- 8,192-frame updates ([Recipe](#recipe)); a restart from upstream's 3B
  ([exposure-bias](exposure-bias.md#scratch-or-fine-tune)).

# GRPO

Read when touching `training/finetune/grpo.py`, `grpo_profile.py`, `rewards.py` or
`h100.sh grpo`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Design](#design) · [Checks](#checks) · [Running](#running)

The stage after SFT ([plan](../grpo_plan.md)): the paper's GRPO with BLEU process rewards, on
streams of the `--cache-dir` caches. The first run is on the two-way model, both directions
(user, 2026-11-10; [plan](../vi_en_parity_plan.md#4-grpo) lists what that adds: both
directions' evaluation sets, a promotion rule over both, the Vietnamese tokenizer for the
EN→VI rewards). Checkpoints, resume and logs are the SFT run's
([sft-recovery](sft-recovery.md)), and evaluation is SFT validation's free-running pass
([sft-evaluation](sft-evaluation.md#checkpoint-selection)).

## Design

- **A window** is `--tau` (20) updates. Each window:
  1. a bf16 copy of the policy samples `--group` (4) rollouts (each conditioned on its stream's
     direction, as the updates are: [two-way](two-way.md#direction)) for each of the window's
     `--streams` (32) × 20 streams. Updates take the planned streams in a seeded order: planning
     keeps a document's rows together, so in plan order one long recording (~75 streams) would
     fill 2–3 updates. The plan is SFT's, late starts included: a late teacher target, which GRPO
     never uses, still pushes the next source later (median gap after a late row 4.8 s against
     2.2 s in `vi-en`);
  2. CPU workers compute the rewards as each generation batch lands;
  3. the policy scores every rollout, giving π_old's logprobs.
- **A rollout's length** is its stream's plus `--tail-seconds` (20 s, the tail evaluation
  generates past a source), up to a whole 16-frame bucket and at most `--frames` (1,200, the
  longest stream plus that tail): `grpo.rollout_frames`. Its last reward sits there, so text a
  rollout emits later scores nothing, as evaluation would not hear it. Until 2026-11-10 every
  rollout ran `--frames`, 21 s past the longest stream and up to 66 s past the shortest.
  - A window's streams are generated longest first (`sample_window`), each batch to its own
    longest rollout; a row's frames past its own length are -1.
  - The loss keeps what a forward of the rollout's own length scores (`kept_tokens`): a
    codebook delayed 2 frames stops 2 frames early. So a rollout's kept tokens do not depend on
    the rollouts it shares a batch or a micro-batch with.
- **After a rollout's first text EOS**, its text tokens and its audio from the EOS frame on stay
  out of the loss (`kept_tokens`), as SFT streams end. No reward reads them, so training them
  would only add noise; in a single-row stream they are most of the frames. They stay in place as
  inputs: set to -1, they would change the context of the delayed codebooks in the two frames
  before the EOS (shown on a tiny CPU model). `eos_rate` in the train log is the share of
  rollouts that ended.
- **An update** takes its own 128 rollouts in `--micro-batch` forwards, cut within its own rows,
  each as long as its rows' last kept token needs, in 16-frame buckets (`forward_frames`,
  `micro_frames`): the model is causal, so the frames after it change no kept logit. π_old's
  pass cuts them the same way, because another batch shape picks other kernels. Every rollout
  is used by exactly one update. The train log's `rollout_frames` and `forward_frames` are an
  update's mean rollout and forward lengths.
- **Rollout layout:** rollouts replay the stream's cached Vietnamese through `LMGen` with
  `stop_on_eos` off, and are stored in the cache's training layout (`grpo.rollout`).
  - `LMGen` reads the input given at step s as source frame s − 1 (step 0's input is never read).
    It samples text frame s at step s, and audio codebook k's frame s − delay_k; the delays are
    0 for text and codebook 0, 2 for the rest.
  - The text and audio hooks capture every sampled token, including those `step()` never returns.
    The hooks copy each token, because CUDA graphs reuse their output buffers.
- **π_old's logprobs** come from the update's own forward: training mode, gradients enabled, the
  result detached and never backpropagated. A no-grad forward picks other bf16 kernels, so the
  first update's ratio would not be 1.
  - Logprobs are taken at the sampling temperature 0.8; top-k is ignored. They are SFT's loss
    kernel negated (`common.token_ce`, dense over the full logits), so the two trainers share
    one kernel ([sft-training](sft-training.md#objective)).
  - Text is weighted `--text-weight` 100 against 1 for each audio codebook. Masked positions are
    the training forward's own.
- **Rewards** (`rewards.py`, the paper's Eq. 3–6):
  - Rewards sit at frame boundaries. A reward at b scores the text of the frames before b, and
    R_t sums the boundaries after t, so a reward credits exactly the tokens it counted.
  - y_b joins the references of the rows whose Vietnamese started before b, the row being spoken
    included. A row's reference is its cache text stream.
  - Texts are decoded as evaluation decodes (`common.decode_text_batch`): tokens above PAD, and a
    rollout's text ends at its first EOS, so an early EOS loses the rest of the translation.
  - A boundary follows every 31 frames (2.5 s) of Vietnamese speech, plus one at T, because the
    cache has no Vietnamese words.
  - Normalization uses the group's unbiased std plus 1e-4, so a frame where every rollout scores
    the same gets 0.
- **Evaluation** runs at update 0 and every `--eval-every` (200) updates, SFT validation's
  free-running pass on `EVAL_PAIRS`' eight sets, both directions of the two-way model GRPO runs
  on (user, 2026-11-10): FLEURS dev both ways, `en-vi-val` clean and noisy, `en-vi-val-en`. The
  first set of each direction selects (`common.selecting_sets`, by the audio a set is heard in):
  promotion goes to the best mean chrF over them among checkpoints that pass every set's gates
  and whose median end offset on each is at most `LATENCY_SLACK_S` (0.2 s) past update 0's.
  The rewards need nothing per direction: a row's reference is its own text track decoded with
  the run's tokenizer (the start weight's, the Vietnamese swap), Vietnamese for an `en-vi` row.
  - The slack: in the dry run, update 2 (two updates at 2e-7, in effect the start) had the
    better chrF (39.90 against 39.29) but measured 4.6 s against 4.5 s. Without slack it was
    blocked, and a blocked checkpoint is not kept (2 pairs).
  - `val_log.jsonl` records each evaluation's `healthy`, `on_time` and `promoted` (this step is
    now the best).
  - That baseline is stored as `start_end_offset_s` (by selecting set) in `run_config.json`, which
    `hf_sync` uploads when its size changes (adding the key at update 0 does). The logs reach the
    Hub only at the end, so a restore needs this file.
- **Length:** `--updates` defaults to the paper's 2,000. Every evaluation promotes, so a run that
  has stopped improving is simply stopped; its stop cannot be extended.
- **Losing work:** `hf_sync` uploads only evaluation-step pairs, so a lost pod costs up to
  `--eval-every` (200) updates. A preemption on the same disk costs at most a window.
- **Saves** land on window boundaries (`--save-every` 20 updates, and `--updates` a multiple of
  τ). A resume starts its window again with the same seeds, unique to each window and batch
  (32-bit, from `SeedSequence((seed, step, batch))`).
- **Reward workers** are forked when the pool starts, before CUDA and any thread. A fork pool
  starts all its workers at the first submit only from Python 3.11, which GRPO requires.
  - Forked workers also hold the launcher's run lock, and a pool never notices its parent dying.
  - So each worker sets `PR_SET_PDEATHSIG` and dies with the trainer. Otherwise a killed trainer
    would leave workers holding the lock, and every relaunch would exit "already running".

## Checks

On the Mac (MPS, bf16 upstream model, 2026-11-04):

- **Reward math:** on a hand-built stream, an on-time correct rollout beats the same words late,
  which beats one wrong word, which beats silence. Advantages are suffix sums of the normalized
  rewards.
- **Rollout alignment:** `LMGen`'s text logits against the training forward agree on argmax 100%
  at the same frame (max difference 0.20) and 55% at ±1 frame (29); 99.87% of sampled audio
  tokens sit inside the teacher-forced top 250, the rest bf16 near-ties.
- **The loss** against its own grad-enabled old logprobs: ratio 1.000000, nothing clipped, the
  loss −Σ c·A, and a step along the gradient raised Σ c·A·logp (a no-grad old pass: ratio
  1.0012, 6 tokens clipped).

The CUDA profile and dry run are under [Running](#running).

## Running

- **Optimize it heavily before its full run, on the box it runs on** (user, 2026-11-10: "remember
  to heavy optimize grpo when it phase start after sft"). On the 80 GB H100 two levers nearly
  halve an update (measured below):
  - the fp32 policy and AdamW's states wait in pinned host memory while a window generates
    (`HostOffload`: 35 → 5.8 GiB resident, so generation batches of 512 rows fit, not 256);
  - rollouts are as long as their stream plus evaluation's tail, and forwards are cut to their
    last kept token ([A rollout's length](#design)). The user approved moving the last reward
    there (2026-11-10, again 2026-11-13).

  Left: micro-batches by a frame budget (3 rows run out of memory at 1,200 frames, though
  forwards average ~635; ~4% of an update). Compare on the same streams by updates per GPU-hour.
  8-bit AdamW states were measured and rejected ([sft-speed](sft-speed.md#measurements)).
- **The GPU's idle between stages** (2026-11-14, 40 updates at the launcher's sizes from the
  two-way model, each host step timed by a wrapper, `/workspace/idle_prof/` on the box): a steady
  window (20 updates, 406 s) idled ~14 s. ~8 s was a `gc.collect()` after each generation batch
  (1.9 s each: the heap grows past the setup's `gc.freeze()`), ~1 s the one after the window's
  generator is dropped, 1.9 s moving the states (10.8 s only in the second window, pinning AdamW's
  new buffers), ~2 s each batch's warm-up and graph capture, 0.4 s `empty_cache`. A batch's
  collection freed ~2 MB: leaving streaming already frees its KV caches and graphs, and only the
  dropped `LMGen`'s own small cycle is left, so the per-batch collection is gone; the generator's
  (~5.8 GiB, `common.bf16_copy`) stays. Re-measured once (40 updates, the same streams):
  generation's gaps fell 8.4 → 0.7 s a window, but the window's end collection rose 1.1 → 5.2 s
  (each window leaves ~2.6M objects of garbage, which a batch's collection took before), and its
  updates spiked to 8–12 s four times where main's held 6.5–7.1 (automatic collections, now
  that nothing collects during generation?). Net ~4 s a window (1%) at best, inside one run's
  noise: it waits for an A/B of 80+ updates each before it merges.
- **Profile first, on the pod**, with `nvidia-smi` sampled beside it:
  `python -m training.finetune.grpo_profile --cache-dir … --model-weight START --out-dir …`.
  - It times generation at each `--gen-batch`, the rewards serially and in workers, and the
    old-logprob pass and the update at each `--micro-batch`.
  - It holds the fp32 policy and AdamW's states as a run does: in host memory while it
    generates, on the GPU for the update stages.
  - It times each generation size on a second call, after compilation. Each batch's CUDA graph
    capture stays in the time, as in a run.
  - An out-of-memory size is recorded and skipped, and `profile.json` is rewritten after each
    stage.
- **Profiled on the box** (2026-11-13, the 700 W H100 SXM, from the two-way run's final model,
  the caches `grpo_launch.sh` passes, T 1,200). Free-threaded Python 3.14 generates faster here
  than CPython 3.12: 29.0 against 30.9 ms a step at 192 rows.

  | Stage | Measured |
  |---|---|
  | Generation with the states on the GPU, 192 / 256 rows | 29.0 / 32.9 ms a step; 5.52 / 6.47 rollouts/s; peak 66 / 74 GiB; 320 runs out of memory |
  | Generation with them on the host, 256 / 384 / 512 rows | 35.6 / 44.0 / 53.6 ms a step; 5.98 / 7.26 / 7.95 rollouts/s; peak 39 / 56 / 72 GiB |
  | Rewards, 128 groups | 2.3 s serial, 0.47 s in 8 workers |
  | π_old's logprobs, micro-batch 2 | 49,300 frames/s |
  | Update, micro-batch 2 | 15,400 frames/s, peak 68 GiB; 3 runs out of memory |
  | First update's ratio | exactly 1 |

  The profiler's estimate for the first window at its own lengths (640 streams; rollouts average
  819 frames and forwards 635 after the EOS cut; 90% of rollouts end) at `--gen-frames 614400`
  (512 rows of 1,200): ~11.7 s of generation, ~1.6 s of old logprobs and ~5.3 s of update, so
  ~18.6 s an update. The profile overstates the short forwards' speed: in a run (2026-11-13, the
  end-to-end check below) an update took ~20.5 s, with ~11.0 s of generation, 2.4 s of old
  logprobs and 7.0 s of update. That compares with main's full-length rollouts at 34.4 s on the
  same streams. So 800 updates take ~4.9 h and 2,000 ~12 h with the evaluations (~3.7 min each).
  The rollouts' mean final BLEU was 33.5.
- **The cut changes almost no reward** (2026-11-13). The same full-length rollouts were scored
  over 1,200 frames and over their cut length: the final model, GRPO's first 128 streams in its
  seeded order, 4 rollouts each.

  | Direction | Groups | Final BLEU, full / cut | Words past the cut | Same best rollout | Rank correlation |
  |---|---|---|---|---|---|
  | EN→VI | 75 | 26.92 / 26.98 | 0.34% | 100% | 0.989 |
  | VI→EN | 53 | 37.55 / 37.56 | 0.07% | 98% | 0.989 |

  So what the cut drops is the rare late babble, and the groups' order, which the advantages
  follow, holds.
- **End-to-end check** (2026-11-13, on 60447659, no Hub). A 40-update run at the launcher's
  sizes was stopped after its step-20 pair and resumed. Beside it, main ran 20 updates on the
  same streams.
  - The first ratio was exactly 1 at the start and after the resume. No recompile landed
    between π_old's pass and an update.
  - The resumed state equalled the saved pair bit for bit, and every `HostOffload` round trip
    restored all 2,016 tensors exactly.
  - Generation peaked at 72.6 of 79.2 GiB, in a batch of 724 rows of 848 frames. Its peak
    follows ~6.3 + 1.07e-4 × rows × frames GiB, so `--gen-frames 614400` holds ~6 GiB in
    reserve in any window.
  - Host memory: 77 GiB pinned (the writer's copy and `HostOffload`'s) and ~17 GiB more that
    can't be given back, so GRPO's preflight needs 128 GiB of RAM.
  - Against main over the same 20 updates: final BLEU 32.45 against 32.50 and `eos_rate`
    0.922 against 0.920, with no systematic difference in loss or clipping.
  - Each process pays once: ~63–80 s of compilation in its first π_old pass, and 10–19 s each
    to pin the writer's and the offload's buffers.
  - A resumed window draws statistically equivalent rollouts, not identical ones: generation
    isn't bit-reproducible across processes.
  - Evaluation itself moves ~±0.7 chrF between runs of the same weights, so judge a trend over
    several evaluations, not one.
- **Profiled on the H200 box** (2026-11-10, alone on the GPU, round 4's final model): generation
  at 192 / 512 / 640 rows 25.1 / 44.4 / 52.3 ms a step (6.36 / 9.60 / 10.17 rollouts/s, peak
  66 / 107 / 124 GiB); π_old's logprobs 53,100–62,300 and the update 16,200–20,400 frames/s at
  micro-batch 2–6. Then every rollout ran 1,202 steps and every forward 1,200 frames; the H100
  profile above has the cut rollouts and forwards.
- **The start policy** loads as SFT's does (`common.load_policy`), so its text EOS input row
  is reset to PAD's ([sft-training](sft-training.md#trainer)); until 2026-11-10 GRPO kept the
  checkpoint's own row. Rollouts and scoring share the row, and the first ratio is still 1.
- **A failed evaluation promotes nothing**, though its pair still reaches the Hub. The
  non-empty gate allows a tenth of a set's rows since 2026-11-10: at 8 of 180 a two-update dry
  run failed both its evaluations on round 4's silent Common Voice rows alone
  ([sft-evaluation](sft-evaluation.md#checkpoint-selection)).
- **Dry run** (2026-11-06, 4 updates): ratio mean 1.000–1.001, the loss finite, the gradient
  norm before clipping 0.42–0.95 million against SFT's ~8–15.
- **The gradient norm's scale:** the loss sums over every kept token of the update's 128 rollouts
  (Σ_q c_q Σ_t, as the paper writes it), so its norm runs ~10⁵ times SFT's mean loss. Clip 1.0
  rescales it, AdamW's step ignores scale, and a clipped parameter gradient (~2e-5) stays far
  above eps 1e-8, so it changes nothing.
- **The two-way run** (user, 2026-11-13): `grpo_launch.sh weights/twoway_final.safetensors`,
  the two-way SFT's final model (`runs/twoway_sft/model_step287558.safetensors`, hard-linked
  into `weights/`), at the H100 profile's sizes, `--updates 2000`. It is judged at 800 from its
  evaluations (every 200, against update 0 too, since promotion only compares checkpoints), and
  goes on while the selecting chrF still rises beyond that noise. The rate is constant, so the
  longer `--updates` is the same run up to 800. Its caches are real Vietnamese speech to
  English (the tranche) and English to Vietnamese from the tranche's voice-cloned English and
  `en_speech`'s real recordings.
  - **Its evaluations** (`runs/grpo_twoway/val_log.jsonl`; "selecting" is the mean chrF of FLEURS
    dev both ways, "of 8" the mean over all eight sets, "EOS" the share of the eight sets' 1,833
    rows that emit the text EOS; every set passes its gates at every update):

    | Update | Selecting | Of 8 | End offset VI→EN / EN→VI | EOS |
    |---|---|---|---|---|
    | 0 | 37.45 | 39.97 | 2.9 / 3.5 s | 0.83 |
    | 200 | 38.00 | 41.04 | 1.6 / 2.1 s | 0.82 |
    | 400 | 38.89 | 41.80 | 1.3 / 1.8 s | 0.82 |
    | 600 | 39.52 | 42.68 | 1.1 / 1.6 s | 0.79 |
    | 800 | 38.88 | 42.68 | 1.1 / 1.5 s | 0.78 |
    | 1,000 | 39.86 | 43.17 | 1.0 / 1.3 s | 0.75 |
    | 1,200 | 39.97 | 43.25 | 1.0 / 1.3 s | 0.72 |
    | 1,400 | 40.76 | 43.77 | 1.1 / 1.5 s | 0.68 |
    | 1,600 | 40.24 | 44.17 | 0.9 / 1.3 s | 0.70 |
    | 1,800 | 40.14 | 43.80 | 0.8 / 1.2 s | 0.71 |

    800 looked like a plateau; the user had it go on to 1,000 ("maybe try one more 1k before we
    stop it", 2026-11-13), which set a new best. It went on toward 2,000, to stop after two
    evaluations in a row without a new best while the mean of 8 stopped rising too (1,600 was no
    best, but its mean of 8 rose): a set's chrF moves ~±0.7 between evaluations, so
    the +0.1 at 1,200 was within noise, and 1,400's +0.8 (both directions up) is not. 200
    updates take ~70 min with their evaluation. 1,800 met the rule, so the run was stopped there
    (TERM to the launcher, 2026-11-14 06:51 UTC+7); its best is update 1,400.
  - **Its EOS share falls** while the train log's `eos_rate` rises (0.93 at update 1 to 0.96):
    no reward pays for an EOS, and one before a stream's last row loses the rest of its text
    ([rewards](#design)), so on validation's single rows the policy holds it back more often,
    while the last word comes sooner (the end offsets above). The live runtimes never stop at an
    EOS, so for them a missing one costs nothing.
  - **Its test sets** (2026-11-14, `eval.py` seed 43 on the 11 two-way test sets, 7,015 rows):
    the mean chrF goes 41.12 (update 0) → 45.48 (1,400) → 45.68 (1,800). 1,400 beats update 0
    on every set (+2.6 FLEURS-vi test, +4.9 FLEURS-en test, up to +5.9), ends sooner (FLEURS
    end offset 2.7 → 0.9 s VI→EN, 3.5 → 1.4 s EN→VI) and leaves 2 rows empty against 18. 1,800
    against 1,400, paired bootstrap: +0.20 on the mean (95% CI −0.09 to +0.53), VI→EN +0.30
    (−0.06 to +0.68), EN→VI −0.06, FLEURS test −0.15 both ways. **Update 1,400 ships** (and
    is the students' teacher): validation chose it before the test sets were read, and they
    cannot tell the two apart. Outputs in `/workspace/evals/test_{start,grpo1400,grpo1800}/`.
- **The run:** `./training/h100.sh grpo RUN_NAME --cache-dir … --model-weight START [--gen-frames
  … --micro-batch …]` claims, locks, preflights and syncs exactly as `run` does
  ([sft-recovery](sft-recovery.md#launcher)). Its boot-hook line, and a restore, use `grpo` where
  an SFT run's use `run`: either subcommand on the other's run fails at argument parsing.
  - **Start weight:** `--model-weight` loads on every resume, before the pair replaces it, so it
    must outlive the run: keep it in `weights/`, not in an SFT run directory that gets cleared.
  - **Disk:** the preflight needs 130 GiB (93 with `--keep-checkpoints 1`), free plus what the
    run directory holds. An SFT run's pairs on the same disk (~81 GiB) go first, once they are
    on the Hub.
  - **Stopping:** `kill -TERM` the launcher's pid alone and wait for its final sync: it uploads
    the newest pair, the best and the logs, which reach the Hub only then. At update 800 it took
    ~53 min: Xet uploaded ~4.8 GB of the model and ~4.5 GB of the trainer state as new data, at
    ~12 MB/s. A group kill (`kill -- -PGID`, `restart.sh`) kills `hf_sync` too. Nothing
    restarts a crashed run.
  - Relaunching a finished run loads the model and exits 0 (`train.py` exits before loading).
  - No commits or edits to tracked files in the run's worktree while it runs: a resume needs
    `run_id.json`'s commit at HEAD, and the preflight no tracked changes.

  `train_log.jsonl` holds, per update:
  - the loss, the ratio mean and its maximum deviation, and the clip fraction;
  - the gradient norm and the final BLEU;
  - the window's generation, reward-wait and old-logprob seconds.

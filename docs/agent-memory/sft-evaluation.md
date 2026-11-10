# SFT evaluation

Read when touching `training/finetune/eval.py`, `training/finetune/validate.py`,
`training/prep/eval_sets.py`, `data/eval/`, or the validation and promotion in
`training/finetune/train.py`.
Back to [CONTEXT.md](../../CONTEXT.md) · [Checkpoint selection](#checkpoint-selection) ·
[Side tools](#side-tools) · [Evaluation sets](#evaluation-sets)

The trainer itself is in [sft-training](sft-training.md); checkpoint files, resume and their sync
in [sft-recovery](sft-recovery.md#checkpoints-and-recovery).

## Checkpoint selection

- Checkpoints are chosen on free-running translation of real Vietnamese (decided 2026-11-02:
  the last run's teacher-forced loss kept improving while its translations got worse). Each
  validation logs the teacher-forced losses of the cache's validation streams, then translates
  `--eval-pairs` (default FLEURS dev, `en-vi-val/cv27` and `en-vi-val/libriheavy`, so build the
  [evaluation sets](#evaluation-sets) before training; rows need `duration_s`) with `eval.py`'s
  sampling, reseeded per batch from `--seed`.
- Each pairs file is scored as its own set, named `<folder>/<stem>` (`common.score_sets`;
  decided 2026-11-03). One blended score would hide the gap between the training data's own
  domain and real speech, weigh sets by their length (corpus chrF and BLEU) and let one set's
  rows carry another's gates.
- `best_step*.safetensors` is a hard link to the checkpoint whose every set passes the
  generation health gates (`common.generation_health`: nonempty, repeated 4-grams, length ratio)
  with a strictly higher corpus chrF on the first set, FLEURS dev, because the phone hears real
  speech (`best.json`: `step`, `model`, `metric` `free_running_chrf`, `chrf`). A run whose
  checkpoints all fail the gates promotes none. The EOS count is reported, not gated (decided
  2026-11-04): no checkpoint of either stream run reached the 91% it once needed (at best 268 of
  361 FLEURS rows) while chrF rose, so it blocked every promotion. `val_log.jsonl` holds
  `teacher_forced`, the validation's `seconds` and, under `free_running`, each set's metrics per
  step; `eval/stepNNNNNN/predictions.csv` has an `eval_set` column.
- Training's chrF ranks the checkpoints of one run. Training (`common.validate_free_running`,
  which SFT and GRPO share) and `eval.py` both translate through `common.translate_rows`, rows
  sorted by `duration_s`, in bf16 batches of 64 by default, and `eval.py`
  passes every row of every `--pairs` file in one call, so given the trainer's `--eval-pairs` in
  order and its `--seed`, `eval.py` builds the trainer's batches; device numerics still differ.
  A set's score depends on the rows it shares batches with (batches are reseeded by index, and a
  row that never ends generates until its batch does), so FLEURS chrF from mixed batches does not
  compare with FLEURS scored alone: compare checkpoints under one `--pairs` list, seed and batch
  size.

## Side tools

- `eval.py` is the correct-source free-running check on a pair file (`id`, `vi_audio`,
  `text_en`, `duration_s`). Its defaults are `common.GenerationConfig`'s, which training's
  checkpoint selection uses too.
  - `translate_rows` samples every token at temperature 0.8, top-k 250, as the paper does
    (Sec. 4.5).
  - It generates until every row has emitted the text EOS, or 20 s past the batch's longest
    source.
  - Each prediction records `end_offset_s`, the time from the end of the source audio to the
    last generated word; the metrics report their median.
  - Decided 2026-11-03. At text temperature 0.4 the step-18,000 model fell behind, stalled and
    ended 11 of 48 FLEURS rows, against 27 at 0.8 with the same 8 s tail and 33 with a longer
    one. Its EOS came up to 14 s after a row ended, and the 8 s tail cut a third of them
    ([analysis](../analysis/eos_and_lag.md)).
  - On the Mac (48 GB M4 Pro, torch 2.15), `--device mps` can score a checkpoint, though the
    pod's validation now scores every one ([environment](environment.md#storage)). One call over
    the second run's five sets in its order with `--seed 43` (1,081 rows) takes 21.5 min at batch
    64, with a 23 GB peak footprint and swap flat. Measured 2026-11-04:
    - A step is bound by per-op dispatch, not the GPU: 120 ms at batch 16 and 171 at 64, so rows
      per second nearly triple from 16 to 64. Batch 96 adds ~15% for 4 GiB more and pads more.
    - The KV cache sized to the batch's steps (`generate_batch`) took a batch-64 step from 318 to
      202 ms and 27 to 12 GiB (torch 2.13). Greedy text tokens matched the full cache over 150
      steps; 97% of audio tokens did, the rest split off at bf16 near-ties.
    - moshi's `torch.compile` gains nothing on MPS (`NO_TORCH_COMPILE=1` is within noise).
    - PyTorch's MPS cache keeps each finished batch's blocks without reusing them (5 GiB a batch,
      to 50 GB and swap; `gc.collect()` frees nothing): `translate_rows` empties it after each
      batch on MPS, which holds it at 7 GiB (2026-11-03).
    - With `--checkpoint`, `eval.py` and `validate.py` neither need nor read `--model-weight`
      (`common.load_lm`), which the checkpoint overwrites: 3.4 s against 11.0 s on MPS, the same
      tensors (2026-11-04). `load_model` converts an fp32 checkpoint tensor by tensor, so its
      fp32 and bf16 copies are never whole at once (that held 18 GiB).
- `validate.py` reports teacher-forced losses with BF16 weights on the validation streams of one
  cache, planned as `train.py` plans them.
- `common.py` is shared by `train.py`, `validate.py` and `eval.py`.
- `train.py` and `eval.py` take a required `--out-dir`.

## Evaluation sets

`training.prep.eval_sets` (huggingface_hub, pyarrow, sphn, soxr and torch, and for `giga-heldout`
PyAV and num2words, so it runs on the Mac or the pod)
writes `eval.py` pair files under `data/eval/`; they are never trained on
([plan](../vi_en_parity_plan.md#2-evaluation-sets)). FLEURS and MultiMed-ST built and checked
on 2026-11-01, `en-vi-val` on 2026-11-03. Since 2026-11-04 they are also in the dataset repo under
`eval/`, audio included, and a pod downloads them ([finetune](../finetune.md#set-up-a-pod)):

| File | Rows | Hours | Use |
|---|---:|---:|---|
| `fleurs-vi/dev.jsonl` | 361 | 1.19 | checkpoint selection |
| `fleurs-vi/test.jsonl` | 857 | 3.00 | final report |
| `multimed-vi/select.jsonl` | 1,718 | 3.01 | non-regression gate |
| `multimed-vi/final.jsonl` | 1,719 | 3.02 | final report |
| `en-vi-val/cv27.jsonl` | 180 | 0.25 | in-domain check at each validation |
| `en-vi-val/libriheavy.jsonl` | 180 | 0.66 | in-domain check at each validation |
| `en-vi-val-noisy/{cv27,libriheavy}.jsonl` | 360 | 0.91 | the same rows under simulated recording conditions ([noisy-source](noisy-source.md#evaluation)) |
| `giga-heldout/test.jsonl` | 1,000 | 1.07 | the real-Vietnamese tranche's own distribution, held out; not in `eval/` (below) |

- `giga-heldout` (built 2026-11-09) samples, with `random.Random(0)`, 1,000 rows of GigaSpeech 2
  archives 86 and 98 (`ingest.SOURCES["giga"].heldout`; 572 and 428 rows from 366 recordings,
  clips 0.7–17.6 s, median 3.38 s), from the rows `filters.vi_gate` keeps that share no run with
  the training text. Those archives are held out of training whole: voice.py and the cache
  build skip them, and nothing had voiced or cached them when they were chosen.
- Holding out archives does not hold out the text: GigaSpeech 2's recordings repeat stories,
  forecasts and outros across archives, and the same Vietnamese got the same machine English
  (`98-40-23` and the cached `147-317-17` share their reference). Of the 44,038 rows the gate
  keeps, 16,717 (38%) share a run (the cache build's rule, `filters.runs`: 13 syllables, or a
  whole sentence of 9–12) with the transcripts of a doc of any other tranche shard, all of which
  are or will be training, joined in order as the trainer joins a doc's rows into a stream (a
  sentence can span two segments: `98-330-175` is `85-142-95` and `-96`); they are dropped before
  sampling (15,774 when each segment was matched alone). The rule cannot see rows under 9
  syllables (stock phrases, not leaked content), so they make 24% of the set against 14% of the
  gated rows (mean clip 3.84 s against 4.18 s), and 66 of those 239 equal a training segment word
  for word ("xin chào và hẹn gặp lại"), so their references were likely trained on. `en_speech` is not scanned: its Vietnamese
  translates English read speech.
- Each row is one segment, as a cache row is, decoded as the cache build decodes, but written as
  16-bit WAV (29 clips reach full scale) and resampled by `eval.py` with sphn rather than soxr;
  `text_vi` is the corpus transcript (upper case) and `text_en` the tranche's machine English, so
  its scores sit apart from FLEURS's human references. The build reads the tranche repo at a pinned
  commit (`GIGA_HELDOUT_REVISION`), so every pod writes the same rows.
- GigaSpeech 2's terms keep its audio off the public dataset repo (an upload of `data/eval` to its
  `eval/` passes `--exclude "giga-heldout/*"`): build it where it runs, from the
  gated tranche repo with an `HF_TOKEN` granted access (`training.prep.eval_sets --sets
  giga-heldout`: ~5 min on the Mac, most of it reading the text columns of every training shard;
  ~0.9 GB of parquet into the HF cache, 354 MB of 48 kHz WAV). It is not in the cache build's
  overlap check (`filters.EVAL_SETS`): the build drops overlapping rows from the set instead.

- FLEURS `google/fleurs` at `70bb2e8`: `vi_vn` 16 kHz WAV read speech; `text_en` is the `en_us`
  raw text of the same sentence id (every Vietnamese row has one). Several speakers read each
  sentence, so ids are the WAV stems.
- MultiMed-ST `leduckhai/MultiMed-ST` at `d8c3b30`, Vietnamese `corrected.test` (3,437 rows),
  split by a `random.Random(0)` shuffle into `select` and `final`. Every clip is **8 kHz** mono
  Vorbis (the card's 16 kHz is the decode rate): telephone band. Segments are cut mid-phrase
  (one starts with the tail of "bệnh viện") while the English is a clean sentence, so read chrF
  rather than BLEU. Only the test split's English was corrected by annotators; `eval` and
  `train` are raw Gemini drafts and are not used. No conversation ids exist, so the halves may
  share conversations.
- `en-vi-val` samples, with `random.Random(0)`, 180 rows per source from the validation split
  of `anquachdev/common-vi-en` at `2ce8777`: Common Voice dev (173 speakers) and Libriheavy dev
  (68 speakers). Its Vietnamese is the TTS the training data uses, so read against FLEURS it
  separates translation failures from the real-speech gap. `sphn` decodes the Opus clips but
  keeps the encoder's pre-skip (312 samples) and the last packet's padding (168 or 648 samples
  here); the builder drops both, then writes 48 kHz WAV (a few peaks above full scale clip). That
  is the training decode (PyAV, FFmpeg's own Opus decoder) on the Common Voice clips, but not on
  the Libriheavy ones: libopus there sits 2 samples off with differences up to 0.45 (pod,
  2026-11-04; [noisy-source](noisy-source.md#checks)). Scores are comparable across checkpoints
  either way. It cuts at the stored duration, not the last granule: the TTS
  encoder (`tts_en_speech.encode`) counts the pre-skip twice, so granules run 312 samples high. It refuses a clip that is not mono, has an output gain, or
  decodes shorter than its stored duration or a 20 ms packet or more longer. The rows keep
  `speaker` and `source_row`.

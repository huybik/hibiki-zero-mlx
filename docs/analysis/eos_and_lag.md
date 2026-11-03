# EOS and lag in free-running validation (2026-11-03)

The full run `en_speech_streams_epoch1b` found the text EOS on 55, 54 and 49 of 361 FLEURS dev
rows at steps 9,000, 18,000 and 27,000. At step 18,000 the previous run found it on 124 of 128
rows, and upstream on 125 ([validation_collapse_analysis](validation_collapse_analysis.md)).
The rows that ended were complete (20 words for a 19-word reference); the others stopped at
about 11 words and stayed silent.

## Method

The step-18,000 checkpoint, in bf16 on the Mac (MPS), ran `common.generate_batch`'s decode:
`encode_inputs` with `add_input_eos`, the warmup and the streaming `LMGen`. A hook on the text
logits logged P(EOS) at every frame. The audio was sampled at temperature 0.8 and top-k 250, as
validation samples it. Each set had 48 rows:

| Set | Source |
|---|---|
| FLEURS | every 7th dev row by duration (real Vietnamese) |
| cache | held-out TTS rows of `en-vi-grounded-v2/validation`, each fed alone as cached codes, the source EOS and steady silence |
| stream | the same rows joined by `plan_streams` and `assemble` into 30–64 s streams, as training sees them |

## Findings

- The eval feeds the source EOS where training puts it. `add_input_eos` writes it at
  ⌈duration · 12.5⌉ with silence after, and `assemble` writes it right after the last row's
  Vietnamese.
- EOS is all or nothing, not a spread-out probability that a low temperature hides:
  - When the model ends, P(EOS) peaks at 0.9–0.95, 5–38 frames after its last word. In training
    the EOS comes a median 13 frames after the last word.
  - Rows that never end peak at 0.002–0.05.
- Text temperature decides most of it:

| Set, tail | Text T | EOS | chrF | BLEU | Empty | Words / ref |
|---|---:|---:|---:|---:|---:|---:|
| cache, 8 s | 0.4 | 16 | 36.3 | 13.9 | 5 | 0.93 |
| cache, 8 s | 0.8 | 38 | 35.3 | 11.2 | 1 | 1.03 |
| stream, 8 s | 0.4 | 32 | | | 0 | |
| FLEURS, 8 s | 0.4 | 11 | 18.2 | 2.4 | 4 | 0.85 |
| FLEURS, 8 s | 0.8 | 27 | 21.6 | 2.9 | 0 | 1.14 |
| FLEURS, 20–25 s | 0.8 | 33 | 22.0 | 2.9 | 0 | 1.17 |

  None of the runs repeated a 4-gram.
- Text temperature also sets how far behind the model ends.
  - At 0.4, on cache rows:
    - the first word comes 31 frames (2.5 s) after the reference's and the last word 46 frames
      after;
    - every frame is a choice between a word and PAD, and temperature 0.4 sharpens it toward
      PAD, so speech falls behind and stalls.
  - At 0.8:
    - the model still starts 27 frames late;
    - it speaks faster than the reference (3.7 frames a token against 4.3) and ends 20 frames
      after it.
- At 0.8 the EOS lands 2–14 s after a FLEURS row ends (median 6.6 s). The 8 s tail, counted from
  the batch's longest source, cut a third of them.
- TTS against real speech: chrF 35 on the held-out TTS rows, 22 on FLEURS.

## Cause and fix

Most of the gap came from the decode settings:

- **Text temperature.** The repo has evaluated at 0.4 since the inference/training split. The
  paper samples every token at temperature 0.8, top-k 250 (Sec. 4.5), as the checkpoint's
  `lm_gen_config` does. Upstream still ended 125 of 128 rows at 0.4, but this model did not.
- **The 8 s tail.** The paper keeps sampling until the text EOS.

`common.GenerationConfig` now samples text at 0.8 and generates up to 20 s past the batch's
longest source. Each prediction records `end_offset_s`, the time from the end of the source audio
to the last generated word's text token; the free-running metrics report their median.

## Still open

- Latency:
  - the model starts about 2.2 s after the reference at both temperatures;
  - its last word lands a median 5.7 s after a FLEURS row ends;
  - the paper's short-form End Offset is 1.9–2.4 s.

  This is GRPO's target ([grpo_plan](../grpo_plan.md)).
- A fifth of the cache rows and a third of the FLEURS rows still never end; their P(EOS) stays
  near 0.
- The real-speech gap: the [plan](../vi_en_parity_plan.md#risks) lists the noisy re-encode and a
  real-Vietnamese tranche.

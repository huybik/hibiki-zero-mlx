# The slices student on the phone

Read when shipping, running or judging the slices student's phone model
(`release/student_slices_coreml6`): its fused Core ML export, Speech Live, its voice on the phone.
Back to [CONTEXT.md](../../CONTEXT.md) · [Release](#release) · [Speech Live](#speech-live) ·
[Voice](#voice). How it was trained and tested: [slices-on-policy](slices-on-policy.md); the
export and runtime: [lm-npu](lm-npu.md#depformer-slices).

## Release

- The 6-bit finish's snap (`phone/pal6_slices_z4000_snap6.safetensors`), staged in bf16
  (`pal6_snap6_bf16`) and exported by `export_lm_coreml.py --bits 6` with the fused depformer
  (the default since #207): `lm-trunk.mlmodelc` 704 MB, `lm-depformer.mlmodelc` 80 MB (9 slices
  with top-k 250 Gumbel-max sampling, `lm_coreml.depformer_top_k` 250), slice 0's table 49 MB, and
  the 16-codebook Mimi decoder copied from the per-slice export. 1.82 GB in 17 files, about 6 min
  on the M4 Pro (2026-11-28, peak 10.6 GB). On the Mac
  `/Volumes/data/models/hibiki/weights/student_slices14/pal6_snap6_ane6f/` (with the card,
  `README.md`), on the Hub `release/student_slices_coreml6/`. The per-slice export
  `pal6_snap6_ane6/` is what [slices-on-policy](slices-on-policy.md#phone-finish) gated.
- Fused against per-slice through the Swift CLI (`swift_gate.py`, the first 20 FLEURS dev rows
  each way, seed 42, `/Volumes/data/hibiki-zero-evals/mac/slices-gate/*_swift20/`): VI→EN chrF
  49.01 / 49.49, EN→VI 34.69 / 32.20, speaker_sim 0.37 / 0.38 and 0.16 / 0.17, HNR 4.3 / 4.4 and
  6.1 / 5.8 dB, every set healthy: the same model within sampling noise.

## Speech Live

- [onevoice-ai-challenge](https://github.com/anthoai97/onevoice-ai-challenge)'s `apps/ios`
  copies `LM.swift` and `NeuralLM.swift` from `inference/swift` (keeping its read-through of
  the mapped tables at load, now of both tables), counts a `depformer_parallel` folder complete
  with `lm-depformer.mlmodelc` and `lm-depformer-embeddings.bin`, and lists the release in
  Settings › Model.
- iPhone 13 Pro Max, `--file two-turns-vi.wav` (two FLEURS Vietnamese sentences), wired, thermal
  fair or nominal (2026-11-28, two runs; files in
  `/Volumes/data/hibiki-zero-evals/mac/slices-phone/speechlive-2026-11-28/`): 60 and 58 ms a
  frame (p50; p90 67 / 62, max 95 / 84), the Core ML LM all of it, ≤ 0.6 s behind, first words
  3.3–3.4 s into the turn, peak footprint 716–721 MB. The heads release between them: 53 ms
  (p90 60), 4.7 s, 719 MB.

## Voice

The user heard the slices model's voice on the phone as fragile (2026-11-28). The same inputs
through the Swift CLI on the Mac, same model and runtime, at the app's input level (−20 dBFS
peak) and main's (−6), seeds 1 and 2, against the phone's own `translation.wav` (what the app
handed the speaker) and the 3B teacher (`twoway_grpo1400_mlx_q6`, MLX); `voice_ab.py` and its
outputs in `/Volumes/data/hibiki-zero-evals/mac/slices-phone/`. HNR dB per run:

| Input (its own HNR) | Slices, phone | Slices, Mac, app level | Slices, Mac, main level | Heads, Mac (both levels) | Teacher, Mac (both levels) |
|---|---|---|---|---|---|
| FLEURS clip (8.0) | 4.9, 4.6 | 4.5, 4.6 | 4.6, 4.2 | 2.9–4.2 | 3.1, 5.6, 6.7 |
| The user's mic, turn 1 (6.7) | | 4.0, 4.6 | 4.0, 3.2 | 0.8–5.3 | 8.7, 9.9, 8.9 |
| The user's mic, turn 2 (6.3) | 5.7 | 5.3, 4.5 | 6.4, 7.2 | 2.9–6.5 | 10.1, 8.5, 10.0 |

- **The app does not thin the voice:** the phone's output scores as the Mac's with the same
  model, at either level (one or two runs each, sampling spread ~1 dB).
- **The model does:** on the user's own speech the teacher keeps 8.5–10.1 dB and the student
  3.2–7.2. The 1.35B students lose the 3B's clarity; the slices keep more than the heads
  (speaker_sim on the clip 0.35–0.41 against 0.01–0.20; the teacher 0.39–0.51).
- Not measured: the speaker itself and Apple's voice processing on the playback path in
  microphone mode (the `translation.wav` tap sits before both).

What makes it hoarse (the user heard "an old man", 2026-11-28): the user's 7 mic clips (`heads-voice`'s
five, the two turns above), seeds 1 and 2, HNR dB / speaker_sim means; scripts, records and the
files to listen to (`listen/`) in `/Volumes/data/hibiki-zero-evals/mac/slices-voice-fix/`.

- **Not the sampling** (`sweep.py`, the Swift CLI, the per-slice export with host sampling): audio
  temp / top-k 0.8 / 250, 0.6 / 250, 0.4 / 250, 0.4 / 25 and 0.2 / 25 score 4.2–4.8 dB, the
  teacher 8.4.
- **Codebooks 8–15, the last slice's, add noise** (`decoder_ab.py`): the same LM runs decoded
  from codebooks 0–7 alone score 3.78 → 4.55 dB in the Swift CLI, higher in 13 of 14 pairs, the
  texts identical and speaker_sim 0.21 either way; in MLX (`codebooks.py`) 5.26 → 6.35 on every
  clip, where the teacher's 8–15 cost nothing (6.09 / 6.15). The Swift runtime takes a decoder of
  fewer codebooks than `dep_q` (`export_mimi_coreml.py --decoder-codebooks 8`):
  `pal6_snap6_ane6f_cb8/` on the Mac, `student_slices_coreml6_cb8` on the phone. The user: clearer,
  not enough.
- **The slices are too small** (`arch_ab.py`, MLX, primed `speaking="vi"`): the slices 3.91 /
  0.226 (at 8 codebooks 4.62 / 0.225), the depformer student (8 sequential slices of 1024 × 3, its
  own 10-layer trunk, [phone-student](phone-student.md)) 5.97 / 0.285, the teacher 7.67 / 0.249,
  above the slices at 8 in 13 of 14 pairs. The depformer student is above them in 7 of 14: on two
  clips the slices fall to noise (−4.5 to 0.7 dB) where it holds 3–8. The user heard it clearer,
  so the slices are retrained wider ([student-slices](student-slices.md#wide-slices)).

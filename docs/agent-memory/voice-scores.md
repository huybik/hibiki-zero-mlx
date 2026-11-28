# Voice scores

Read when touching `training/finetune/voice_scores.py` or `training/prep/kaldi.py`, or the speech decode and scores in
`common.generate_batch` (`hibiki_helpers.decode_wavs`) and `inference/tools/gate.py`.
Back to [CONTEXT.md](../../CONTEXT.md) · evaluation itself: [sft-evaluation](sft-evaluation.md)

Every free-running evaluation that decodes the speech scores it (`training/finetune/voice_scores.py`):
training's validation (`TRAINER_GENERATION` decodes and scores but writes no wav), `eval.py`
(not with `--text-only`; `--no-write-audio` skips the wavs) and the Mac gate. Text-only
evaluation had missed a phone student speaking in a voice unlike the speaker's
([student-heads](student-heads.md#codebooks)).

- Per row `speaker_sim`: cosine of WeSpeaker ResNet34-LM embeddings of the source file as
  recorded (resampled through the translation's rate, so both lose the same band) and the
  translation. Each is cut to its stretches of speech: 25 ms frames within 30 dB of its loudest,
  with gaps under 0.3 s between them kept; the translation's also above −70 dBFS, over Mimi's
  silence (about −74), so a silent translation, whose loudest frame is the decoder's onset click
  (−53 to −60), gets no score (it scored 0.03 / 4.1 dB, near the heads' real clarity), while
  FLEURS-en's quiet recordings (the quietest peaks at −61) keep all their speech. Joined
  loose frames moved the score (two borderline frames flipped by 16-bit rounding took a clip's
  similarity to itself to 0.96); cutting only the ends let the pauses dominate, so every model
  scored alike; a fixed −60 dBFS floor cut most of the speech from FLEURS-en's quiet recordings
  (median peak −48 dBFS; source embeddings at cosine 0.51–0.96 to the full cut). The source is
  not taken through the level stage, which lifts the pauses' noise (a review measured the score
  moving by up to 0.19 a row). A noisy recording's pauses sit within 30 dB and stay: 95% or more
  of the clip is kept on 152 of 392 FLEURS-en and 211 of 361 FLEURS-vi dev sources, so absolute
  similarities are diluted there; every model meets the same source embedding, so a row's
  ranking holds. Audio with a NaN or infinity raises.
- Per row `hnr_db`: the translation's harmonics-to-noise ratio, 10 log10(m / (1 − m)) of the
  mean m over its 40 ms frames within 26 dB of its loudest of each frame's autocorrelation peak
  between the lags of 400 and 60 Hz. Every loud frame counts, voiced or not, so speech turning to
  noise scores low (white noise −7 dB); a steady buzz is periodic and scores high.
- Each set reports `mean_speaker_sim` and `mean_hnr_db`: rows with under a second of speech
  have neither, and rows whose source has under a second no `speaker_sim`. Reported, not gated:
  checkpoints are still chosen on the text.
- WeSpeaker's runtimes (its package, ONNX Runtime) are not in the trainer's free-threaded
  environment, so its ResNet34 and Kaldi filterbank (`training/prep/kaldi.py`, Hamming windows; the voicer's
  speaker encoder takes it with povey windows) are written in PyTorch and load its checkpoint
  (`avg_model`, Hub revision `f0c48c2`, fetched by `h100.sh preflight`). Against WeSpeaker's ONNX
  model and torchaudio's `kaldi.fbank` on the Mac (2026-11-27): filterbanks within 2.6e-4,
  embeddings at cosine 0.999999. WavLM's speaker head (`microsoft/wavlm-base-plus-sv`, in
  transformers) did not separate the models: 0.83 for the teacher and 0.83 for the heads student
  the user heard as unlike them.
- What they read (2026-11-27). On 5 of the user's Vietnamese mic clips (input 5.2 dB): the
  teacher 0.33 / 8.0 dB, the depformer student 0.32 / 6.9, the heads student −0.02–0.16 /
  1.6–4.7 over its checkpoints and sampling variants (the release on the phone 0.16 / 2.7). On the first 5 FLEURS dev rows each way, VI→EN
  then EN→VI, the gate: the teacher (GRPO 1,400, 6 bits) 0.47 / 4.4 dB and 0.29 / 8.1, the
  depformer student (GRPO 2,000, 6 bits) 0.27 / 4.7 and 0.20 / 7.8, the heads release 0.21 / 3.2
  and 0.13 / 4.8; `eval.py` on MPS, the heads student's update 112,000: 0.19 / 3.5 and 0.13 / 5.0.
  Clarity separates the heads less on English speech, which has more unvoiced sounds. Files:
  `/Volumes/data/hibiki-zero-evals/mac/heads-voice/` and `voice-scores/`. On the slices runs'
  seven validation sets (`eval.py`, seed 47) the teacher means 0.389 / 6.50 dB, the slices student
  after Stage 1's decay 0.293 / 4.94 ([student-slices](student-slices.md#runs)).
- Cost: scoring takes ~0.08 s a row on the M4 Pro's GPU; source embeddings are cached for the
  process. The validation also decodes every row's speech up to its end frame (`decode_wavs`):
  on CUDA in length-sorted batches of 8 rows (`hibiki_helpers.MIMI_DECODE_CHUNK_CUDA`), each
  padded only to its group's longest, the same audio as each row alone (Mimi is causal; 8 rows
  of a validation batch 857 → 244 ms on the RTX 5090, [sft-speed](sft-speed.md#fast-path));
  on MPS and the CPU each row alone, to keep the peak small (8 rows of a 52 s batch took 4 GiB of
  the MPS pool and 2.45 s, against 15 GiB and 2.93 s as one chunk).

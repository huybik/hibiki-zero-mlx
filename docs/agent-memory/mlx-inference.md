# MLX inference

Read when touching anything under `inference/` except `inference/swift/`
([swift-runtime](swift-runtime.md)).
Back to [CONTEXT.md](../../CONTEXT.md) · [Entry points](#entry-points) · [Runtime](#runtime) ·
[Model directories](#model-directories) · [Vendored moshi-mlx](#vendored-moshi-mlx) · [Gates](#gates) ·
[Memory](#memory)

## Entry points

- `python inference/main.py INPUT` writes `translations/<stem>_translated.wav` (repo root) and
  a matching `.txt`; `--mic` plays the translation live, capped by `--minutes` (default 30).
- `--model` takes `3b` (`weights/hibiki-zero-3b-q4`, upstream's 3B staged by `convert_mlx.py
  --bits 4` from the files in `weights/`, the symlink to `/Volumes/data/models/hibiki/weights`) or
  any staged model directory. Sampling is the model's `config.json` `lm_gen_config`; the seed,
  299792458, is set after loading, so a file samples as the gate's row of that seed does.
- `inference/macos/HibikiTestApp/` is a SwiftUI wrapper that runs `inference/main.py` (file mode,
  which it can stop, or the mic) with the conda Python path hard-coded. It finds the repo root by
  walking up from its bundle to a directory holding `inference/main.py` and `inference/hibiki_mlx/`,
  and sets `PYTHONPATH` to that checkout's `inference` and `inference/moshi-mlx`: the editable
  installs point at the main checkout, so an app built in a worktree would otherwise run the
  worktree's `hibiki_mlx` on main's `moshi_mlx`. It opens on the model last chosen (user defaults
  `com.huybik.hibiki-test` `modelPath`, written only by a choice, so a model on an unmounted drive
  is kept), else `weights/release/twoway_grpo1400_mlx_q6` (the README's download), and quitting it
  ends its backend. It decodes the backend's output across pipe reads (a read can end inside a
  Vietnamese character, which dropped the whole read before) and reads what is left at exit.
  `build.sh` produces `build/Hibiki Test.app`. Its window can be driven by System Events (buttons
  by index: `button 2 of group 2 of group 1` is Translate, then Stop), once the terminal has
  Accessibility; type a path into the file panel's Go-to field a segment at a time, since its
  completion swallows a `/` typed right after a folder name.

## Runtime

`inference/hibiki_mlx/pipeline.py` owns load and run, plus the repo-root constants `W`
(`weights/`), `UPSTREAM` (`weights/hibiki-zero-3b-q4`) and `SAMPLE` (`inference/assets/leon.wav`).
`pip install -e .` installs `hibiki_mlx` from `inference/`.

- `load()` returns a `Model`: the LM, its config, the SentencePiece tokenizer and the text and
  audio samplers `lm_gen_config` sets. `Model.mimi()` makes a fresh `rustymimi.Tokenizer`: one
  instance cannot be borrowed by two threads, and each stream needs a fresh one, since
  rustymimi's `reset()` leaves state behind (2026-11-14: after `reset()`, the next clip's first
  frame came out different in 14 of 16 codebooks from a fresh instance's).
- `translate()` overlaps a CPU Mimi encoder thread and decoder thread with the GPU LM on the main
  thread. Queues carry NumPy arrays because lazy MLX graphs are bound to their creating thread.
  `audio=False` skips the decoder (the gate scores text only); `run()` writes its WAV and text.
- One frame is 1920 samples at 24 kHz (80 ms, 12.5 Hz).
- The source goes through the input level stage (`hibiki_mlx/level.py`, `Leveller`) before
  Mimi: a file's first channel as one stream, a mic session as another, and `tools/bench.py`'s
  frames (timed in "level + mimi encode"). It raises quiet speech toward −20 dBFS; how and why
  in [sft-evaluation](sft-evaluation.md#input-level). The evaluation (`training/finetune/
  common.py`) imports it as `inference.hibiki_mlx.level`, so the module needs only NumPy.
- `hibiki_mlx` and `moshi_mlx` are installed editable from the main checkout, so in a worktree
  run the tools with `PYTHONPATH=<worktree>/inference:<worktree>/inference/moshi-mlx`: otherwise
  they import main's packages (`main.py` finds its own `hibiki_mlx` first, not `moshi_mlx`). With
  main's older `moshi_mlx` the source EOS was code 2047 and no clip ended.
- File mode feeds a file as training's free-running evaluation does
  (`common.generate_batch`), so its scores compare with the box's: the levelled first channel
  zero-padded to whole frames, then one frame whose 16 source codes are all the source EOS
  (`card`, 2048, `LmConfig.audio_eos_token`; Mimi still codes a silent frame there, as a
  whole-file encode codes the padding), then silence to 20 s after the clip (`TAIL_S`, training's
  `tail_s`), stopping at the first text EOS of a step that heard the source EOS (step t hears
  the frames fed before t). Upstream's own offline inference sends the source EOS the same way.
  - An EOS before that stops nothing, where training's evaluation stops at any EOS and drops
    the rest: on single FLEURS rows it came before the source ended on 6 of 361 VI→EN rows
    (bf16), but on six 66–124 s files of joined FLEURS dev rows (6 bits, two seeds each) on 3 of
    12 runs, both runs of the 124 s file among them, losing up to half the file. Kept going, the
    model translated on to the end and ended right after the source: chrF 31.4 → 50.3,
    43.2 → 47.3 and 40.8 → 51.2 on those runs, the other nine identical (2026-11-14). It is not
    free: past an EOS written a frame or two before the source ended, the model may add a few
    junk words or run the whole tail, and past one mid-file it may go silent for a while (on
    the 124 s file, 30 s in Python, dropping a sentence, and 16 s in Swift, which then caught up).
    Still, on the gate's rows with an early EOS the net chrF rose in all four runs (bf16 and 6
    bits, both ways). So the gate differs from training's
    evaluation only on rows whose EOS came early, and its `eos_found` counts only an EOS after
    the source EOS.
  - `run` (main.py's file mode, so the macOS app) also stops after `QUIET_S` (5 s) without a word
    once the source has ended (`translate(quiet_s=)`; the gate passes none). The shipping model
    seldom writes the EOS on short clips, so file mode ran the 20 s tail on silence: on OneVoice's
    200 clips the stop cuts 27% of the frames (32,756 → 23,814), chrF −0.07 EN→VI and +0.03
    VI→EN. 3 rows lose words: on short clips the model sometimes waits long after the source
    before speaking (EV-0523 waited 9 s and lost its whole text). 10 s lost one row, 11 s none,
    and those cut only 11–18%.
  - Step t hears source frames before t, as training's forward does; moshi's `LMGen` in the
    evaluation hears one silent frame more first, so its outputs come 80 ms later throughout,
    and the gate adds that frame to each End Offset. Mic mode never sends the EOS.
- Sampling is `lm_gen_config`: every training checkpoint's config and upstream's own say
  temperature 0.8, top-k 250 for text and audio, which training's evaluation and upstream's
  inference (`run.py`, `**lm_gen_config`) use. Until 2026-11-14 the runtimes sampled text at 0.4,
  top-k 25, which makes the fine-tuned model lag and seldom end
  ([analysis](../analysis/eos_and_lag.md)); upstream's 3B now samples as upstream does.
- Text comes from `TextDecoder` (both modes): a byte-fallback piece (`<0xE1>`) carries one
  UTF-8 byte, so a character the tokenizer lacks spans frames; the decoder (an incremental UTF-8
  codec with `errors="replace"`) prints it once complete, skips `<unk>`/`</s>`/`<pad>` without
  breaking a sequence, and `flush` ends the stream. For valid UTF-8 its stream equals
  SentencePiece's `decode` of the non-special tokens, as the evaluation decodes
  (`decode_text_batch`; 2026-11-14, scratch check: the repo's ~6,500 lines of docs and
  Vietnamese sentences, on both tokenizers, PADs between pieces); an invalid run of bytes is one
  U+FFFD where SentencePiece writes one per byte. Without byte pieces its text equals the old
  piece-by-piece output. `<s>` is printed as before. Mic mode prints the flushed tail before
  `[stopping]`.
- Mic mode keeps the LM and generator on the main thread; only Mimi encode/decode run in
  workers, which adds one frame of output latency. A worker's error ends the session and is
  raised once it has stopped. Use headphones: the level stage raises the speakers' echo of the
  translation toward speech level ([input level](sft-evaluation.md#input-level)).
- rustymimi fails on long sessions: encoding a 6-minute file raised `narrow invalid args
  start + len > dim_len: [8192, 32]` at frame 2,920 (3.9 min). The Swift runtime's Mimi has no
  such limit ([swift-runtime](swift-runtime.md)).

## Model directories

- `config.json` names the MLX weights (`moshi_name`, required), the sampling (`lm_gen_config`,
  required) and, for quantized weights, `quantization`: `group_size` and `bits` for every Linear
  and Embedding, plus an entry `{"group_size", "bits"}` of its own for each layer quantized
  otherwise, keyed by its module path (mlx-lm's layout). Both runtimes quantize the bf16 model
  by it (`pipeline.quantize`, MLX's per-layer `class_predicate`; Swift `quantize(model:filter:)`
  with a per-layer tuple, mapping its depformer cores to the checkpoint's slice paths), then load
  strictly, shapes included: a record that disagrees with the weights fails to load. Without
  `quantization` the weights are bf16. Older keys (`weight_dtype`, `artifact_format`) are ignored.
- The recipes (`convert_mlx.py --bits {4,6}`, plan stage 6): that many bits on every
  transformer linear (main and depformer layers), group size 32 at 4 bits and 64 at 6; 8 bits,
  group size 32, on the 34 embeddings, the text head (`text_linear`) and each depformer slice's
  `linear_in` and `linear_out` (82 entries). Those are where fewer bits hurt most: q3 on the heads
  made the 3B babble ([inference_matrix](../vision/reports/inference_matrix.md)), and 8 bits
  there matches bf16 heads (teacher-forced argmax agreement with bf16, text 96.7 against 96.5%).
  Quantized from bf16, so scales and biases are bf16. The 3B is 3.28 GB at 6 bits and 2.65 GB at
  4 (uniform q4 was 2.41 GB, BF16 7.7 GB); the 8-bit layers are ~0.48B of its 3.85B parameters
  (with its 16 depformer copies).
- **The two-way model ships at 6 bits** (2026-11-14): on the gate (GRPO update 600, every FLEURS dev
  row, `--seed 42`; bf16 minus the variant with a paired bootstrap's 95% interval), VI→EN bf16 42.57
  chrF, 6 bits 42.15 (0.41, −0.54 to 1.26), 4 bits 38.56 (4.01, 2.90 to 4.99); EN→VI bf16 37.21, 6
  bits 37.50 (−0.29, −1.07 to 0.54). Those ran with file mode stopping at any EOS, as training's
  evaluation does; rerunning the rows with an early EOS under the current rule gives VI→EN 42.76 /
  42.22 and EN→VI 37.34 / 37.66 for bf16 / 6 bits. Upstream's 3B stays at 4 bits: the phone timings
  are at 4, and 6 is unmeasured on its languages.
- **The shipping bundle** is GRPO update 1,400 at 6 bits ([grpo](grpo.md#running)), staged from the
  recovery repo's `grpo_twoway/best/best_step001400.safetensors` with `convert_mlx.py --bits 6` and
  the vi6k tokenizer (3.28 GB; add `mimi-decoder.mlmodelc` for Swift). On the six long files, two
  seeds each (2026-11-14), every run translated to the end, its last word within ~4 s of the
  source's: VI→EN mean chrF 52.6 against update 600's 48.7, EN→VI 46.9 against 45.8. It ends with
  an EOS less often (3 of 12 runs, none on the EN→VI files, against update 600's 9 of 12, as GRPO's
  EOS share fell), so file mode more often runs the whole 20 s tail.
  The recovery repo's `release/twoway_grpo1400_mlx_q6/` holds this bundle ready to run (3.3 GB with
  Mimi and `mimi-decoder.mlmodelc`). On OneVoice (2026-11-14; 100 VI→EN and 100 EN→VI short
  conversational clips, 2–13 s at 16 kHz, from a Google Drive folder the user shared; the gate,
  `--seed 42`): VI→EN chrF 46.3, BLEU 23.6, 72 rows ended; EN→VI chrF 32.2, BLEU 11.7, 79 ended,
  18 rows in English ([two-way](two-way.md#next)), the other 81 at chrF 37.9. 26–28 ms a frame on the
  M4 Pro (~3x real time).
- **The release** (user, 2026-11-14: "upload the full f32 bundle, so he can run it and use it to
  optimize for qualcomm device"): the recovery repo's `release/twoway_grpo1400/` holds update
  1,400 as `model.safetensors` (fp32), a `config.json` naming its files, the vi6k tokenizer, Mimi
  and a model card (results, MLX and moshi loading, porting notes, the licence's research-only use).
  It is the one public place the vi6k tokenizer is; the folder alone stages or loads the model.
- **Where 4 bits loses:** teacher-forced on bf16's sampled tokens (12 FLEURS dev clips both
  ways, 1,788 frames; scratch script), the mean KL from bf16's sampling distribution (logits /
  0.8), text / codebooks: the 4-bit recipe 0.057 / 0.17; 8 bits g64 on the main transformer and
  4 on the depformer 0.0007 / 0.085; 4 on the main and 8 on the depformer 0.057 / 0.096; 5 bits
  g32 0.018 / 0.043; 6 bits g64 0.0055 / 0.018; 8 bits g64 throughout (4.09 GB) 0.0008 / 0.004.
  The main transformer's 4-bit linears carry all of the text's loss; both halves carry the
  codebooks'.
- The tokenizer is the file `tokenizer_name` names (required, no default: the Vietnamese swap
  and the base share `text_card`, so a wrong one loads and prints the wrong words,
  [vi-tokenizer](vi-tokenizer.md#runtimes)); its vocabulary must equal `text_card`. The Mimi
  file is `mimi_name` (default `mimi-pytorch-e351c8d8@125.safetensors`).
- `inference/tools/convert_mlx.py --checkpoint --config --mimi [--tokenizer] [--bits {4,6}]
  --out-dir` stages an exact PyTorch checkpoint (upstream's, or an SFT or GRPO
  `*_step*.safetensors`, fp32 or bf16) into a new directory, quantized or BF16, and refuses to
  overwrite. It ships the tokenizer the checkpoint is on: its safetensors metadata `text_tokenizer`
  (else the base, as in training), from `weights/` or `--tokenizer`, whose file name must match; the
  directory's `config.json` names it. A two-way checkpoint stages with the swap and no flag. The
  12.5 GB fp32 GRPO checkpoint stages in under a minute, with a peak footprint of 6.3–6.6 GB at 4
  bits, 7.5 at 6 and 10.7 in BF16 (2026-11-14).
- Upstream's 3B stages into `weights/hibiki-zero-3b-q4` with `--bits 4` (README). A `weights/` from
  before 2026-11-14 holds `hibiki.q4.safetensors` (uniform q4) beside upstream's PyTorch
  `config.json`, which names no MLX weights or quantization, so it no longer loads: stage it again,
  then delete that file.

## Vendored moshi-mlx

- `inference/moshi-mlx/` is a minimal LM, generator, transformer, KV cache and sampler subset,
  installed with `pip install -e ./inference/moshi-mlx`. It and the root `pyproject.toml` both
  require `mlx>=0.31`.
- Hibiki-Zero deltas: grouped-query attention via `kv_repeat`, FFN width `hidden_scale * dim`,
  `rope_concat` as non-traditional RoPE, and a learned per-slice depformer output LayerNorm.
- The depformer is the only audio head. `load_pytorch_weights` accepts both the upstream fused
  `in_proj_weight` layout and moshi 0.2.13's per-slice `in_projs.N` layout written by SFT
  checkpoints.

## Gates

- `inference/tools/gate.py` (`python -m inference.tools.gate --model DIR --pairs … [--limit N]
  [--seed S] --out-dir DIR`, from the repo root, since it imports `training`) scores a model
  directory as training's free-running evaluation scores a checkpoint, to gate a quantization
  against BF16 (plan stage 6). It translates each row alone in file mode, the MLX seed set to
  `--seed` + the row's index, and scores each pairs file as its own set with training's
  `common.score_sets` (corpus chrF and BLEU, length ratio, empty and ended rows, the health gates).
  It writes `hypotheses.jsonl` (a row each, as it finishes) and `scores.json`. Its scores compare
  with the box's in sampling and input, not in batching: training samples rows 64 at a time. The
  pairs are `data/eval/` files ([sft-evaluation](sft-evaluation.md#evaluation-sets)), their audio
  paths relative to the repo root.
- `inference/tools/verify_mlx_q4.py` translates `inference/assets/leon.wav` (French) with
  `weights/hibiki-zero-3b-q4` into `translations/leon_mlx_q4.wav`.
- `inference/tools/bench.py --model M` reports per-stage milliseconds (Mimi encode, LM, audio head,
  Mimi decode) and a projected iPhone frame time from an assumed `--scale` (0.5). The
  projection is not a measurement. `--silence` gates rms < 0.10 and peak < 1.1.

## Memory

These rules hold for every MLX model here (the Hibiki LM, VieNeu, Mimi, wav2vec2), not only
`inference/`.

- Compile only over a small, fixed set of input shapes. A function under `mx.compile` is traced
  again for each new input shape, and memory grows with every one: MLX wav2vec2-base compiled and
  fed 252 clips of 252 different lengths reached a 61 GB footprint and filled 43 GB of swap
  (2026-10-27); uncompiled, the same run peaked at 3.6 GB. `mx.set_cache_limit` (1 GB in that
  run) does not bound it. Bucket lengths before compiling (`training/prep/mimi_mlx.py` pads to
  multiples of 16 frames: at most 18 shapes, 1.8 GB), or keep the function eager. A model whose
  output changes with padding needs the padding masked first (`training/prep/wav2vec2_mlx.py`
  masks wav2vec2-base's GroupNorm, positional convolution and attention keys).
- Every compiled function in the repo sees fixed shapes: the depformer step, one compile per
  slice (`moshi_mlx/models/lm.py`); the attention regions of a single-token step, with the RoPE
  offset passed as an array (`moshi_mlx/modules/transformer.py`); the samplers; the VieNeu
  frame (`training/prep/vieneu_mlx.py`); the bucketed Mimi encoder (`mimi_mlx.py`) and wav2vec2
  (`wav2vec2_mlx.py`). Hold any new compiled path to the same rule.
- MLX also keeps freed buffers across shapes; cap the cache with `mx.set_cache_limit` when shapes
  vary ([vi-tts](vi-tts.md#engine) caps the codec at 2 GB). Too small a cap costs speed instead:
  the cache builder cycles through ~36 bucketed shapes and ran 195x at 4 GB, 225x at 16 GB
  ([cache-build](cache-build.md#speed-on-the-mac)). Unbucketed shapes (eager wav2vec2 over raw
  lengths) fill any cap with buffers that are never reused.
- PyTorch MPS keeps memory per input shape too: Mimi encode at batch 1 over per-clip lengths
  reached 28–30 GB; 16-frame length buckets held it at 8 GB.
- Watch a run with `top -l 1 -pid <pid> -stats mem` (the physical footprint; RSS hides
  compressed pages) and `sysctl vm.swapusage`. `footprint -p <pid>` reports the same number but
  can hang system-wide, even on a shell (2026-11-02, for over an hour), so a watchdog that calls
  it needs a timeout.

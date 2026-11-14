# Hibiki MLX

Hibiki-Zero speech translation with two maintained paths:

- MLX inference on Apple Silicon (upstream's 3B for FR/ES/PT/DE → English, or any staged
  checkpoint, quantized or BF16);
- full-model PyTorch SFT on one CUDA H100 for Vietnamese → English.

Upstream: [kyutai-labs/hibiki-zero](https://github.com/kyutai-labs/hibiki-zero) ·
[paper](https://arxiv.org/abs/2602.11072) · weights license CC BY-NC-SA 4.0.

## Layout

```text
inference/            Apple Silicon · MLX
  main.py             file and microphone translation CLI
  hibiki_mlx/         pipelined runtime (CPU Mimi codec overlapped with the GPU LM)
  moshi-mlx/          minimal vendored MLX model with the Hibiki-Zero deltas
  tools/              quantized/BF16 staging, the quantization gate, benchmark, Core ML Mimi
  assets/leon.wav     French gate clip
  macos/              native SwiftUI file and microphone test app
training/             CUDA · PyTorch
  h100.sh             pinned H100 setup, preflight, train and resume with HF recovery
  paths.py            repository defaults and file guards
  prep/               en_speech corpus builder, pair rows → grounded Mimi + CTC caches
  finetune/           full-model SFT trainer, evaluation, HF recovery sync
docs/                 plans, analyses and the technical report
weights → …           upstream and staged models (gitignored, shared by both paths)
data/  runs/          caches and training runs (gitignored)
translations/         inference outputs (gitignored)
```

## Inference

```bash
pip install -e ./inference/moshi-mlx
pip install -e .
```

Download the 3B checkpoint and stage it as q4 into `weights/hibiki-zero-3b-q4`, the model
`--model 3b` runs (a `weights/` staged before 2026-11-14, with `hibiki.q4.safetensors` beside
upstream's files, no longer loads: stage it again):

```bash
hf download kyutai/hibiki-zero-3b-pytorch-bf16 \
  config.json \
  "hibiki-pytorch-77f82164@110.safetensors" \
  "mimi-pytorch-e351c8d8@125.safetensors" \
  tokenizer_spm_48k_multi6_2.model \
  --local-dir weights
python inference/tools/convert_mlx.py --bits 4 \
  --checkpoint "weights/hibiki-pytorch-77f82164@110.safetensors" --config weights/config.json \
  --mimi "weights/mimi-pytorch-e351c8d8@125.safetensors" --out-dir weights/hibiki-zero-3b-q4
```

A fine-tuned checkpoint stages the same way into a directory of its own, quantized
(`--bits 4` or `--bits 6`) or, without `--bits`, BF16. `--bits` is the transformer linears' bits
(group size 32 at 4, 64 at 6); the embeddings, the text head and the depformer's input and output
projections take 8 bits, group size 32. The two-way model ships at 6 bits: 3.28 GB, within noise
of BF16's chrF on FLEURS dev both ways, where 4 bits (2.65 GB) loses ~4 on Vietnamese→English;
BF16 is 7.7 GB. The model's `config.json` records the quantization per layer (`quantization`)
and how to sample (`lm_gen_config`).

Run file or microphone translation:

```bash
python inference/main.py inference/assets/leon.wav
python inference/main.py --mic
```

`--model` accepts `3b` or a staged model directory. File mode feeds the source as training's
evaluation does: the clip, the source EOS, then up to 20 s of silence, stopping at the model's
text EOS once it has heard the source EOS (an EOS before that stops nothing, so a long file is
translated to its end), or once the model has said nothing for 5 s after the source ended.
Both modes pass the source through a voice gate (Silero VAD): the model hears silence until
someone speaks and wherever no one does, so noise alone is never translated. Mic mode also
cancels the speakers' echo (WebRTC's echo canceller), so the model does not hear its own
translation and translate it again: the Mac's own speakers need no headphones, but a Bluetooth or
AirPlay speaker may delay its echo past the canceller's ~450 ms reach.
`--speaking vi|en` (both modes) says which language is spoken: the two-way model guesses the
direction from the voice and gets some voices wrong, so the stream then starts with a short
unheard primer in that language, which sets the direction (no accuracy cost measured where the
guess was right).

Build and open the native macOS test app:

```bash
./inference/macos/HibikiTestApp/build.sh
open "inference/macos/HibikiTestApp/build/Hibiki Test.app"
```

The SwiftUI app provides model and audio pickers, the spoken language (Auto, or `--speaking`),
file translation (with stop) and playback, live microphone start/stop, the translated text, and
the backend log. It runs its own checkout's
`inference/main.py` with the conda Python at `/opt/homebrew/Caskroom/miniconda/base/bin/python`,
and opens on the model last chosen, else the shipping two-way bundle where this puts it,
`weights/release/twoway_grpo1400_mlx_q6`:

```bash
hf download huybik/hibiki-zero-vi-full-sft --include "release/twoway_grpo1400_mlx_q6/*" --local-dir weights
```

The inference tools are:

- `inference/tools/convert_mlx.py`: stage an exact PyTorch checkpoint as an MLX model directory,
  quantized (`--bits`) or BF16, with the tokenizer it is on.
- `inference/tools/gate.py`: score a model directory on evaluation pairs through the runtime's
  file mode, as training scores its checkpoints (`python -m inference.tools.gate`, from the repo
  root).
- `inference/tools/verify_mlx_q4.py`: translate the gate clip with `weights/hibiki-zero-3b-q4`.
- `inference/tools/bench.py`: stage timing and silence-input gate.

## Training

Training runs only on a Linux pod with one H100 or H200 of 80 GB or more, through the pinned
CUDA 13.4 launcher:

```bash
./training/h100.sh setup
./training/h100.sh preflight
./training/h100.sh run RUN_NAME --cache-dir data/cache/EN_SPEECH_TRAIN --val-cache-dir data/cache/VALIDATION
./training/h100.sh run RUN_NAME   # continues it, e.g. after a preemption
```

Training modules run from the repository root as `python -m training.<package>.<module>`.
The only trainer is `training.finetune.train`: base-start, full-model SFT with fp32 master
weights and bf16 autocast on grounded caches. Checkpoints are exact full-model states;
partial or adapter checkpoints are rejected.

The training data is the 5–22.4 s clips of the English read-speech corpus `en_speech`
(Common Voice 27.0 EN + Libriheavy, kept in the EN-VI dataset built by `training.prep.publish_en_vi_speech`) plus the
~0.8k-hour PhoMT grounded cache, together ~11.3k hours. See the
[pod guide](docs/finetune.md) and [data preparation](training/prep/README.md).

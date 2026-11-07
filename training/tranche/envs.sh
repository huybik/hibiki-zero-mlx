#!/usr/bin/env bash
# The tranche's envs on the box, at the versions their repos document (researched 2026-11-04): one
# uv env per TTS bake-off candidate (training/tranche/bakeoff.py), one for its scoring, one for the
# production TTS (training/tranche/voice.py) and one for the ASR check (training/tranche/checks.py),
# which also serves voice.py's checks (qwen-asr-serve), and one for the VI→EN translation
# (training/tranche/translate.py). Built without uv's cache, which on /workspace grows by tens of GB.
set -Eeuo pipefail
export UV_NO_CACHE=1
log() { echo "$(date -u '+%F %T') $*"; }
trap 'log "FAILED at line $LINENO: $BASH_COMMAND"' ERR
E=/workspace/envs; mkdir -p $E
UV="nice -n 19 ionice -c3 uv"
mk() { [[ -x $E/$1/bin/python ]] || $UV venv -q -p "$2" $E/$1; }

log voxcpm; mk voxcpm 3.12
$UV pip install -q -p $E/voxcpm voxcpm==2.0.3 soundfile
log nanovox; mk nanovox 3.12
# The production TTS (training/tranche/voice.py): nano-vllm-voxcpm needs flash-attn, whose newest
# prebuilt wheel is for torch 2.10 on CUDA 13.
$UV pip install -q -p $E/nanovox "torch==2.10.*" "torchaudio==2.10.*" --index-url https://download.pytorch.org/whl/cu130
$UV pip install -q -p $E/nanovox nano-vllm-voxcpm==2.0.5 pyarrow av httpx jiwer num2words "torch==2.10.*" "torchaudio==2.10.*" \
  "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
log dots; mk dots 3.12
$UV pip install -q -p $E/dots dots.tts==0.3.1 torch==2.8.0 torchaudio==2.8.0 transformers==4.57.0 soundfile
log qwentts; mk qwentts 3.12
$UV pip install -q -p $E/qwentts qwen-tts==0.1.1 soundfile
log cosyvoice
if [[ ! -d $E/CosyVoice ]]; then
  git clone -q https://github.com/FunAudioLLM/CosyVoice.git $E/CosyVoice
  git -C $E/CosyVoice checkout -q 074ca6d
  git -C $E/CosyVoice submodule update -q --init --recursive
fi
mk cosyvoice 3.10
grep -v -E "^(deepspeed|tensorrt|gradio|fastapi|grpcio|uvicorn|tensorboard)" $E/CosyVoice/requirements.txt > $E/cosyvoice-requirements.txt
# openai-whisper 20231117 builds with pkg_resources, which setuptools 81 dropped.
$UV pip install -q -p $E/cosyvoice "setuptools<81" wheel
$UV pip install -q -p $E/cosyvoice -r $E/cosyvoice-requirements.txt --index-strategy unsafe-best-match \
  --no-build-isolation-package openai-whisper
log score; mk score 3.12
# WeSpeaker calls torchaudio.set_audio_backend, which torchaudio 2.9 removed.
$UV pip install -q -p $E/score torch==2.8.0 torchaudio==2.8.0 transformers==5.22.1 soundfile jiwer onnxruntime \
  "wespeaker @ git+https://github.com/wenet-e2e/wespeaker.git@9fecd6cb4f47475d01761d87c826298dff4ef18c"
log asr; mk asr 3.12
$UV pip install -q -p $E/asr "qwen-asr[vllm]==0.0.6" jiwer
log mt; mk mt 3.12  # training/tranche/translate.py, vLLM in process; qwen-asr pins vLLM 0.14
# It also imports translate_en_speech (anthropic, google-genai) and ingest (av).
$UV pip install -q -p $E/mt vllm==0.33.0 pyarrow anthropic google-genai av==15.1.0
# FlashInfer's prebuilt kernels for the flashinfer and CUDA vLLM brought: without them the first
# run compiles its kernels itself (an FP8 MoE took about an hour).
fi=$($E/mt/bin/python -c "import flashinfer, torch; print(flashinfer.__version__, 'cu' + torch.version.cuda.replace('.', ''))")
$UV pip install -q -p $E/mt "flashinfer-jit-cache==${fi% *}+${fi#* }" --index-url "https://flashinfer.ai/whl/${fi#* }"
for e in voxcpm nanovox dots qwentts cosyvoice score asr mt; do
  echo "$e: $($E/$e/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda)')"
done
du -sh $E; df -h /workspace | tail -1
log "envs done"

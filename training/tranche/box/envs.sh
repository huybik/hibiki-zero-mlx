#!/usr/bin/env bash
# The tranche's envs under /workspace/envs: those named, or all, built side by side, each with the models it loads at
# their pinned revisions. Without uv's cache, which on /workspace grows by tens of GB.
#   ingest  ingest, filters and end_rental.sh (CPU)
#   vieneu  voice.py (VieNeu-TTS v3 Turbo)
#   asr     the Qwen3-ASR server voice.py, ingest and filters hear with
#   cache   the cache builds (restart.sh cache, envi-cache)
#   mt      translate.py
# From the box's copy of the box scripts: /workspace/envs.sh [env ...]
set -Eeuo pipefail
# Builds run side by side and the script fails after all have ended if any failed; an exit before (an interrupt)
# stops those still running, each job's whole process group (set -m below).
trap 'for p in $(jobs -p); do kill -- -$p 2>/dev/null || true; done' EXIT
export UV_NO_CACHE=1
E=/workspace/envs; mkdir -p $E
UV="uv"
mk() { [[ -x $E/$1/bin/python ]] || $UV venv -q -p "$2" $E/$1; }

ingest() {
  mk ingest 3.12
  $UV pip install -q -p $E/ingest av==19.0.2 numpy==2.5.5 pyarrow==26.0.1 scipy==1.18.1 soundfile==0.14.0 httpx==0.28.1 \
    jiwer==4.0.0 num2words==0.5.14 huggingface-hub==0.36.2
  $E/ingest/bin/python -c "import av, numpy, pyarrow, scipy, soundfile, httpx, jiwer, num2words, huggingface_hub; print('ingest ok')"
}

vieneu() {
  # The SDK on torch 2.15 (CUDA 13.4) without its cuda extra, which pins torch 2.8 (CUDA 12.8): its runtime
  # dependencies listed instead. Its ONNX graphs run in PyTorch (onnx_graph.py).
  mk vieneu 3.12
  $UV pip install -q -p $E/vieneu torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
  $UV pip install -q -p $E/vieneu --no-deps vieneu==3.7.1
  $UV pip install -q -p $E/vieneu torch==2.15.0 sea-g2p==0.10.0 onnxruntime==1.32.0 numpy==2.5.5 soundfile==0.14.0 soxr==1.1.0 \
    kaldi-native-fbank==1.23.3 tokenizers==0.22.2 huggingface-hub==0.36.2 pyyaml==6.0.4 librosa==1.0.1 \
    transformers==4.57.6 safetensors==0.8.0 onnx==1.23.2 pyarrow==26.0.1 av==19.0.2 httpx==0.28.1 jiwer==4.0.0 \
    num2words==0.5.14 scipy==1.18.1 --extra-index-url https://download.pytorch.org/whl/cu134 --index-strategy unsafe-best-match
  $E/vieneu/bin/python -c "
import torch
from huggingface_hub import snapshot_download as s
print('vieneu:', torch.__version__, torch.version.cuda)
print(s('pnnbao-ump/VieNeu-TTS-v3-Turbo', revision='61b85e3d937fbbacb387714180e8182823512523',
        allow_patterns=['config.json', 'update/*', 'denoiser.onnx', 'speaker_encoder.onnx']))
print(s('OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano', revision='6aa02b01e445cc585582cf0ba480bc3ea6c8dd68'))
print(s('openai/whisper-large-v3', revision='06f233f', allow_patterns=['*.json', '*.txt']))"
}

asr() {
  mk asr 3.12
  $UV pip install -q -p $E/asr "qwen-asr[vllm]==0.0.6" jiwer  # it pins vLLM 0.14
  $E/asr/bin/python -c "
import torch
from huggingface_hub import snapshot_download as s
print('asr:', torch.__version__, torch.version.cuda)
print(s('Qwen/Qwen3-ASR-1.7B', revision='7278e1e'))"
}

cache() {
  # Python 3.14t with Triton, for Blackwell's fused SEANet, masked GroupNorm and RVQ kernels and torch.compile: the
  # CPU decode and Viterbi run in spawned worker processes, so the GIL Triton turns back on costs little against
  # Triton, compile and FP16's 3-5x (cache-speed-cuda.md). safetensors has no free-threaded wheel: built with Rust.
  command -v cargo >/dev/null || [[ -x ~/.cargo/bin/cargo ]] || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
  export PATH="$HOME/.cargo/bin:$PATH"
  mk cache 3.14t
  $UV pip install -q -p $E/cache torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
  $UV pip install -q -p $E/cache av==19.0.2 einops==0.8.2 huggingface-hub==2.4.0 num2words==0.5.14 numpy==2.5.4 \
    pyarrow==26.0.1 safetensors==0.8.0 sentencepiece==0.2.3 soxr==1.1.0 tqdm==4.70.1 transformers==5.22.1
  $UV pip install -q -p $E/cache --no-deps moshi==0.2.13
  $E/cache/bin/python -c "import torch, triton, transformers, moshi.models.loaders, safetensors, sentencepiece, soxr, av, pyarrow
print('cache:', torch.__version__, torch.version.cuda, 'triton:', triton.__version__)"
}

mt() {
  # vLLM in process; it also imports translate_en_speech (anthropic, google-genai) and ingest (av).
  mk mt 3.12
  $UV pip install -q -p $E/mt vllm==0.33.0 pyarrow anthropic google-genai av==15.1.0
  # FlashInfer's prebuilt kernels for the flashinfer and CUDA vLLM brought: without them the first run compiles its
  # kernels itself (an FP8 MoE took about an hour).
  local fi
  fi=$($E/mt/bin/python -c "import flashinfer, torch; print(flashinfer.__version__, 'cu' + torch.version.cuda.replace('.', ''))")
  $UV pip install -q -p $E/mt "flashinfer-jit-cache==${fi% *}+${fi#* }" --index-url "https://flashinfer.ai/whl/${fi#* }"
  $E/mt/bin/python -c "import torch, vllm; print('mt:', torch.__version__, torch.version.cuda, vllm.__version__)"
}

if (($#)); then envs=("$@"); else envs=(ingest vieneu asr cache mt); fi
set -m
pids=()
for e in "${envs[@]}"; do
  case $e in ingest|vieneu|asr|cache|mt) ;; *) echo "no env $e: ingest, vieneu, asr, cache or mt"; exit 1 ;; esac
  ("$e") & pids+=($!)
done
set +m
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done  # each by its PID: a bare wait returns 0 whatever they returned
((rc == 0))
du -sh $E; df -h /workspace | tail -1
echo envs done

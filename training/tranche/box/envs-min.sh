#!/usr/bin/env bash
# Only the nanovox and asr parts of training/tranche/envs.sh (the bake-off envs are not needed), and the
# cache build's env.
set -Eeuo pipefail
export UV_NO_CACHE=1
E=/workspace/envs; mkdir -p $E
UV="uv"
mk() { [[ -x $E/$1/bin/python ]] || $UV venv -q -p "$2" $E/$1; }
# The tranche's cache build (restart.sh cache, envi-cache) on free-threaded Python: no GIL, so its GPU thread waits on
# none of the threads beside it (cache-build.md#speed-on-cuda). Without Triton, which turns the GIL back on: Mimi's one
# compiled function (its rope) runs eager under NO_TORCH_COMPILE=1. The .venv's versions (h100.sh); safetensors has no
# free-threaded wheel: built with Rust.
(command -v cargo >/dev/null || [[ -x ~/.cargo/bin/cargo ]] || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
 export PATH="$HOME/.cargo/bin:$PATH"
 mk cache 3.14t
 $UV pip install -q -p $E/cache torch==2.15.0 --index-url https://download.pytorch.org/whl/cu134
 $UV pip install -q -p $E/cache av==15.1.0 einops==0.8.2 huggingface-hub==2.4.0 num2words==0.5.14 numpy==2.5.4 \
   pyarrow==26.0.0 safetensors==0.8.0 sentencepiece==0.2.2 soxr==1.1.0 tqdm==4.70.1 transformers==5.22.1
 $UV pip install -q -p $E/cache --no-deps moshi==0.2.13
 $UV pip uninstall -q -p $E/cache triton
 $E/cache/bin/python -c "import sys, torch, transformers, moshi.models.loaders, safetensors, sentencepiece, soxr, av, pyarrow
assert not sys._is_gil_enabled(), 'an import turned the GIL back on'
print('cache:', torch.__version__, torch.version.cuda, 'GIL off')") &
cache=$!
(mk asr 3.12; $UV pip install -q -p $E/asr "qwen-asr[vllm]==0.0.6" jiwer; echo "asr: $($E/asr/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)")") &
asr=$!
mk nanovox 3.12
$UV pip install -q -p $E/nanovox "torch==2.10.*" "torchaudio==2.10.*" --index-url https://download.pytorch.org/whl/cu130
$UV pip install -q -p $E/nanovox nano-vllm-voxcpm==2.0.5 pyarrow av httpx jiwer num2words "torch==2.10.*" "torchaudio==2.10.*" \
  "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
echo "nanovox: $($E/nanovox/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)")"
wait $cache  # each by its PID: a bare wait returns 0 whatever they returned
wait $asr
$E/asr/bin/python -c "
from huggingface_hub import snapshot_download as s
print(s(\"Qwen/Qwen3-ASR-1.7B\", revision=\"7278e1e\"))
print(s(\"openbmb/VoxCPM2\", revision=\"32279ef\"))
print(s(\"openai/whisper-large-v3\", revision=\"06f233f\", allow_patterns=[\"*.json\",\"*.txt\"]))"
echo envs done

#!/usr/bin/env bash
# Only the nanovox and asr parts of training/tranche/envs.sh (the bake-off envs are not needed).
set -Eeuo pipefail
export UV_NO_CACHE=1
E=/workspace/envs; mkdir -p $E
UV="uv"
mk() { [[ -x $E/$1/bin/python ]] || $UV venv -q -p "$2" $E/$1; }
(mk asr 3.12; $UV pip install -q -p $E/asr "qwen-asr[vllm]==0.0.6" jiwer; echo "asr: $($E/asr/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)")") &
mk nanovox 3.12
$UV pip install -q -p $E/nanovox "torch==2.10.*" "torchaudio==2.10.*" --index-url https://download.pytorch.org/whl/cu130
$UV pip install -q -p $E/nanovox nano-vllm-voxcpm==2.0.5 pyarrow av httpx jiwer num2words "torch==2.10.*" "torchaudio==2.10.*" \
  "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
echo "nanovox: $($E/nanovox/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)")"
wait
$E/asr/bin/python -c "
from huggingface_hub import snapshot_download as s
print(s(\"Qwen/Qwen3-ASR-1.7B\", revision=\"7278e1e\"))
print(s(\"openbmb/VoxCPM2\", revision=\"32279ef\"))
print(s(\"openai/whisper-large-v3\", revision=\"06f233f\", allow_patterns=[\"*.json\",\"*.txt\"]))"
echo envs done

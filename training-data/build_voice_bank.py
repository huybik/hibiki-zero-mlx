"""Build a VieNeu voice bank from VIVOS speakers for VI speaker diversity.

Downloads the VIVOS corpus (46 train speakers, 16 kHz read speech, CC BY-NC-SA),
picks one clean reference clip per speaker, enrolls each with VieNeu's
``add_voice`` (denoise + speaker embedding + reference codes), and persists the
bank to ``<PHOMT_DATA_DIR>/voice_bank/vi_voices.json``. ``pipeline.py`` picks the
bank up automatically on the next run.
"""
from __future__ import annotations

import sys
import tarfile
from pathlib import Path

TRAINING_DATA_DIR = Path(__file__).resolve().parent
if str(TRAINING_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DATA_DIR))

from paths import DATASETS_DIR

VIVOS_REPO = "AILAB-VNUHCM/vivos"
REF_MIN_S = 6.0
REF_MAX_S = 12.0
VIVOS_BYTES_PER_S = 32_000  # 16 kHz, 16-bit, mono

VOICE_BANK_DIR = DATASETS_DIR / "voice_bank"
REFS_DIR = VOICE_BANK_DIR / "refs"
BANK_JSON = VOICE_BANK_DIR / "vi_voices.json"


def load_genders(tar: tarfile.TarFile) -> dict[str, str]:
    member = next(m for m in tar.getmembers() if m.name.endswith("train/genders.txt"))
    text = tar.extractfile(member).read().decode("utf-8")
    genders = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            genders[parts[0]] = "male" if parts[1].lower().startswith("m") else "female"
    return genders


def extract_reference_clips(tar_path: Path) -> dict[str, tuple[Path, str]]:
    """Extract one 6-12 s wav per train speaker -> {speaker_id: (wav_path, gender)}."""
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    refs: dict[str, tuple[Path, str]] = {}

    with tarfile.open(tar_path, "r:gz") as tar:
        genders = load_genders(tar)
        print(f"genders.txt: {len(genders)} speakers")
        for member in tar.getmembers():
            if "/train/waves/" not in member.name or not member.name.endswith(".wav"):
                continue
            speaker = Path(member.name).parent.name
            if speaker in refs:
                continue
            duration = (member.size - 44) / VIVOS_BYTES_PER_S
            if not REF_MIN_S <= duration <= REF_MAX_S:
                continue
            out_path = REFS_DIR / f"{speaker}.wav"
            out_path.write_bytes(tar.extractfile(member).read())
            refs[speaker] = (out_path, genders.get(speaker, "unknown"))

    print(f"Extracted {len(refs)} reference clips to {REFS_DIR}")
    return refs


def patch_torchaudio_load() -> None:
    """torchaudio >= 2.10 delegates load() to torchcodec (no torch-2.13 build).

    VIVOS references are plain PCM wavs, so soundfile covers enrollment.
    """
    import soundfile as sf
    import torch
    import torchaudio

    def sf_load(path, *args, **kwargs):
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(wav.T), sr

    torchaudio.load = sf_load


def enroll(refs: dict[str, tuple[Path, str]]) -> None:
    patch_torchaudio_load()
    import pipeline

    tts = pipeline.load_tts(pipeline.VI_TTS)
    for number, (speaker, (wav_path, gender)) in enumerate(sorted(refs.items()), start=1):
        tts.add_voice(
            speaker,
            wav_path,
            gender=gender,
            style="cloned / VIVOS",
            description=f"VIVOS {gender}",
        )
        print(f"[{number}/{len(refs)}] enrolled {speaker} ({gender})", flush=True)
    tts.save_voices(BANK_JSON)
    print(f"Voice bank saved: {BANK_JSON}")


def main() -> None:
    from huggingface_hub import hf_hub_download

    tar_path = Path(
        hf_hub_download(VIVOS_REPO, "data/vivos.tar.gz", repo_type="dataset")
    )
    refs = extract_reference_clips(tar_path)
    enroll(refs)


if __name__ == "__main__":
    main()

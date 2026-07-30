"""Build the VI→EN voice map by speaker-embedding similarity.

One-time offline step: synthesizes calibration audio for every EN voice in the
pipeline pool (Kokoro singles + weighted same-gender blends) and embeds both
sides with VieNeu's 192-d speaker encoder — the VI side from the synthesized
calibration wavs written by qa_vi_voices.py, so the match measures what the
training data actually sounds like, not the raw enrollment clips. Each VI voice
pairs with its most similar same-gender EN candidate by cosine. pipeline.py
derives each row's EN voice from its VI voice through this map.
Rerun after changing the EN pool, the blocklist, or re-running qa_vi_voices.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

TRAINING_DATA_DIR = Path(__file__).resolve().parent
if str(TRAINING_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DATA_DIR))

import pipeline
from build_voice_bank import patch_torchaudio_load
from paths import DATASETS_DIR
from qa_vi_voices import SYNTH_DIR, cosine

# Phonetically rich Harvard sentences; ~10 s per voice is plenty for the encoder.
CALIBRATION_TEXTS = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "These days a chicken leg is a rare dish.",
    "The juice of lemons makes fine punch.",
]

EN_REFS_DIR = DATASETS_DIR / "voice_bank" / "en_refs"


def en_ref_path(voice: str) -> Path:
    return EN_REFS_DIR / (voice.replace(":", "").replace(",", "+") + ".wav")


def synthesize_en_references() -> dict[str, Path]:
    tts = pipeline.load_tts(pipeline.EN_TTS)
    EN_REFS_DIR.mkdir(parents=True, exist_ok=True)
    refs: dict[str, Path] = {}
    for number, voice in enumerate(pipeline.EN_TTS.voices, start=1):
        path = en_ref_path(voice)
        speed = pipeline.get_voice_speed(pipeline.EN_TTS, voice)
        audios = tts.infer_batch(CALIBRATION_TEXTS, voice=voice, speed=speed)
        gap = np.zeros(int(0.3 * tts.sample_rate), dtype=np.float32)
        tts.save(np.concatenate([part for audio in audios for part in (audio, gap)]), str(path))
        refs[voice] = path
        print(f"[{number}/{len(pipeline.EN_TTS.voices)}] synthesized {voice}", flush=True)
    return refs


def main() -> None:
    en_refs = synthesize_en_references()

    patch_torchaudio_load()
    vi = pipeline.load_tts(pipeline.VI_TTS)
    embed = lambda path: vi.engine.prepare_reference(
        str(path), denoise=False, use_ref_codes=False
    )[0]
    en_embs = {voice: embed(path) for voice, path in en_refs.items()}

    mapping: dict[str, str] = {}
    scores: dict[str, dict[str, float]] = {}
    for vi_voice, spec in pipeline.VI_TTS.voices.items():
        synth_wav = SYNTH_DIR / f"{vi_voice}.wav"
        if not synth_wav.exists():
            raise FileNotFoundError(f"{synth_wav} missing; run qa_vi_voices.py first.")
        emb = embed(synth_wav)
        sims = {
            en_voice: cosine(emb, en_emb)
            for en_voice, en_emb in en_embs.items()
            if spec.gender not in ("female", "male")
            or pipeline.EN_TTS.voices[en_voice].gender == spec.gender
        }
        best = max(sims, key=sims.get)
        mapping[vi_voice] = best
        scores[vi_voice] = {name: round(sim, 4) for name, sim in sorted(sims.items(), key=lambda kv: -kv[1])}
        print(f"{vi_voice} ({spec.gender}) -> {best} (cos {sims[best]:.3f})", flush=True)

    pipeline.VI_TO_EN_JSON.parent.mkdir(parents=True, exist_ok=True)
    pipeline.VI_TO_EN_JSON.write_text(
        json.dumps({"map": mapping, "similarity": scores}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"VI->EN voice map saved: {pipeline.VI_TO_EN_JSON} ({len(mapping)} VI voices)")


if __name__ == "__main__":
    main()

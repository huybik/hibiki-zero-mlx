"""Score every VI voice (presets + VIVOS bank) for pruning decisions.

Per voice: synthesizes calibration sentences, then scores
- CER: PhoWhisper-small transcript vs the calibration text (garble/instability),
- fidelity: cosine(synth-audio embedding, enrolled reference embedding),
- rate: normalized chars per second (dragging / rushing outliers).

Writes per-voice wavs to voice_bank/vi_refs_synth/ (reused by match_voices.py
so matching measures what the training data actually sounds like) and a
scorecard to voice_bank/vi_voice_qa.tsv sorted worst-CER first.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

TRAINING_DATA_DIR = Path(__file__).resolve().parent
if str(TRAINING_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DATA_DIR))

import pipeline
from build_voice_bank import patch_torchaudio_load
from paths import DATASETS_DIR

CAL_TEXTS = [
    "Sáng nay trời rất đẹp, chúng tôi quyết định đi dạo quanh hồ rồi ghé quán quen uống cà phê.",
    "Cuối tuần này cả nhà sẽ về quê thăm ông bà và mang theo một ít quà từ thành phố.",
    "Giá xăng dầu trong nước dự kiến tiếp tục tăng nhẹ vào kỳ điều chỉnh ngày mai.",
]

SYNTH_DIR = DATASETS_DIR / "voice_bank" / "vi_refs_synth"
QA_TSV = DATASETS_DIR / "voice_bank" / "vi_voice_qa.tsv"
ASR_MODEL = "vinai/PhoWhisper-small"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def synthesize_all(voices: list[str]) -> dict[str, dict]:
    """Per voice: save concatenated wav, return per-sentence audio + embeddings."""
    import vieneu_mps_patch

    tts = pipeline.load_tts(pipeline.VI_TTS)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    jobs = [(voice, text) for voice in voices for text in CAL_TEXTS]
    audios: list[np.ndarray] = []
    for start in range(0, len(jobs), pipeline.BATCH_SIZE):
        chunk = jobs[start : start + pipeline.BATCH_SIZE]
        audios += vieneu_mps_patch.infer_batch_voices(
            tts, [job[1] for job in chunk], [job[0] for job in chunk]
        )
        print(f"synth {min(start + len(chunk), len(jobs))}/{len(jobs)}", flush=True)

    sr = pipeline.VI_TTS.sample_rate
    gap = np.zeros(int(0.3 * sr), dtype=np.float32)
    results: dict[str, dict] = {}
    for i, voice in enumerate(voices):
        parts = [np.asarray(a, dtype=np.float32) for a in audios[i * len(CAL_TEXTS) : (i + 1) * len(CAL_TEXTS)]]
        path = SYNTH_DIR / f"{voice}.wav"
        tts.save(np.concatenate([p for part in parts for p in (part, gap)]), str(path))
        synth_emb = tts.engine.prepare_reference(str(path), denoise=False, use_ref_codes=False)[0]
        ref_emb = tts._preset_voices[voice]["speaker_emb"]
        results[voice] = {
            "sentences": parts,
            "fidelity": cosine(synth_emb, ref_emb),
            "duration_s": sum(len(p) for p in parts) / sr,
        }
    pipeline.close_tts(tts)
    return results


def transcribe_and_score(results: dict[str, dict]) -> None:
    import torch
    import torchaudio
    from transformers import pipeline as hf_pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    asr = hf_pipeline("automatic-speech-recognition", model=ASR_MODEL, device=device)
    sr = pipeline.VI_TTS.sample_rate

    refs = [normalize(t) for t in CAL_TEXTS]
    for number, (voice, data) in enumerate(results.items(), 1):
        errors = 0
        ref_chars = 0
        for sentence_audio, ref in zip(data["sentences"], refs):
            wav16 = torchaudio.functional.resample(
                torch.from_numpy(sentence_audio), sr, 16_000
            ).numpy()
            hyp = normalize(asr({"raw": wav16, "sampling_rate": 16_000})["text"])
            errors += edit_distance(ref, hyp)
            ref_chars += len(ref)
        data["cer"] = errors / ref_chars
        data["rate"] = ref_chars / data["duration_s"]
        print(f"[{number}/{len(results)}] {voice}: cer={data['cer']:.3f}", flush=True)


def main() -> None:
    patch_torchaudio_load()
    voices = list(pipeline.VI_TTS.voices)
    results = synthesize_all(voices)
    transcribe_and_score(results)

    rows = sorted(results.items(), key=lambda kv: -kv[1]["cer"])
    lines = ["voice\tkind\tgender\tcer\tfidelity\trate_chars_s\tduration_s"]
    for voice, data in rows:
        spec = pipeline.VI_TTS.voices[voice]
        kind = "preset" if voice in pipeline.VIE_NEU_VOICES else "bank"
        lines.append(
            f"{voice}\t{kind}\t{spec.gender}\t{data['cer']:.4f}"
            f"\t{data['fidelity']:.4f}\t{data['rate']:.1f}\t{data['duration_s']:.1f}"
        )
    QA_TSV.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nScorecard: {QA_TSV}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

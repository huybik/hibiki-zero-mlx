from __future__ import annotations

import csv
import hashlib
import logging
import os
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# =========================
# Config
# =========================

# Pick one or both: ("vi",), ("en",), or ("vi", "en").
LANGUAGES = ("vi","en")

# Run selected languages in separate Python processes.
# *_WORKERS > 1 shards that language across multiple processes; each loads its own model.
PARALLEL_LANGUAGES = True
VI_WORKERS = 3
EN_WORKERS = 2

# Dataset range. If END_INDEX is None, the pipeline uses START_INDEX + N_SAMPLES.
START_INDEX = 140800
END_INDEX: int | None = None

# N_SAMPLES = 262000 # Total need to 1000 hours.
N_SAMPLES = 12800 # Total need to 1000 hours.

# Batch generation by voice when the TTS backend supports batched inference.
BATCH_SIZE = 16

# VieNeu's public infer_batch is sequential; its v3_turbo_serve engine is real
# CUDA batching. Keep this on for VI throughput, and fall back safely otherwise.
VI_USE_BATCH_ENGINE = True
VI_BATCH_USE_CUDAGRAPH = False

RANDOMIZE_VOICE = True
MATCH_VOICE_GENDER = True
MATCH_GENDERS = ("female", "male")
SEED: int | None = 0
SKIP_EXISTING = True
SCAN_AUDIO_TOTALS = False

# Override with a fixed voice by setting one of these to a voice name.
VI_VOICE: str | None = None
EN_VOICE: str | None = None

# Device control: "auto" picks cuda > mps > cpu; or force "cuda", "mps", "cpu".
TTS_DEVICE = "auto"
VI_DEVICE = TTS_DEVICE
EN_DEVICE = TTS_DEVICE

# Kokoro hits a few ops MPS doesn't implement; fall back to CPU for those only.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# VieNeu uses PyTorch on CUDA and ONNX on CPU when backend is "auto".
VI_BACKEND = "auto"
VI_DTYPE = "auto"

# Keep runtime output compact.
QUIET_HF_DOWNLOADS = True

if QUIET_HF_DOWNLOADS:
    os.environ.setdefault("UV_LINK_MODE", "copy")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.dynamic_module_utils").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=".*torch.nn.utils.weight_norm.*deprecated.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*dropout option adds dropout after all but last recurrent layer.*",
        category=UserWarning,
    )


# =========================
# Paths / Models
# =========================

TRAINING_DATA_DIR = Path(__file__).resolve().parent
if str(TRAINING_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DATA_DIR))

from load_raw import load_train_samples
from paths import DATASETS_DIR, HF_CACHE_DIR

PIPELINE_LANGUAGE_ENV = "PHOMT_PIPELINE_LANGUAGE"
PIPELINE_FIXED_GENDER_ENV = "PHOMT_PIPELINE_FIXED_GENDER"
PIPELINE_WORKER_INDEX_ENV = "PHOMT_PIPELINE_WORKER_INDEX"
PIPELINE_WORKER_COUNT_ENV = "PHOMT_PIPELINE_WORKER_COUNT"


@dataclass(frozen=True)
class VoiceSpec:
    gender: str
    style: str


@dataclass(frozen=True)
class TTSConfig:
    name: str
    provider: str
    model_repo: str
    mode: str
    lang_code: str
    default_voice: str
    device: str
    backend: str
    dtype: str
    sample_rate: int
    cache_dir: Path
    output_dir: Path
    voices: dict[str, VoiceSpec]


VIE_NEU_VOICES = {
    "Ngọc Lan": VoiceSpec(gender="female", style="soft / gentle"),
    "Ngọc Linh": VoiceSpec(gender="female", style="bright"),
    "Trúc Ly": VoiceSpec(gender="female", style="youthful"),
    "Mỹ Duyên": VoiceSpec(gender="female", style="smooth"),
    "Xuân Vĩnh": VoiceSpec(gender="male", style="upbeat"),
    "Thái Sơn": VoiceSpec(gender="male", style="firm"),
    "Gia Bảo": VoiceSpec(gender="male", style="smooth"),
    "Đức Trí": VoiceSpec(gender="male", style="clear"),
    "Trọng Hữu": VoiceSpec(gender="male", style="knowledgeable"),
    "Bình An": VoiceSpec(gender="male", style="even / calm"),
}

KOKORO_EN_VOICES = {
    "af_heart": VoiceSpec(gender="female", style="default"),
    "af_bella": VoiceSpec(gender="female", style="expressive"),
    "af_nicole": VoiceSpec(gender="female", style="soft"),
    "af_sarah": VoiceSpec(gender="female", style="clear"),
    "am_fenrir": VoiceSpec(gender="male", style="deep"),
    "am_michael": VoiceSpec(gender="male", style="clear"),
    "am_puck": VoiceSpec(gender="male", style="bright"),
}

# Kokoro voice speed multipliers. af_nicole is noticeably slow at 1.0 and
# creates large EN/VI duration mismatches.
KOKORO_EN_VOICE_SPEEDS = {
    "af_nicole": 1.35,
    "am_michael": 1.10,
}

VI_TTS = TTSConfig(
    name="vieNeu",
    provider="vieneu",
    model_repo="pnnbao-ump/VieNeu-TTS-v3-Turbo",
    mode="v3turbo",
    lang_code="vi",
    default_voice="Ngọc Lan",
    device=VI_DEVICE,
    backend=VI_BACKEND,
    dtype=VI_DTYPE,
    sample_rate=48_000,
    cache_dir=HF_CACHE_DIR,
    output_dir=DATASETS_DIR / "vieNeu" / "outputs" / "vi",
    voices=VIE_NEU_VOICES,
)

EN_TTS = TTSConfig(
    name="kokoro",
    provider="kokoro",
    model_repo="hexgrad/Kokoro-82M",
    mode="",
    lang_code="a",
    default_voice="af_heart",
    device=EN_DEVICE,
    backend="auto",
    dtype="auto",
    sample_rate=24_000,
    cache_dir=HF_CACHE_DIR,
    output_dir=DATASETS_DIR / "english" / "outputs" / "en",
    voices=KOKORO_EN_VOICES,
)

LANGUAGE_CONFIGS = {
    "vi": VI_TTS,
    "en": EN_TTS,
}

MANIFEST_FIELDNAMES = [
    "index",
    "language",
    "text",
    "audio_path",
    "provider",
    "model_repo",
    "voice",
    "gender",
    "style",
    "speed",
    "sample_rate",
    "duration_s",
    "status",
]


class KokoroTTS:
    supports_real_batch = True

    def __init__(self, config: TTSConfig, device: str):
        try:
            from kokoro import KPipeline
        except ImportError as error:
            raise ImportError(
                "Kokoro is not installed; see training-data/README.md for env setup."
            ) from error

        repo_id = config.model_repo or None
        self.pipeline = KPipeline(lang_code=config.lang_code, repo_id=repo_id, device=device)
        self.default_voice = config.default_voice
        self.sample_rate = config.sample_rate

    def infer(self, text: str, voice: str | None = None, speed: float = 1.0, **kwargs):
        import numpy as np

        parts = []
        for result in self.pipeline(text, voice=voice or self.default_voice, speed=speed, **kwargs):
            if result.audio is not None:
                parts.append(result.audio.detach().cpu().numpy())

        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def infer_batch(
        self, texts: list[str], voice: str | None = None, speed: float = 1.0, **kwargs
    ) -> list:
        import numpy as np

        grouped: list[list] = [[] for _ in texts]
        for result in self.pipeline(
            texts, voice=voice or self.default_voice, speed=speed, **kwargs
        ):
            if result.audio is None:
                continue
            text_index = result.text_index if result.text_index is not None else 0
            grouped[text_index].append(result.audio.detach().cpu().numpy())

        return [
            np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            for parts in grouped
        ]

    def save(self, audio, output_path: str) -> None:
        import soundfile as sf

        sf.write(output_path, audio, self.sample_rate)


class BatchedVieneuTTS:
    supports_real_batch = True

    def __init__(self, tts, *, use_cudagraph: bool = VI_BATCH_USE_CUDAGRAPH):
        if getattr(tts, "backend", None) != "pytorch":
            raise ValueError("VieNeu batch engine requires the PyTorch backend.")

        from vieneu.v3_turbo_serve.engine import V3TurboBatchEngine

        self.tts = tts
        self.batch_engine = V3TurboBatchEngine(tts.engine)
        self.use_cudagraph = use_cudagraph
        self.sample_rate = tts.sample_rate

    def infer(self, text: str, **kwargs):
        return self.infer_batch([text], **kwargs)[0]

    def infer_batch(
        self,
        texts: list[str],
        ref_audio=None,
        ref_codes=None,
        ref_text=None,  # noqa: ARG002 - v3 turbo ignores reference text.
        voice=None,
        emotion: str = "natural",
        temperature: float = 0.8,
        top_k: int = 25,
        top_p: float = 0.95,
        max_new_frames: int = 300,
        repetition_penalty: float = 1.2,
        max_chars: int = 384,
        silence_p: float = 0.15,
        crossfade_p: float = 0.0,
        apply_watermark: bool = True,
        **kwargs,
    ) -> list:
        import numpy as np
        from vieneu_utils.core_utils import join_audio_chunks
        from vieneu_utils.phonemize_text import (
            normalize_to_chunks_v3,
            phonemize_text_with_emotions,
        )

        ref_codes, voice_token_id = self.tts._resolve_v3_ref(
            voice, ref_audio, ref_codes
        )
        grouped_wavs: list[list] = [[] for _ in texts]
        requests = []
        request_text_indexes = []

        for text_index, text in enumerate(texts):
            for chunk in normalize_to_chunks_v3(text, max_chars=max_chars):
                requests.append(
                    {
                        "text": "",
                        "phonemes": phonemize_text_with_emotions(chunk),
                        "ref_codes": ref_codes,
                        "emotion": emotion,
                        "voice_token_id": voice_token_id,
                    }
                )
                request_text_indexes.append(text_index)

        if requests:
            chunk_wavs = self.batch_engine.generate_batch(
                requests,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_frames=max_new_frames,
                use_cudagraph=self.use_cudagraph,
            )
            for text_index, wav in zip(request_text_indexes, chunk_wavs):
                grouped_wavs[text_index].append(wav)

        outputs = []
        for wavs in grouped_wavs:
            wav = (
                join_audio_chunks(wavs, self.sample_rate, silence_p, crossfade_p)
                if wavs
                else np.zeros(0, dtype=np.float32)
            )
            outputs.append(self.tts._apply_watermark(wav) if apply_watermark else wav)
        return outputs

    def save(self, audio, output_path: str) -> None:
        self.tts.save(audio, output_path)

    def close(self) -> None:
        self.tts.close()


def resolve_device(requested: str) -> str:
    valid_devices = {"auto", "cpu", "cuda", "mps"}
    if requested not in valid_devices:
        raise ValueError(f"TTS device must be one of {sorted(valid_devices)}")

    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return requested


def load_tts(config: TTSConfig):
    device = resolve_device(config.device)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    if config.provider == "kokoro":
        return KokoroTTS(config, device)

    if config.provider != "vieneu":
        raise ValueError(f"Unsupported TTS provider: {config.provider}")

    try:
        from vieneu import Vieneu
    except ImportError as error:
        raise ImportError(
            "VieNeu is not installed; see training-data/README.md for env setup."
        ) from error

    # ONNX is VieNeu's CPU path; GPU devices need the PyTorch backend.
    backend = config.backend
    if device in ("cuda", "mps"):
        if backend == "onnx":
            raise ValueError("VieNeu on cuda/mps requires backend='auto' or 'pytorch'.")
        backend = "pytorch"

    tts = Vieneu(
        mode=config.mode,
        backbone_repo=config.model_repo,
        device=device,
        backend=backend,
        dtype=config.dtype,
    )
    # V3TurboBatchEngine is CUDA-specific batching; MPS/CPU use VieNeu's sequential path.
    if (
        VI_USE_BATCH_ENGINE
        and device == "cuda"
        and getattr(tts, "backend", None) == "pytorch"
    ):
        return BatchedVieneuTTS(tts)
    return tts


def close_tts(tts) -> None:
    close = getattr(tts, "close", None)
    if close is not None:
        close()


def batched(items: list, batch_size: int) -> Iterable[list]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def stable_rng(seed: int | None, *parts: object) -> random.Random:
    seed_text = "|".join(
        ["none" if seed is None else str(seed), *(str(part) for part in parts)]
    )
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def pick_target_gender(
    dataset_index: int, seed: int | None, fixed_matched_gender: str | None = None
) -> str | None:
    if not MATCH_VOICE_GENDER:
        return None
    if fixed_matched_gender:
        return fixed_matched_gender
    return stable_rng(seed, dataset_index, "gender").choice(MATCH_GENDERS)


def pick_voice(
    config: TTSConfig,
    rng: random.Random,
    fixed_voice: str | None,
    target_gender: str | None = None,
) -> str:
    if fixed_voice:
        return fixed_voice
    if RANDOMIZE_VOICE and config.voices:
        voices = [
            voice
            for voice, spec in config.voices.items()
            if target_gender is None or spec.gender == target_gender
        ]
        if voices:
            return rng.choice(voices)
        return rng.choice(list(config.voices))
    return config.default_voice


def get_voice_spec(config: TTSConfig, voice: str) -> VoiceSpec:
    if voice in config.voices:
        return config.voices[voice]

    if config.provider == "kokoro":
        gender = "female" if len(voice) > 1 and voice[1] == "f" else "male"
        return VoiceSpec(gender=gender, style="unknown")

    return VoiceSpec(gender="unknown", style="unknown")


def get_voice_speed(config: TTSConfig, voice: str) -> float | None:
    if config.provider != "kokoro":
        return None
    return KOKORO_EN_VOICE_SPEEDS.get(voice, 1.0)


def speed_matches(existing_speed: str | None, speed: float | None) -> bool:
    if speed is None:
        return True
    if not existing_speed:
        existing = 1.0
    else:
        try:
            existing = float(existing_speed)
        except ValueError:
            return False
    return abs(existing - speed) < 1e-6


def get_fixed_voice(language: str) -> str | None:
    if language == "vi":
        return VI_VOICE
    if language == "en":
        return EN_VOICE
    return None


def get_fixed_matched_gender(languages: tuple[str, ...]) -> str | None:
    fixed_genders = set()
    for language in languages:
        fixed_voice = get_fixed_voice(language)
        if not fixed_voice:
            continue

        config = LANGUAGE_CONFIGS[language]
        gender = get_voice_spec(config, fixed_voice).gender
        if gender != "unknown":
            fixed_genders.add(gender)

    if len(fixed_genders) > 1:
        raise ValueError("Fixed VI_VOICE and EN_VOICE must use the same gender.")

    return next(iter(fixed_genders), None)


def infer_one_or_batch(tts, texts: list[str], voice: str, **infer_kwargs) -> list:
    kwargs = {**infer_kwargs, "voice": voice}
    if len(texts) == 1:
        return [tts.infer(texts[0], **kwargs)]

    if supports_real_batch(tts):
        return tts.infer_batch(texts, **kwargs)

    return [tts.infer(text, **kwargs) for text in texts]


def supports_real_batch(tts) -> bool:
    return bool(getattr(tts, "supports_real_batch", False))


def get_language_worker_count(language: str) -> int:
    if language == "vi":
        return max(1, VI_WORKERS)
    if language == "en":
        return max(1, EN_WORKERS)
    return 1


def get_worker_label(language: str, worker_index: int, worker_count: int) -> str:
    if worker_count == 1:
        return language
    return f"{language}[{worker_index + 1}/{worker_count}]"


def synthesize_language(
    language: str,
    *,
    n_samples: int = N_SAMPLES,
    start_index: int = START_INDEX,
    end_index: int | None = END_INDEX,
    batch_size: int = BATCH_SIZE,
    rows: list[dict] | None = None,
    seed: int | None = SEED,
    fixed_voice: str | None = None,
    fixed_matched_gender: str | None = None,
    skip_existing: bool = SKIP_EXISTING,
    worker_index: int = 0,
    worker_count: int = 1,
    **infer_kwargs,
) -> list[Path]:
    if language not in LANGUAGE_CONFIGS:
        raise ValueError(f"Unsupported language: {language}")
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")

    config = LANGUAGE_CONFIGS[language]
    worker_label = get_worker_label(language, worker_index, worker_count)
    if rows is None:
        rows = load_train_samples(
            n=n_samples, start_index=start_index, end_index=end_index, verbose=False
        )

    jobs = []
    for offset, row in enumerate(rows):
        dataset_index = start_index + offset
        if dataset_index % worker_count != worker_index:
            continue

        text = row[language].strip()
        if not text:
            continue

        output_path = config.output_dir / f"{language}_{dataset_index:06d}.wav"
        target_gender = pick_target_gender(dataset_index, seed, fixed_matched_gender)
        voice_rng = stable_rng(seed, dataset_index, language, "voice")
        voice = pick_voice(config, voice_rng, fixed_voice, target_gender)
        speed = get_voice_speed(config, voice)
        if "speed" in infer_kwargs and config.provider == "kokoro":
            speed = float(infer_kwargs["speed"])
        jobs.append((dataset_index, text, voice, speed, output_path))

    if not jobs:
        return []

    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = get_manifest_path(config, worker_index, worker_count)
    final_manifest_path = get_manifest_path(config)
    final_manifest = read_manifest(final_manifest_path)
    worker_manifest = {} if manifest_path == final_manifest_path else read_manifest(manifest_path)
    existing_manifest = final_manifest.copy()
    existing_manifest.update(worker_manifest)

    paths: list[Path] = []
    manifest_rows = (
        existing_manifest.copy()
        if manifest_path == final_manifest_path
        else worker_manifest.copy()
    )
    generated_count = 0
    skipped_count = 0

    tts = None
    try:
        generation_jobs = []

        for index, text, voice, speed, output_path in jobs:
            spec = get_voice_spec(config, voice)
            existing_row = existing_manifest.get(index)
            existing_voice = existing_row.get("voice") if existing_row else None
            existing_gender = existing_row.get("gender") if existing_row else None
            existing_speed = existing_row.get("speed") if existing_row else None
            can_skip = (
                skip_existing
                and output_path.exists()
                and existing_voice == voice
                and existing_gender == spec.gender
                and speed_matches(existing_speed, speed)
            )

            if can_skip:
                skipped_count += 1
                paths.append(output_path)
                manifest_rows[index] = make_manifest_row(
                    index,
                    language,
                    text,
                    output_path,
                    config,
                    voice,
                    spec,
                    speed,
                    "skipped",
                    duration_s=existing_row.get("duration_s") if existing_row else None,
                )
                continue

            generation_jobs.append((index, text, voice, speed, output_path, spec))

        jobs_by_voice = {}
        for generation_job in generation_jobs:
            jobs_by_voice.setdefault((generation_job[2], generation_job[3]), []).append(
                generation_job
            )

        total_batches = sum(
            (len(voice_jobs) + batch_size - 1) // batch_size
            for voice_jobs in jobs_by_voice.values()
        )
        batch_number = 0

        for (voice, speed), voice_jobs in jobs_by_voice.items():
            voice_infer_kwargs = dict(infer_kwargs)
            if speed is not None:
                voice_infer_kwargs["speed"] = speed

            for voice_batch in batched(voice_jobs, batch_size):
                batch_number += 1
                batch_indexes = [job[0] for job in voice_batch]
                speed_text = "" if speed is None else f", speed={speed:g}"
                print(
                    f"{worker_label}: batch {batch_number}/{total_batches} "
                    f"indexes {min(batch_indexes)}-{max(batch_indexes)} "
                    f"({len(voice_batch)} files), voice={voice}{speed_text}",
                    flush=True,
                )

                if tts is None:
                    tts = load_tts(config)

                texts = [job[1] for job in voice_batch]
                audios = infer_one_or_batch(
                    tts, texts, voice=voice, **voice_infer_kwargs
                )
                if len(audios) != len(voice_batch):
                    raise RuntimeError(
                        f"{config.name} returned {len(audios)} audio outputs "
                        f"for {len(voice_batch)} texts."
                    )

                for (index, text, voice, speed, output_path, spec), audio in zip(
                    voice_batch, audios
                ):
                    tts.save(audio, str(output_path))
                    generated_count += 1
                    paths.append(output_path)
                    manifest_rows[index] = make_manifest_row(
                        index,
                        language,
                        text,
                        output_path,
                        config,
                        voice,
                        spec,
                        speed,
                        "generated",
                        duration_s=get_audio_array_duration_seconds(
                            audio, config.sample_rate
                        ),
                    )
                write_manifest(manifest_path, list(manifest_rows.values()))
    finally:
        if tts is not None:
            close_tts(tts)

    write_manifest(manifest_path, list(manifest_rows.values()))
    print(
        f"{worker_label}: {len(paths)}/{len(jobs)} complete "
        f"({generated_count} generated, {skipped_count} skipped)"
    )
    return paths


def get_manifest_path(
    config: TTSConfig, worker_index: int = 0, worker_count: int = 1
) -> Path:
    if worker_count == 1:
        return config.output_dir / "manifest.csv"
    return config.output_dir / f"manifest.worker{worker_index:02d}-of-{worker_count:02d}.csv"


def merge_worker_manifests(language: str, worker_count: int) -> None:
    if worker_count == 1:
        return

    config = LANGUAGE_CONFIGS[language]
    merged = read_manifest(get_manifest_path(config))
    for worker_index in range(worker_count):
        merged.update(read_manifest(get_manifest_path(config, worker_index, worker_count)))
    write_manifest(get_manifest_path(config), list(merged.values()))


def read_manifest(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}

    with path.open("r", newline="", encoding="utf-8") as file:
        return {int(row["index"]): row for row in csv.DictReader(file)}


def get_audio_array_duration_seconds(audio, sample_rate: int) -> float:
    shape = getattr(audio, "shape", None)
    frames = int(shape[-1]) if shape else len(audio)
    return frames / sample_rate


def make_manifest_row(
    index: int,
    language: str,
    text: str,
    output_path: Path,
    config: TTSConfig,
    voice: str,
    spec: VoiceSpec,
    speed: float | None,
    status: str,
    duration_s: float | str | None = None,
) -> dict:
    return {
        "index": index,
        "language": language,
        "text": text,
        "audio_path": str(output_path),
        "provider": config.provider,
        "model_repo": config.model_repo,
        "voice": voice,
        "gender": spec.gender,
        "style": spec.style,
        "speed": "" if speed is None else f"{speed:g}",
        "sample_rate": config.sample_rate,
        "duration_s": "" if duration_s in (None, "") else f"{float(duration_s):.9g}",
        "status": status,
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["index"])))


def format_audio_duration(seconds: float) -> str:
    hours = seconds / 3600
    minutes = seconds / 60
    return f"{hours:.2f} hours ({minutes:.1f} minutes)"


def get_audio_duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf
    except ImportError as error:
        raise ImportError(
            "soundfile is required to calculate audio totals. Run `uv sync` first."
        ) from error

    info = sf.info(path)
    return info.frames / info.samplerate


def current_generated_audio_totals(
    languages: tuple[str, ...] = LANGUAGES, *, print_summary: bool = True
) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, float | int]] = {}

    for language in languages:
        config = LANGUAGE_CONFIGS[language]
        manifest = read_manifest(config.output_dir / "manifest.csv")
        seconds = 0.0
        files = 0
        missing = 0

        for row in manifest.values():
            audio_path = Path(row["audio_path"])
            if not audio_path.exists():
                missing += 1
                continue

            seconds += get_audio_duration_seconds(audio_path)
            files += 1

        totals[language] = {
            "files": files,
            "missing": missing,
            "seconds": seconds,
            "hours": seconds / 3600,
        }

    if print_summary:
        print("Current generated audio totals:")
        for language, total in totals.items():
            missing_text = f", {total['missing']} missing" if total["missing"] else ""
            print(
                f"{language}: {format_audio_duration(float(total['seconds']))} "
                f"from {total['files']} files{missing_text}"
            )

    return totals


def run_pipeline() -> None:
    fixed_matched_gender = get_fixed_matched_gender(LANGUAGES)
    total_workers = sum(get_language_worker_count(language) for language in LANGUAGES)

    if PARALLEL_LANGUAGES and total_workers > 1:
        processes = []
        for language in LANGUAGES:
            worker_count = get_language_worker_count(language)
            for worker_index in range(worker_count):
                label = get_worker_label(language, worker_index, worker_count)
                process = start_language_process(
                    language,
                    fixed_matched_gender,
                    worker_index=worker_index,
                    worker_count=worker_count,
                )
                processes.append((label, language, worker_count, process))

        failed_workers = [
            label for label, _language, _worker_count, process in processes
            if process.wait() != 0
        ]
        if failed_workers:
            raise RuntimeError(f"Synthesis failed for: {', '.join(failed_workers)}")

        for language in LANGUAGES:
            merge_worker_manifests(language, get_language_worker_count(language))
    else:
        rows = list(
            load_train_samples(
                n=N_SAMPLES, start_index=START_INDEX, end_index=END_INDEX, verbose=False
            )
        )
        for language in LANGUAGES:
            fixed_voice = get_fixed_voice(language)
            synthesize_language(
                language,
                rows=rows,
                fixed_voice=fixed_voice,
                fixed_matched_gender=fixed_matched_gender,
            )

    if SCAN_AUDIO_TOTALS:
        current_generated_audio_totals(LANGUAGES)


def start_language_process(
    language: str,
    fixed_matched_gender: str | None,
    *,
    worker_index: int = 0,
    worker_count: int = 1,
) -> subprocess.Popen:
    env = os.environ.copy()
    env[PIPELINE_LANGUAGE_ENV] = language
    env[PIPELINE_WORKER_INDEX_ENV] = str(worker_index)
    env[PIPELINE_WORKER_COUNT_ENV] = str(worker_count)
    if fixed_matched_gender:
        env[PIPELINE_FIXED_GENDER_ENV] = fixed_matched_gender
    else:
        env.pop(PIPELINE_FIXED_GENDER_ENV, None)

    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=str(TRAINING_DATA_DIR.parent),
        env=env,
    )


def run_pipeline_worker() -> None:
    language = os.environ[PIPELINE_LANGUAGE_ENV]
    worker_index = int(os.environ.get(PIPELINE_WORKER_INDEX_ENV, "0"))
    worker_count = int(os.environ.get(PIPELINE_WORKER_COUNT_ENV, "1"))
    fixed_voice = get_fixed_voice(language)
    fixed_matched_gender = os.environ.get(PIPELINE_FIXED_GENDER_ENV) or None
    synthesize_language(
        language,
        fixed_voice=fixed_voice,
        fixed_matched_gender=fixed_matched_gender,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    if worker_count == 1 and SCAN_AUDIO_TOTALS:
        current_generated_audio_totals((language,))


def pipeline_main() -> None:
    if os.environ.get(PIPELINE_LANGUAGE_ENV):
        run_pipeline_worker()
    else:
        run_pipeline()


if __name__ == "__main__":
    pipeline_main()

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
PARALLEL_LANGUAGES = True

# Dataset range. If END_INDEX is None, the pipeline uses START_INDEX + N_SAMPLES.
START_INDEX = 0
END_INDEX: int | None = None

# N_SAMPLES = 262000 # Total need to 1000 hours.
N_SAMPLES = 10240 # Total need to 1000 hours.

# Batch generation by voice when the TTS backend supports batched inference.
BATCH_SIZE = 16

RANDOMIZE_VOICE = True
MATCH_VOICE_GENDER = True
MATCH_GENDERS = ("female", "male")
SEED: int | None = 0
SKIP_EXISTING = True

# Override with a fixed voice by setting one of these to a voice name.
VI_VOICE: str | None = None
EN_VOICE: str | None = None

# Device control. Use "cuda" to require GPU, "cpu" to force CPU, or "auto" to let the model choose.
TTS_DEVICE = "cuda"
VI_DEVICE = TTS_DEVICE
EN_DEVICE = TTS_DEVICE

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
DATASETS_DIR = Path(r"D:\Code\datasets")
HF_CACHE_DIR = DATASETS_DIR / ".hf_cache"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
PIPELINE_LANGUAGE_ENV = "PHOMT_PIPELINE_LANGUAGE"
PIPELINE_FIXED_GENDER_ENV = "PHOMT_PIPELINE_FIXED_GENDER"

if str(TRAINING_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DATA_DIR))

from load_raw import load_train_samples


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
    "status",
]


class KokoroTTS:
    def __init__(self, config: TTSConfig):
        config.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            from kokoro import KPipeline
        except ImportError as error:
            raise ImportError(
                "Kokoro is not installed in this Python environment. "
                "Run this script with `uv run python training-data/pipeline.py` "
                "or run `uv sync` first."
            ) from error

        device = None if config.device == "auto" else config.device
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
        try:
            import soundfile as sf
        except ImportError as error:
            raise ImportError(
                "soundfile is not installed in this Python environment. "
                "Run this script with `uv run python training-data/pipeline.py` "
                "or run `uv sync` first."
            ) from error

        sf.write(output_path, audio, self.sample_rate)


def load_tts(config: TTSConfig):
    validate_tts_runtime(config)

    config.cache_dir.mkdir(parents=True, exist_ok=True)

    if config.provider == "kokoro":
        return KokoroTTS(config)

    if config.provider != "vieneu":
        raise ValueError(f"Unsupported TTS provider: {config.provider}")

    try:
        from vieneu import Vieneu
    except ImportError as error:
        raise ImportError(
            "VieNeu is not installed in this Python environment. "
            "Run this script with `uv run python training-data/pipeline.py` or run `uv sync` first."
        ) from error

    return Vieneu(
        mode=config.mode,
        backbone_repo=config.model_repo,
        device=config.device,
        backend=config.backend,
        dtype=config.dtype,
    )


def validate_tts_runtime(config: TTSConfig) -> None:
    valid_devices = {"auto", "cpu", "cuda"}
    if config.device not in valid_devices:
        raise ValueError(f"{config.name} device must be one of {sorted(valid_devices)}")

    if config.provider == "vieneu" and config.device == "cuda" and config.backend == "onnx":
        raise ValueError("VieNeu CUDA requires backend='auto' or backend='pytorch'.")

    if config.device != "cuda":
        return

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("CUDA was requested, but torch is not installed.") from error

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build or set TTS_DEVICE = 'auto'/'cpu'."
        )


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

    infer_batch = getattr(tts, "infer_batch", None)
    if infer_batch is not None:
        return infer_batch(texts, **kwargs)

    return [tts.infer(text, **kwargs) for text in texts]


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
    **infer_kwargs,
) -> list[Path]:
    if language not in LANGUAGE_CONFIGS:
        raise ValueError(f"Unsupported language: {language}")

    config = LANGUAGE_CONFIGS[language]
    if rows is None:
        rows = load_train_samples(
            n=n_samples, start_index=start_index, end_index=end_index, verbose=False
        )

    jobs = []
    for offset, row in enumerate(rows):
        dataset_index = start_index + offset
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
    manifest_path = config.output_dir / "manifest.csv"
    existing_manifest = read_manifest(manifest_path)

    paths: list[Path] = []
    manifest_rows = existing_manifest.copy()
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
                    f"{language}: batch {batch_number}/{total_batches} "
                    f"indexes {min(batch_indexes)}-{max(batch_indexes)} "
                    f"({len(voice_batch)} files), voice={voice}{speed_text}",
                    flush=True,
                )

                if tts is None:
                    tts = load_tts(config)

                infer_batch = getattr(tts, "infer_batch", None)
                if len(voice_batch) > 1 and infer_batch is None:
                    for index, text, voice, speed, output_path, spec in voice_batch:
                        audio = infer_one_or_batch(
                            tts, [text], voice=voice, **voice_infer_kwargs
                        )[0]
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
                        )
                    continue

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
                    )
    finally:
        if tts is not None:
            close_tts(tts)

    write_manifest(manifest_path, list(manifest_rows.values()))
    print(
        f"{language}: {len(paths)}/{len(jobs)} complete "
        f"({generated_count} generated, {skipped_count} skipped)"
    )
    return paths


def read_manifest(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}

    with path.open("r", newline="", encoding="utf-8") as file:
        return {int(row["index"]): row for row in csv.DictReader(file)}


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

    if PARALLEL_LANGUAGES and len(LANGUAGES) > 1:
        processes = {
            language: start_language_process(language, fixed_matched_gender)
            for language in LANGUAGES
        }
        failed_languages = [
            language for language, process in processes.items() if process.wait() != 0
        ]
        if failed_languages:
            raise RuntimeError(f"Synthesis failed for: {', '.join(failed_languages)}")
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

    current_generated_audio_totals(LANGUAGES)


def start_language_process(
    language: str, fixed_matched_gender: str | None
) -> subprocess.Popen:
    env = os.environ.copy()
    env[PIPELINE_LANGUAGE_ENV] = language
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
    fixed_voice = get_fixed_voice(language)
    fixed_matched_gender = os.environ.get(PIPELINE_FIXED_GENDER_ENV) or None
    synthesize_language(
        language,
        fixed_voice=fixed_voice,
        fixed_matched_gender=fixed_matched_gender,
    )
    current_generated_audio_totals((language,))


def run_language(language: str) -> list[Path]:
    fixed_voice = get_fixed_voice(language)
    fixed_matched_gender = get_fixed_matched_gender((language,))
    return synthesize_language(
        language, fixed_voice=fixed_voice, fixed_matched_gender=fixed_matched_gender
    )


# Backward-compatible helpers.
def synthesize_vi_train_samples(
    n: int = N_SAMPLES, start_index: int = START_INDEX, **kwargs
) -> list[Path]:
    return synthesize_language("vi", n_samples=n, start_index=start_index, **kwargs)


def synthesize_en_train_samples(
    n: int = N_SAMPLES, start_index: int = START_INDEX, **kwargs
) -> list[Path]:
    return synthesize_language("en", n_samples=n, start_index=start_index, **kwargs)


def pipeline_main() -> None:
    if os.environ.get(PIPELINE_LANGUAGE_ENV):
        run_pipeline_worker()
    else:
        run_pipeline()


if __name__ == "__main__":
    pipeline_main()

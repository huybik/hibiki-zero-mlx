from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass
from itertools import combinations
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
VI_WORKERS = 1
# Five Core ML workers trade a little EN throughput for ~3-4GB RAM headroom
# (EN finishes before VI anyway); non-macOS keeps the bandwidth-bound
# compiled-CPU layout at ten single-thread workers.
EN_WORKERS = 5 if sys.platform == "darwin" else 10

# Dataset range. If END_INDEX is None, the pipeline uses START_INDEX + N_SAMPLES.
START_INDEX = 345600
END_INDEX: int | None = None

# Hub has 337,519 rows ≈ 568 VI-h; this tranche (mean 6.35 s, ~98.6% in-band)
# adds ~634 VI-h to land ~1,200 VI-h total (200 h safety over the 1k goal).
N_SAMPLES = 364800

# Batch generation by voice when the TTS backend supports batched inference.
# vieneu >= 3.2 batches natively on the PyTorch backend (chunks from all texts
# share forward steps, max_batch_size=32 by default).
BATCH_SIZE = 32

# CUDA per-frame cost is launch-bound (graphed) and nearly flat in batch size;
# used as both the vieneu engine max_batch_size and the per-call text batch.
CUDA_BATCH_SIZE = 128

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
# Kokoro is faster on CPU than MPS here, and keeping it off the GPU leaves MPS to vieneu.
EN_DEVICE = "cpu"

# Run the vieneu backbone/acoustic decoder in fp16 (vieneu_mps_patch.apply_fp16).
VI_FP16 = True

# Fold weight_norm and torch.compile Kokoro's iSTFTNet decoder (~92% of EN CPU
# time); inductor's elementwise fusion buys ~+20% aggregate on the bandwidth-bound
# M-series. Warmup ~50s/worker with a warm inductor cache (~8 min cold).
EN_COMPILE = True

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


# vieneu 3.2.3 v3-turbo preset roster (from the package's voices_v3_turbo.json).
VIE_NEU_VOICES = {
    "Trúc Ly": VoiceSpec(gender="female", style="natural / Bắc"),
    "Ngọc Linh": VoiceSpec(gender="female", style="storytelling / Bắc"),
    "Đoan Trang": VoiceSpec(gender="female", style="natural / Bắc"),
    "Mai Anh": VoiceSpec(gender="female", style="news / Bắc"),
    "Thục Đoan": VoiceSpec(gender="female", style="storytelling / Nam"),
    "Thùy Dung": VoiceSpec(gender="female", style="news / Nam"),
    "Ngọc Trân": VoiceSpec(gender="female", style="natural / Trung"),
    "Minh Đức": VoiceSpec(gender="male", style="news / Bắc"),
    "Phạm Tuyên": VoiceSpec(gender="male", style="natural / Bắc"),
    "Thanh Bình": VoiceSpec(gender="male", style="storytelling / Bắc"),
    "Thái Sơn": VoiceSpec(gender="male", style="storytelling / Nam"),
    "Xuân Vĩnh": VoiceSpec(gender="male", style="natural / Nam"),
    "Minh Triết": VoiceSpec(gender="male", style="news / Nam"),
    "Quang Sơn": VoiceSpec(gender="male", style="natural / Trung"),
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


def kokoro_blend_voices() -> dict[str, VoiceSpec]:
    """Same-gender pair blends at 25/50/75 weights. Plain "a,b" names use
    KPipeline's mean blend; "a:0.75,b:0.25" is resolved by KokoroTTS."""
    by_gender: dict[str, list[str]] = {}
    for name, spec in KOKORO_EN_VOICES.items():
        by_gender.setdefault(spec.gender, []).append(name)
    blends: dict[str, VoiceSpec] = {}
    for gender, names in by_gender.items():
        for first, second in combinations(names, 2):
            blends[f"{first},{second}"] = VoiceSpec(gender=gender, style="blend")
            blends[f"{first}:0.75,{second}:0.25"] = VoiceSpec(gender=gender, style="blend")
            blends[f"{first}:0.25,{second}:0.75"] = VoiceSpec(gender=gender, style="blend")
    return blends

# Cloned VIVOS speakers enrolled by build_voice_bank.py; merged into the VI
# voice pool (and registered into the engine at load) when the bank exists.
VI_VOICE_BANK_JSON = DATASETS_DIR / "voice_bank" / "vi_voices.json"

# QA-pruned voices (qa_vi_voices.py scorecard, confirmed on a second pass):
# Xuân Vĩnh / Quang Sơn / SPK22 have unstable pronunciation (CER); the rest
# drag under 12 chars/s and chronically produce EN/VI duration-ratio outliers.
VI_VOICE_BLOCKLIST = {
    "Xuân Vĩnh", "Quang Sơn", "VIVOSSPK22",
    "VIVOSSPK07", "VIVOSSPK15", "VIVOSSPK21", "VIVOSSPK30", "VIVOSSPK45",
}

# Speaker-embedding-matched EN voice per VI voice, built by match_voices.py; the
# EN side derives its voice from the row's VI voice so paired rows share timbre.
VI_TO_EN_JSON = DATASETS_DIR / "voice_bank" / "vi_to_en_voices.json"


def load_voice_bank_specs() -> dict[str, VoiceSpec]:
    if not VI_VOICE_BANK_JSON.exists():
        return {}
    presets = json.loads(VI_VOICE_BANK_JSON.read_text(encoding="utf-8")).get("presets", {})
    return {
        name: VoiceSpec(gender=preset.get("gender") or "unknown", style=preset.get("style") or "cloned")
        for name, preset in presets.items()
        if name not in VIE_NEU_VOICES
    }


def load_vi_to_en_map() -> dict[str, str]:
    if not VI_TO_EN_JSON.exists():
        return {}
    return json.loads(VI_TO_EN_JSON.read_text(encoding="utf-8"))["map"]


VI_TO_EN_MAP = load_vi_to_en_map()


VI_TTS = TTSConfig(
    name="vieNeu",
    provider="vieneu",
    model_repo="pnnbao-ump/VieNeu-TTS-v3-Turbo",
    mode="v3turbo",
    lang_code="vi",
    default_voice="Ngọc Linh",
    device=VI_DEVICE,
    backend=VI_BACKEND,
    dtype=VI_DTYPE,
    sample_rate=48_000,
    cache_dir=HF_CACHE_DIR,
    output_dir=DATASETS_DIR / "vieNeu" / "outputs" / "vi",
    voices={
        name: spec
        for name, spec in {**VIE_NEU_VOICES, **load_voice_bank_specs()}.items()
        if name not in VI_VOICE_BLOCKLIST
    },
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
    voices={**KOKORO_EN_VOICES, **kokoro_blend_voices()},
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
        self._voice_cache: dict = {}

        if sys.platform == "darwin" and self.pipeline.model is not None:
            from kokoro_coreml import CoreMLDecoder

            self.pipeline.model.decoder = CoreMLDecoder(self.pipeline.model.decoder)
        elif EN_COMPILE and device == "cpu" and self.pipeline.model is not None:
            import torch
            from torch.nn.utils import remove_weight_norm
            from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

            model = self.pipeline.model
            for mod in model.modules():
                if is_parametrized(mod, "weight"):
                    remove_parametrizations(mod, "weight")
                else:
                    try:
                        remove_weight_norm(mod)
                    except ValueError:
                        pass
            model.decoder = torch.compile(model.decoder, dynamic=True)

    def resolve_voice(self, voice: str):
        """Weighted blend names ("a:0.75,b:0.25") -> pack tensor; others pass through."""
        if ":" not in voice:
            return voice
        cached = self._voice_cache.get(voice)
        if cached is None:
            cached = sum(
                self.pipeline.load_single_voice(name) * float(weight)
                for name, _, weight in (part.partition(":") for part in voice.split(","))
            )
            self._voice_cache[voice] = cached
        return cached

    def infer_batch(
        self, texts: list[str], voice: str | None = None, speed: float = 1.0, **kwargs
    ) -> list:
        import numpy as np

        grouped: list[list] = [[] for _ in texts]
        for result in self.pipeline(
            texts, voice=self.resolve_voice(voice or self.default_voice), speed=speed, **kwargs
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
        max_batch_size=CUDA_BATCH_SIZE if device == "cuda" else BATCH_SIZE,
    )
    if VI_VOICE_BANK_JSON.exists():
        register_voice_bank(tts, VI_VOICE_BANK_JSON)

    import vieneu_mps_patch

    vieneu_mps_patch.apply(tts)
    if VI_FP16:
        vieneu_mps_patch.apply_fp16(tts)
    return tts


def register_voice_bank(tts, path: Path) -> None:
    """Load bank presets (build_voice_bank.py / vieneu save_voices format) into the engine."""
    import numpy as np

    presets = json.loads(path.read_text(encoding="utf-8")).get("presets", {})
    for name, preset in presets.items():
        emb = preset.get("speaker_emb")
        codes = preset.get("codes")
        tts._preset_voices[name] = {
            "description": preset.get("description", ""),
            "gender": preset.get("gender", ""),
            "style": preset.get("style", "tu_nhien"),
            "speaker_emb": None if emb is None else np.asarray(emb, dtype=np.float32),
            "codes": None if codes is None else np.asarray(codes, dtype=np.int64),
        }


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


def pick_row_voice(
    language: str,
    dataset_index: int,
    seed: int | None,
    fixed_voice: str | None,
    target_gender: str | None,
) -> str:
    """VI picks freely; EN follows the row's VI voice through the embedding map."""
    if fixed_voice:
        return fixed_voice
    vi_rng = stable_rng(seed, dataset_index, "vi", "voice")
    vi_voice = pick_voice(LANGUAGE_CONFIGS["vi"], vi_rng, VI_VOICE, target_gender)
    if language == "vi":
        return vi_voice
    if not VI_TO_EN_MAP:
        raise RuntimeError(
            f"{VI_TO_EN_JSON} missing; run training-data/match_voices.py first."
        )
    return VI_TO_EN_MAP[vi_voice]


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
    total = weight_total = 0.0
    for part in voice.split(","):
        name, _, weight_text = part.partition(":")
        weight = float(weight_text) if weight_text else 1.0
        total += KOKORO_EN_VOICE_SPEEDS.get(name, 1.0) * weight
        weight_total += weight
    return total / weight_total


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
        voice = pick_row_voice(language, dataset_index, seed, fixed_voice, target_gender)
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

        if config.provider == "vieneu":
            if resolve_device(config.device) == "cuda":
                batch_size = max(batch_size, CUDA_BATCH_SIZE)
            # Global length-sorted batches (voices mixed per batch): every batch
            # runs to its longest member, so grouping similar lengths kills the
            # per-voice tail waste.
            generation_jobs.sort(key=lambda job: len(job[1]), reverse=True)
            job_batches = list(batched(generation_jobs, batch_size))
        else:
            jobs_by_voice: dict = {}
            for generation_job in generation_jobs:
                jobs_by_voice.setdefault(
                    (generation_job[2], generation_job[3]), []
                ).append(generation_job)
            job_batches = [
                voice_batch
                for voice_jobs in jobs_by_voice.values()
                for voice_batch in batched(voice_jobs, batch_size)
            ]

        for batch_number, job_batch in enumerate(job_batches, 1):
            batch_indexes = [job[0] for job in job_batch]
            print(
                f"{worker_label}: batch {batch_number}/{len(job_batches)} "
                f"indexes {min(batch_indexes)}-{max(batch_indexes)} "
                f"({len(job_batch)} files)",
                flush=True,
            )

            if tts is None:
                tts = load_tts(config)

            texts = [job[1] for job in job_batch]
            if config.provider == "vieneu":
                import vieneu_mps_patch

                audios = vieneu_mps_patch.infer_batch_voices(
                    tts, texts, [job[2] for job in job_batch]
                )
            else:
                voice, speed = job_batch[0][2], job_batch[0][3]
                voice_infer_kwargs = dict(infer_kwargs)
                if speed is not None:
                    voice_infer_kwargs["speed"] = speed
                audios = tts.infer_batch(texts, voice=voice, **voice_infer_kwargs)

            if len(audios) != len(job_batch):
                raise RuntimeError(
                    f"{config.name} returned {len(audios)} audio outputs "
                    f"for {len(job_batch)} texts."
                )

            batch_rows = []
            for (index, text, voice, speed, output_path, spec), audio in zip(
                job_batch, audios
            ):
                tts.save(audio, str(output_path))
                generated_count += 1
                paths.append(output_path)
                row = make_manifest_row(
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
                manifest_rows[index] = row
                batch_rows.append(row)
            append_manifest_rows(manifest_path, batch_rows)
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


def append_manifest_rows(path: Path, rows: list[dict]) -> None:
    """Append rows without rewriting the file; read_manifest last-wins on dupes."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDNAMES)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


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
    if language == "en":
        # Kokoro runs on CPU; 1 thread per worker is the most core-efficient layout.
        env["OMP_NUM_THREADS"] = "1"
        # Persist inductor artifacts across reboots (default TMPDIR cache is wiped).
        env.setdefault(
            "TORCHINDUCTOR_CACHE_DIR",
            str(Path.home() / ".cache" / "torchinductor-kokoro"),
        )
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

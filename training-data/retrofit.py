"""Rewrite pre-tranche-3 Hub shards with timbre-matched EN audio (see RETROFIT_PLAN.md).

Each row: embed the VI audio (VieNeu 192-d speaker encoder), classify gender by
nearest in-pool VI voice, pick the nearest same-gender EN voice from the 34-candidate
Kokoro grid, resynthesize EN, and rewrite the shard in place on the Hub.

Usage:
  python retrofit.py --pilot          # process one shard, print stats, no upload
  python retrofit.py                  # full resumable run
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "0")

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from paths import DATASETS_DIR

import pipeline

REPO = "anquachdev/PhoMT-en-vi-speech"
BOUNDARY = 345600
SHARDS_PER_COMMIT = 5
WORKERS = 7  # no VI generation to compete with; Kokoro+CoreML ran ~70x RT at 7 workers
SYNTH_BATCH = 32
STATE_FILE = "retrofit-state.json"
STAGING = DATASETS_DIR / "retrofit_staging"
EN_REFS = DATASETS_DIR / "voice_bank" / "en_refs"
VI_REFS = DATASETS_DIR / "voice_bank" / "vi_refs_synth"
MAX_REF_SECONDS = 8.0
MIN_RMS = 1e-4

_worker = {}


def load_encoder():
    from vieneu._v3_turbo_engine.speaker import OnnxSpeakerEncoder

    return OnnxSpeakerEncoder.from_pretrained(
        "pnnbao-ump/VieNeu-TTS-v3-Turbo", filename="speaker_encoder.onnx", device="cpu"
    )


def embed_audio(enc, wav: np.ndarray, sr: int) -> np.ndarray:
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    wav = wav[: int(MAX_REF_SECONDS * sr)]
    return np.asarray(enc.embed(wav.astype(np.float32), sr))


def embed_file(enc, path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    return embed_audio(enc, wav, sr)


def build_grids(enc):
    """(en_names, en_embs, en_genders), (vi_embs, vi_genders) as arrays for fast cosine."""
    en_names, en_embs, en_genders = [], [], []
    for voice in pipeline.EN_TTS.voices:
        ref = EN_REFS / (voice.replace(":", "").replace(",", "+") + ".wav")
        en_names.append(voice)
        en_embs.append(embed_file(enc, ref))
        en_genders.append(pipeline.get_voice_spec(pipeline.EN_TTS, voice).gender)
    vi_embs, vi_genders = [], []
    for voice in pipeline.VI_TTS.voices:
        vi_embs.append(embed_file(enc, VI_REFS / f"{voice}.wav"))
        vi_genders.append(pipeline.get_voice_spec(pipeline.VI_TTS, voice).gender)
    norm = lambda m: m / np.linalg.norm(m, axis=1, keepdims=True)
    return (
        (en_names, norm(np.stack(en_embs)), np.array(en_genders)),
        (norm(np.stack(vi_embs)), np.array(vi_genders)),
    )


def init_worker():
    _worker["enc"] = load_encoder()
    _worker["en_grid"], _worker["vi_grid"] = build_grids(_worker["enc"])
    _worker["tts"] = pipeline.load_tts(pipeline.EN_TTS)


def pick_en_voice(vi_audio_bytes: bytes) -> tuple[str, float]:
    enc = _worker["enc"]
    en_names, en_embs, en_genders = _worker["en_grid"]
    vi_embs, vi_genders = _worker["vi_grid"]
    wav, sr = sf.read(io.BytesIO(vi_audio_bytes), dtype="float32")
    emb = embed_audio(enc, wav, sr)
    emb = emb / np.linalg.norm(emb)
    gender = vi_genders[int(np.argmax(vi_embs @ emb))]
    sims = en_embs @ emb
    sims[en_genders != gender] = -np.inf
    best = int(np.argmax(sims))
    return en_names[best], float(sims[best])


def wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, pipeline.EN_TTS.sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def synth_rows(jobs: list[dict]) -> dict:
    """jobs: [{i, text, voice, vi_dur}] -> per-row audio with gate + ratio retry."""
    tts = _worker["tts"]
    out, kept_old = {}, 0
    by_voice = {}
    for job in jobs:
        by_voice.setdefault(job["voice"], []).append(job)
    for voice, group in by_voice.items():
        speed = pipeline.get_voice_speed(pipeline.EN_TTS, voice)
        for start in range(0, len(group), SYNTH_BATCH):
            chunk = group[start : start + SYNTH_BATCH]
            audios = tts.infer_batch([j["text"] for j in chunk], voice=voice, speed=speed)
            for job, audio in zip(chunk, audios):
                for _attempt in range(2):
                    dur = len(audio) / pipeline.EN_TTS.sample_rate
                    ratio = dur / job["vi_dur"] if job["vi_dur"] else 0.0
                    ok = (
                        np.isfinite(audio).all()
                        and float(np.sqrt(np.mean(audio**2))) >= MIN_RMS
                        and upload_ratio_ok(ratio)
                    )
                    if ok:
                        out[job["i"]] = (wav_bytes(audio), dur, ratio)
                        break
                    audio = tts.infer_batch([job["text"]], voice=voice, speed=speed)[0]
                else:
                    kept_old += 1  # keep the original EN clip; row count must not change
    return {"rows": out, "kept_old": kept_old}


def upload_ratio_ok(ratio: float) -> bool:
    import upload

    return upload.MIN_DURATION_RATIO <= ratio <= upload.MAX_DURATION_RATIO


def process_shard(path_in_repo: str) -> dict:
    started = time.perf_counter()
    local = hf_hub_download(REPO, path_in_repo, repo_type="dataset")
    table = pq.read_table(local)
    old_size = Path(local).stat().st_size

    vi_col = table.column("audio_vi").to_pylist()
    en_col = table.column("audio_en").to_pylist()
    texts = table.column("en").to_pylist()
    vi_durs = table.column("duration_vi_s").to_pylist()
    old_en_durs = table.column("duration_en_s").to_pylist()

    jobs, sims = [], []
    for i in range(table.num_rows):
        voice, sim = pick_en_voice(vi_col[i]["bytes"])
        jobs.append({"i": i, "text": texts[i], "voice": voice, "vi_dur": vi_durs[i]})
        sims.append(sim)
    result = synth_rows(jobs)

    new_en, new_dur, new_ratio = [], [], []
    for i in range(table.num_rows):
        if i in result["rows"]:
            b, dur, ratio = result["rows"][i]
            new_en.append({"bytes": b, "path": en_col[i]["path"]})
            new_dur.append(dur)
            new_ratio.append(ratio)
        else:
            new_en.append(en_col[i])
            new_dur.append(old_en_durs[i])
            new_ratio.append(old_en_durs[i] / vi_durs[i] if vi_durs[i] else 0.0)

    import pyarrow as pa

    def replace(tbl, name, values):
        idx = tbl.schema.get_field_index(name)
        return tbl.set_column(idx, tbl.schema.field(name), pa.array(values, type=tbl.schema.field(name).type))

    table = replace(table, "audio_en", new_en)
    table = replace(table, "duration_en_s", new_dur)
    table = replace(table, "duration_ratio_en_vi", new_ratio)

    STAGING.mkdir(parents=True, exist_ok=True)
    staged = STAGING / Path(path_in_repo).name
    pq.write_table(table, staged, compression="NONE")

    voices = sorted({j["voice"] for j in jobs})
    return {
        "path": path_in_repo,
        "staged": str(staged),
        "rows": table.num_rows,
        "rewritten": len(result["rows"]),
        "kept_old": result["kept_old"],
        "mean_sim": float(np.mean(sims)),
        "voices": voices,
        "old_size": old_size,
        "new_size": staged.stat().st_size,
        "seconds": time.perf_counter() - started,
    }


def affected_shards(api: HfApi) -> list[str]:
    files = api.list_repo_files(REPO, repo_type="dataset")
    parquets = {f for f in files if f.startswith("data/") and f.endswith(".parquet")}
    state = json.load(
        open(hf_hub_download(REPO, "upload-state.json", repo_type="dataset", force_download=True))
    )
    campaign = []
    for s in state["shards"]:
        if s["source_index_min"] < BOUNDARY:
            assert s["source_index_max"] < BOUNDARY, f"shard straddles boundary: {s}"
            campaign.append(s["path"])
    pre_campaign = sorted(parquets - {s["path"] for s in state["shards"]})
    return pre_campaign + campaign


def load_retrofit_state(api: HfApi) -> dict:
    try:
        p = hf_hub_download(REPO, STATE_FILE, repo_type="dataset", force_download=True)
        return json.load(open(p))
    except EntryNotFoundError:
        return {"done": [], "kept_old_total": 0}


def commit_batch(api: HfApi, batch: list[dict], state: dict) -> None:
    state["done"].extend(r["path"] for r in batch)
    state["kept_old_total"] += sum(r["kept_old"] for r in batch)
    operations = [
        *(CommitOperationAdd(path_in_repo=r["path"], path_or_fileobj=r["staged"]) for r in batch),
        CommitOperationAdd(
            path_in_repo=STATE_FILE,
            path_or_fileobj=json.dumps(state, indent=2).encode(),
        ),
    ]
    first, last = batch[0]["path"].split("/")[-1], batch[-1]["path"].split("/")[-1]
    api.create_commit(
        repo_id=REPO,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Retrofit timbre-matched EN audio: {first} .. {last} ({len(batch)} shards)",
    )
    for r in batch:
        Path(r["staged"]).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="one shard, stats only, no upload")
    args = parser.parse_args()

    api = HfApi()
    shards = affected_shards(api)
    state = load_retrofit_state(api)
    todo = [p for p in shards if p not in set(state["done"])]
    print(f"{len(shards)} affected shards, {len(state['done'])} done, {len(todo)} to go")

    if args.pilot:
        init_worker()
        r = process_shard(todo[0])
        print(json.dumps(r, indent=2))
        return

    done_rows = 0
    started = time.perf_counter()
    with Pool(WORKERS, initializer=init_worker) as pool:
        batch = []
        for r in pool.imap(process_shard, todo):
            batch.append(r)
            done_rows += r["rewritten"]
            print(
                f"{r['path'].split('/')[-1]}: {r['rewritten']}/{r['rows']} rewritten, "
                f"{r['kept_old']} kept old, sim {r['mean_sim']:.3f}, {r['seconds']:.0f}s",
                flush=True,
            )
            if len(batch) >= SHARDS_PER_COMMIT:
                commit_batch(api, batch, state)
                elapsed = (time.perf_counter() - started) / 3600
                print(
                    f"  committed {len(state['done'])}/{len(shards)} shards "
                    f"({done_rows} rows rewritten, {elapsed:.2f}h elapsed)",
                    flush=True,
                )
                batch = []
        if batch:
            commit_batch(api, batch, state)
    print(f"DONE: {len(state['done'])}/{len(shards)} shards, kept_old {state['kept_old_total']}")


if __name__ == "__main__":
    main()

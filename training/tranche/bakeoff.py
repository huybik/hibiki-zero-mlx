#!/usr/bin/env python
"""English TTS bake-off for the real-Vietnamese tranche: which model best voices a real Vietnamese
speaker in English ([real-vi-tranche](../../docs/agent-memory/real-vi-tranche.md#english-tts)).

Each step runs on the box in its own env (`training/tranche/envs.sh` builds them), all under
--root (default /workspace/tranche/bakeoff):

  clips  repo .venv                real Vietnamese clips (PhoAudiobook, viVoice, GigaSpeech 2), each
                                   with another clip of the same speaker as its reference and an
                                   English line to speak (`en-vi-val`'s English)
  synth  /workspace/envs/<env>     voice every clip's line in its speaker's voice, in each of the
                                   model's cloning modes, timed, its rows split over --workers
                                   processes on the GPU (envs voxcpm, dots, qwentts and cosyvoice)
  mimi   repo .venv                round-trip every output through Mimi at 16 codebooks, as the
                                   trainer hears it
  score  /workspace/envs/score     speaker similarity to the row's own Vietnamese clip (WeSpeaker
                                   ResNet293-LM), Whisper-large-v3 WER, the broken share, speed

From the repo root:
  .venv/bin/python -m training.tranche.bakeoff clips
  /workspace/envs/voxcpm/bin/python -m training.tranche.bakeoff synth voxcpm2 --workers 8  # dots,
  .venv/bin/python -m training.tranche.bakeoff mimi                     # qwen3tts, cosyvoice3
  /workspace/envs/score/bin/python -m training.tranche.bakeoff score

synth and mimi skip what they already wrote, so they resume; run mimi once every synth is done.
synth's workers need CUDA MPS (`nvidia-cuda-mps-control -d`): without it the GPU runs one
process's kernels at a time, and 4 workers gained 2.2x over one instead of 3.1x. Run one synth at
a time with the GPU otherwise idle, and each in one go: its RTF is the wall time from its first
row to its last.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.paths import DEFAULT_MIMI_WEIGHT

ROOT = Path("/workspace/tranche/bakeoff")
REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 1234
CLIP_S = (5.0, 16.0)  # clips and references; every model takes a reference this long
# Hub commits checked on 2026-11-04.
PHO = ("thivux/phoaudiobook", "f19f241", 803)  # repo, revision, train shards
VIVOICE = ("capleaf/viVoice", "693e988", 354)
GIGA = ("speechcolab/gigaspeech2", "8dc0d0e")
MODELS = {
    "voxcpm2": ("openbmb/VoxCPM2", "32279ef", ["ref", "ult"]),
    "dots": ("dots-studio/dots.tts-soar", "2f9b3e1", ["xvec", "cont"]),  # post-trained; base is pretraining
    "qwen3tts": ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "fd4b254", ["xvec", "icl"]),
    "cosyvoice3": ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "29e01c4", ["xling"]),
}
COSYVOICE_SRC = Path("/workspace/envs/CosyVoice")  # the repo at 074ca6d, with Matcha-TTS
WESPEAKER = ("Wespeaker/wespeaker-voxceleb-resnet293-LM", "6e6bffe")
WHISPER = ("openai/whisper-large-v3", "06f233f", ["*.json", "*.txt", "model.safetensors"])
SAME_SPEAKER = 0.4  # WeSpeaker cosine between a clip and its reference that counts as one speaker
BROKEN_WER = 0.2


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def snapshot(repo: str, revision: str, allow: list[str] | None = None) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, allow_patterns=allow)


def read_rows(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "clips.jsonl").open()]


# --- clips -------------------------------------------------------------------------------------


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes (all three corpora's) -> mono float32 and its rate."""
    import sphn

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(data)
        f.flush()
        pcm, rate = sphn.read(f.name)
    return pcm.mean(axis=0), int(rate)


def pairs_from_groups(groups: dict[str, list[tuple[np.ndarray, int, str]]], rng: random.Random,
                      per_group: int) -> list[tuple[str, tuple, tuple]]:
    """Per group (one speaker) with two clips in range: up to per_group (clip, reference) pairs."""
    out = []
    for key in sorted(groups):
        clips = [c for c in groups[key] if CLIP_S[0] <= len(c[0]) / c[1] <= CLIP_S[1]]
        rng.shuffle(clips)
        for k in range(min(per_group, len(clips) // 2)):
            out.append((key, clips[2 * k], clips[2 * k + 1]))
    return out


def phoaudiobook(rng: random.Random, n: int) -> list[tuple[str, tuple, tuple]]:
    """One pair per reader, from shards spread over the corpus (~7 readers a shard)."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, rev, count = PHO
    pairs: list = []
    used: set[str] = set()  # a reader can span several shards
    for shard in np.linspace(0, count - 1, 40).astype(int):
        name = f"data/train-{shard:05d}-of-{count:05d}.parquet"
        groups = defaultdict(list)
        for r in pq.read_table(hf_hub_download(repo, name, repo_type="dataset", revision=rev)).to_pylist():
            if r["speaker"] not in used:
                groups[r["speaker"]].append((*decode(r["audio"]["bytes"]), r["text"]))
        picked = pairs_from_groups(groups, rng, 1)
        used |= {speaker for speaker, _, _ in picked}
        pairs += picked
    rng.shuffle(pairs)
    return pairs[:n]


def vivoice(rng: random.Random, n: int) -> list[tuple[str, tuple, tuple]]:
    """One pair per channel; a channel can hold several voices, which scoring checks."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, rev, count = VIVOICE
    groups = defaultdict(list)
    for shard in np.linspace(0, count - 1, 3).astype(int):
        name = f"data/train-{shard:05d}-of-{count:05d}.parquet"
        for r in pq.read_table(hf_hub_download(repo, name, repo_type="dataset", revision=rev)).to_pylist():
            groups[r["channel"]].append((*decode(r["audio"]["bytes"]), r["text"]))
    pairs = pairs_from_groups(groups, rng, 1)
    rng.shuffle(pairs)
    return pairs[:n]


def gigaspeech2(rng: random.Random, n: int) -> list[tuple[str, tuple, tuple]]:
    """One pair per recording of the first train archive (refined transcripts, upper case)."""
    from huggingface_hub import hf_hub_download

    repo, rev = GIGA
    texts = {}
    for line in open(hf_hub_download(repo, "data/vi/train_refined.tsv", repo_type="dataset", revision=rev)):
        key, text = line.rstrip("\n").split("\t", 1)
        if key.startswith("0-"):
            texts[key] = text.lower()  # all upper case, unlike the other sources' prompts
    groups = defaultdict(list)
    archive = hf_hub_download(repo, "data/vi/train/0.tar.gz", repo_type="dataset", revision=rev)
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            key = Path(member.name).stem
            if member.isfile() and key in texts:
                groups[key.rsplit("-", 1)[0]].append((*decode(tar.extractfile(member).read()), texts[key]))
    pairs = pairs_from_groups(groups, rng, 1)
    rng.shuffle(pairs)
    return pairs[:n]


def clips(args: argparse.Namespace) -> None:
    import sphn

    root = args.root
    (root / "wav").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    lines = []
    for name in ("cv27", "libriheavy"):
        lines += [json.loads(x)["text_en"] for x in open(REPO_ROOT / f"data/eval/en-vi-val/{name}.jsonl")]
    rng.shuffle(lines)
    sources = [("pho", phoaudiobook, args.pho), ("vivoice", vivoice, args.vivoice),
               ("giga", gigaspeech2, args.giga)]
    rows = []
    for source, pick, n in sources:
        log(f"{source}: picking {n} pairs")
        for k, (speaker, clip, ref) in enumerate(pick(rng, n)):
            row_id = f"{source}{k:03d}"
            for kind, (pcm, rate, _) in (("clip", clip), ("ref", ref)):
                sphn.write_wav(str(root / f"wav/{row_id}.{kind}.wav"), pcm, rate)
            rows.append({
                "id": row_id, "source": source, "speaker": speaker,
                "clip": str(root / f"wav/{row_id}.clip.wav"), "clip_s": round(len(clip[0]) / clip[1], 2),
                "clip_text": clip[2],
                "ref": str(root / f"wav/{row_id}.ref.wav"), "ref_s": round(len(ref[0]) / ref[1], 2),
                "ref_text": ref[2], "text_en": lines[len(rows) % len(lines)],
            })
    with (root / "clips.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    log(f"{len(rows)} rows, {len({(r['source'], r['speaker']) for r in rows})} speakers")


# --- synth -------------------------------------------------------------------------------------


def load_model(model: str):
    """-> generate(row, mode, seed) -> (mono float32, rate), in the model's own env."""
    import torch

    repo, rev, _ = MODELS[model]
    path = snapshot(repo, rev)
    if model == "voxcpm2":
        from voxcpm import VoxCPM

        tts = VoxCPM.from_pretrained(path, load_denoiser=False)

        def generate(row, mode, seed):
            torch.manual_seed(seed)  # voxcpm 2.0.3's generate takes no seed
            extra = {"prompt_wav_path": row["ref"], "prompt_text": row["ref_text"]} if mode == "ult" else {}
            # Its default retries an output whose length is off up to 3 times and keeps the last;
            # off, so its broken share counts as the other models' do.
            wav = tts.generate(text=row["text_en"], reference_wav_path=row["ref"], retry_badcase=False, **extra)
            return np.asarray(wav, dtype=np.float32), tts.tts_model.sample_rate
    elif model == "dots":
        from dots_tts.runtime import DotsTtsRuntime
        from dots_tts.utils.util import seed_everything

        tts = DotsTtsRuntime.from_pretrained(path, precision="bfloat16")

        def generate(row, mode, seed):
            seed_everything(seed)
            # With a transcript, dots puts the language tag on it, which here is Vietnamese: tag
            # only the x-vector mode's English text.
            cont = mode == "cont"
            out = tts.generate(text=row["text_en"], prompt_audio_path=row["ref"],
                               language=None if cont else "en", prompt_text=row["ref_text"] if cont else None,
                               num_steps=10, guidance_scale=1.2)
            return out["audio"].float().cpu().squeeze().numpy(), out["sample_rate"]
    elif model == "qwen3tts":
        from qwen_tts import Qwen3TTSModel

        tts = Qwen3TTSModel.from_pretrained(path, device_map="cuda:0", dtype=torch.bfloat16)

        def generate(row, mode, seed):
            torch.manual_seed(seed)
            wavs, rate = tts.generate_voice_clone(
                text=row["text_en"], language="English", ref_audio=row["ref"],
                ref_text=row["ref_text"] if mode == "icl" else None,
                x_vector_only_mode=mode == "xvec", max_new_tokens=2048)
            return np.asarray(wavs[0], dtype=np.float32), rate
    else:
        sys.path[:0] = [str(COSYVOICE_SRC), str(COSYVOICE_SRC / "third_party/Matcha-TTS")]
        from cosyvoice.cli.cosyvoice import AutoModel
        from cosyvoice.utils.common import set_all_random_seed

        # The card ranks the RL-tuned LLM (llm.rl.pt) above llm.pt, which the loader hardcodes:
        # load the snapshot through a folder of links with llm.pt pointing at the RL weights.
        model_dir = Path(tempfile.mkdtemp(prefix="cosyvoice3-rl-"))
        for f in Path(path).iterdir():
            (model_dir / f.name).symlink_to((Path(path) / ("llm.rl.pt" if f.name == "llm.pt" else f.name)).resolve())
        tts = AutoModel(model_dir=str(model_dir))

        def generate(row, mode, seed):
            set_all_random_seed(seed)
            parts = [o["tts_speech"] for o in tts.inference_cross_lingual(
                "You are a helpful assistant.<|endofprompt|>" + row["text_en"], row["ref"], stream=False)]
            return torch.cat(parts, dim=-1).squeeze(0).float().numpy(), tts.sample_rate
    return generate


def synth_worker(args: argparse.Namespace, mode: str, worker: int) -> None:
    """Voice every args.workers-th row of the mode, from row `worker`, appending to its timing."""
    import soundfile as sf
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 1) // args.workers))
    rows = read_rows(args.root)[: args.limit]
    system = f"{args.model}-{mode}"
    out = args.root / "out" / system
    timing_path = out / "timing.jsonl"
    recorded = {json.loads(line)["id"] for line in timing_path.open()} if timing_path.exists() else set()
    todo = [(k, row) for k, row in enumerate(rows)
            if k % args.workers == worker and row["id"] not in recorded]
    if not todo:
        return
    generate = load_model(args.model)
    # Every worker appends one line per write() to the shared file, which O_APPEND keeps whole.
    with timing_path.open("a") as timing:
        for k, row in todo:
            path = out / f"{row['id']}.wav"
            torch.cuda.synchronize()
            start = time.perf_counter()
            record = {"id": row["id"], "workers": args.workers}
            try:
                wav, rate = generate(row, mode, SEED + k)
                if len(wav) < rate // 2:
                    raise ValueError(f"only {len(wav) / rate:.2f} s of audio")
            except Exception as exc:  # a failed row is a result: it counts as broken
                record["error"] = f"{type(exc).__name__}: {exc}"[:300]
                log(f"{system} {row['id']}: {record['error']}")
            else:
                torch.cuda.synchronize()
                record |= {"seconds": round(time.perf_counter() - start, 3),
                           "audio_s": round(len(wav) / rate, 3)}
                part = path.with_name(path.name + ".part")
                sf.write(part, np.clip(wav, -1, 1), rate, subtype="PCM_16", format="WAV")
                os.replace(part, path)
            record["end"] = round(time.time(), 3)
            timing.write(json.dumps(record) + "\n")
            timing.flush()
    log(f"{system} worker {worker}: done; peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


def synth(args: argparse.Namespace) -> None:
    """Each mode in turn, its rows split over args.workers processes sharing the GPU: one row at a
    time per process, as the models' Python APIs run, so the GPU fills from several."""
    import multiprocessing

    mps = Path(os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")) / "control"
    if args.workers > 1 and not mps.exists():
        raise SystemExit(f"no CUDA MPS daemon ({mps}): start it with `nvidia-cuda-mps-control -d`")
    spawn = multiprocessing.get_context("spawn")  # CUDA can't follow a fork
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # so `kill` reaches the finally below
    for mode in MODELS[args.model][2]:
        system = f"{args.model}-{mode}"
        (args.root / "out" / system).mkdir(parents=True, exist_ok=True)
        log(f"{system}: {args.workers} workers")
        procs = [spawn.Process(target=synth_worker, args=(args, mode, w)) for w in range(args.workers)]
        try:
            for p in procs:
                p.start()
            for p in procs:
                p.join()
        finally:  # a stopped synth stops its workers, which would otherwise hold the GPU
            for p in procs:
                if p.is_alive():
                    p.terminate()
        failed = [w for w, p in enumerate(procs) if p.exitcode]
        if failed:
            raise SystemExit(f"{system}: workers {failed} failed")
        log(f"{system}: done")


# --- mimi --------------------------------------------------------------------------------------


def mimi(args: argparse.Namespace) -> None:
    import sphn
    import torch
    from moshi.models import loaders

    codec = loaders.get_mimi(DEFAULT_MIMI_WEIGHT, device="cuda", num_codebooks=16)
    for src in sorted((args.root / "out").glob("*/*.wav")):
        dst = args.root / "mimi" / src.parent.name / src.name
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        pcm, rate = sphn.read(str(src))
        pcm = sphn.resample(pcm.mean(axis=0), rate, codec.sample_rate)
        with torch.no_grad():
            x = torch.from_numpy(pcm).to("cuda")[None, None]
            y = codec.decode(codec.encode(x))[0, 0, : x.shape[-1]].float().cpu().numpy()
        part = dst.with_name(dst.name + ".part")
        sphn.write_wav(str(part), y, codec.sample_rate)
        os.replace(part, dst)
    log("Mimi round trip done")


# --- score -------------------------------------------------------------------------------------


def score(args: argparse.Namespace) -> None:
    import jiwer
    import soundfile as sf
    import torch
    import torchaudio
    import wespeaker
    from transformers import pipeline

    root = args.root
    rows = {r["id"]: r for r in read_rows(root)}
    systems = sorted(p.name for p in (root / "out").iterdir() if p.is_dir())

    speaker = wespeaker.load_model(snapshot(*WESPEAKER))
    speaker.set_device("cuda")
    cache: dict[str, torch.Tensor] = {}

    def embed(path: str) -> torch.Tensor:
        """WeSpeaker's own extract_embedding, with the features moved to the GPU (its current
        code leaves them on the CPU) and every file read the same way, whatever its sample format."""
        if path not in cache:
            pcm, rate = sf.read(path, dtype="float32", always_2d=True)
            x = torch.from_numpy(pcm.mean(axis=1))[None] * (1 << 15)  # its int16 scale
            x = torchaudio.functional.resample(x, rate, speaker.resample_rate)
            with torch.no_grad():
                out = speaker.model(speaker.compute_features(x, speaker.resample_rate).to("cuda"))
            out = out[-1] if isinstance(out, tuple) else out
            cache[path] = torch.nn.functional.normalize(out[0].float().cpu(), dim=-1)
        return cache[path]

    def cos(a: str, b: str) -> float:
        return float(embed(a) @ embed(b))

    ids = sorted(rows)
    ceiling = {i: cos(rows[i]["clip"], rows[i]["ref"]) for i in ids}
    # Floor: each clip against the next row's clip, another speaker.
    floor = {i: cos(rows[i]["clip"], rows[ids[(k + 1) % len(ids)]]["clip"]) for k, i in enumerate(ids)}

    asr = pipeline("automatic-speech-recognition", model=snapshot(*WHISPER), dtype=torch.float16,
                   device="cuda")
    norm = asr.tokenizer.normalize

    def transcribe(paths: list[str]) -> list[str]:
        audio = []
        for path in paths:
            pcm, rate = sf.read(path, dtype="float32")
            audio.append({"raw": pcm, "sampling_rate": rate})
        # Chunked, so a clip over 30 s is heard whole and still batches with the rest.
        out = asr(audio, batch_size=32, chunk_length_s=30,
                  generate_kwargs={"language": "english", "task": "transcribe"})
        return [o["text"] for o in out]

    table = []
    per_row = []
    for system in systems:
        timing = {}
        for line in (root / "out" / system / "timing.jsonl").open():
            t = json.loads(line)
            timing[t["id"]] = t
        done = [i for i in ids if i in timing and "error" not in timing[i]]
        for kind in ("raw", "mimi"):
            folder = root / ("out" if kind == "raw" else "mimi") / system
            paths = [str(folder / f"{i}.wav") for i in done]
            hyps = transcribe(paths)
            for i, path, hyp in zip(done, paths, hyps, strict=True):
                ref = norm(rows[i]["text_en"])
                wer = jiwer.wer(ref, norm(hyp)) if ref else 0.0
                per_row.append({"system": system, "kind": kind, "id": i, "source": rows[i]["source"],
                                "sim": cos(rows[i]["clip"], path), "ceiling": ceiling[i],
                                "floor": floor[i], "wer": wer, "hyp": hyp,
                                "s_per_char": timing[i]["audio_s"] / max(len(rows[i]["text_en"]), 1)})
        failed = sum("error" in t for t in timing.values())
        voiced = [t for t in timing.values() if "seconds" in t]
        # Wall time from the first row's start to the last row's end, per second of audio: the
        # GPU's cost with every worker running, warm-ups included.
        wall = max(t["end"] for t in voiced) - min(t["end"] - t["seconds"] for t in voiced) if voiced else 0
        table.append({"system": system, "rows": len(done), "failed": failed,
                      "workers": voiced[0]["workers"] if voiced else 0,
                      "rtf": wall / sum(t["audio_s"] for t in voiced) if voiced else float("nan")})

    pace = float(np.median([r["s_per_char"] for r in per_row]))
    for r in per_row:
        r["broken"] = r["wer"] > BROKEN_WER or not 0.5 * pace <= r["s_per_char"] <= 2 * pace
    with (root / "scores.jsonl").open("w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in per_row)

    same = {i for i in ids if ceiling[i] >= SAME_SPEAKER}
    lines = [f"{len(same)} of {len(ids)} rows have a same-speaker reference (cosine >= {SAME_SPEAKER}); "
             f"ceiling {np.mean([ceiling[i] for i in same]):.3f}, floor {np.mean([floor[i] for i in same]):.3f}",
             "",
             "| System | Rows | Failed | Broken (Mimi) | WER raw / Mimi | SIM raw / Mimi | SIM by source (Mimi) | Workers | RTF |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for t in table:
        # WER and the broken share over every row; similarity only where the reference is the speaker.
        def pick(kind: str, only_same: bool) -> list[dict]:
            return [r for r in per_row if r["system"] == t["system"] and r["kind"] == kind
                    and (r["id"] in same or not only_same)]

        raw, mim = pick("raw", False), pick("mimi", False)
        raw_same, mim_same = pick("raw", True), pick("mimi", True)
        by_source = " ".join(f"{s} {np.mean([r['sim'] for r in mim_same if r['source'] == s]):.3f}"
                             for s in ("pho", "vivoice", "giga") if any(r["source"] == s for r in mim_same))
        broken = (sum(r["broken"] for r in mim) + t["failed"]) / max(len(mim) + t["failed"], 1)
        lines.append(f"| {t['system']} | {t['rows']} | {t['failed']} | {broken:.1%} | "
                     f"{np.mean([r['wer'] for r in raw]):.3f} / {np.mean([r['wer'] for r in mim]):.3f} | "
                     f"{np.mean([r['sim'] for r in raw_same]):.3f} / {np.mean([r['sim'] for r in mim_same]):.3f} | "
                     f"{by_source} | {t['workers']} | {t['rtf']:.3f} |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("clips")
    p.add_argument("--pho", type=int, default=100)
    p.add_argument("--vivoice", type=int, default=150)
    p.add_argument("--giga", type=int, default=50)
    p = steps.add_parser("synth")
    p.add_argument("model", choices=sorted(MODELS))
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8, help="processes sharing the GPU through MPS, ~6 GiB each")
    steps.add_parser("mimi")
    steps.add_parser("score")
    args = parser.parse_args()
    {"clips": clips, "synth": synth, "mimi": mimi, "score": score}[args.step](args)


if __name__ == "__main__":
    main()

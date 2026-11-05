#!/usr/bin/env python
"""English TTS for the real-Vietnamese tranche: VoxCPM2 with the reference's transcript (`ult` in the
bake-off), batched on nano-vllm-voxcpm's engine ([tranche-tts](../../docs/agent-memory/tranche-tts.md)).

The engine runs in this process. Its runner returns each step's latent patch and stop flag
without decoding audio: nano-vllm's own streaming VAE decoder loops over every stream in Python in
each causal conv layer, which took ~70% of the engine loop at 256 streams. Each row's audio is
decoded once the row ends, as voxcpm's own non-streaming path decodes it, many rows per batched
pass of the VAE decoder in fp16 under torch.compile (3.4x fp32's speed; 57 dB SNR against fp32,
which voxcpm runs). The VAE is causal throughout, so a batch's right padding leaves every row's
audio as decoded alone. Prompts are encoded in fp32 batches the same way.

Env /workspace/envs/nanovox (training/tranche/envs.sh). From the repo root, with the HF token in
the environment and the ASR server up:
  /workspace/envs/asr/bin/qwen-asr-serve <Qwen3-ASR-1.7B snapshot> --served-model-name asr \
      --gpu-memory-utilization 0.2 --port 8100 --max-num-seqs 256 &
  /workspace/envs/nanovox/bin/python -m training.tranche.voice run pho vivoice giga
      every shard of segments and text_en on the Hub without en_audio yet: each row voiced with its
      own clip and transcript as prompt and reference, heard by the ASR, voiced again while broken
      (WER over 20% or pace off by 2x) up to 3 tries, the best try kept; writes
      <root>/<public|restricted>/<source>/en_audio/<shard> (id, en_audio as 24 kHz Ogg Opus,
      en_duration_s, en_asr, en_wer, en_tries, en_broken) and pushes each source's folder every
      30 min and at its end. A rerun skips the shards already written or pushed.
  /workspace/envs/nanovox/bin/python -m training.tranche.voice bench --concurrency 512 --repeats 8
      the bake-off's rows in its `ult` setup (prompt and reference: the row's reference clip),
      timed; --write saves the first pass as <out>/out/<system>/ in the bake-off's layout, beside a
      link to its clips.jsonl, for `bakeoff mimi --root <out>` and `bakeoff score --root <out>`
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BAKEOFF = Path("/workspace/tranche/bakeoff")
BENCH = Path("/workspace/tranche/voice-bench")
MODEL = ("openbmb/VoxCPM2", "32279ef")  # the bake-off's revision
CFG = 2.0  # voxcpm 2.0.3's generate defaults, which the bake-off ran
TIMESTEPS = 10
MIN_PATCHES = 4  # voxcpm stops no earlier than its fourth patch (`i > min_len`, min_len 2)
MAX_MODEL_LEN = 4096
ENCODE_RATE, OUT_RATE = 16000, 48000
CONTEXT_FRAMES = 12  # the prompt's last 3 patches lead the decode, then are cut, as voxcpm decodes
BATCH_FRAMES = 8_000  # VAE frames per batched encode or decode (~5 min of audio)
BUCKET = 64  # VAE frames (2.56 s)
WARM_FRAMES = 1024  # the widest decode shape warmed at start (41 s; longer rows are rare)
ROW_S = (5.0, 22.4)  # the tranche's row lengths
MAX_WAIT_S = 120
# Verification, as the bake-off scored ([tranche-tts](../../docs/agent-memory/tranche-tts.md)):
WHISPER_NORMALIZER = ("openai/whisper-large-v3", "06f233f")  # its English normalizer only
BROKEN_WER = 0.2
PACE = 0.0636  # the bake-off's median seconds of English audio per character of its line
MAX_TRIES = 3  # attempts per row; the best by WER is kept
ASR_CALLS = 64  # concurrent requests to the ASR server
PUSH_S = 1800
MAX_OPEN = 3  # shards in the engine at once: those finishing their slowest rows, and the next


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T', time.gmtime())} {msg}", flush=True)


def snapshot(repo: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision)


@dataclass
class Request:
    key: str
    text: str  # what to say
    prompt_text: str  # the transcript of the prompt audio
    latents: np.ndarray  # the prompt audio, VAE frames [F, 64]: both voxcpm's reference and its prompt


def build_engine(path: str, max_seqs: int, gpu_memory: float):
    """nano-vllm-voxcpm's VoxCPM2 engine with a runner that decodes no audio."""
    import torch
    from nanovllm_voxcpm.config import Config
    from nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
    from nanovllm_voxcpm.layers.audio_vae_v2 import AudioVAEV2
    from nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine
    from nanovllm_voxcpm.models.voxcpm2.model import VoxCPM2Model
    from nanovllm_voxcpm.models.voxcpm2.runner import VoxCPM2Runner
    from nanovllm_voxcpm.utils.loader import load_model
    from torch.nn.utils import remove_weight_norm

    class Runner(VoxCPM2Runner):
        def init_model(self, model_config, model_path):
            self.model = VoxCPM2Model(model_config, self.inference_timesteps, lora_config=self.lora_config)
            load_model(self.model, model_path)
            torch.set_default_dtype(torch.float32)  # the encoder stays fp32, as voxcpm runs it
            self.vae = AudioVAEV2(config=model_config.audio_vae_config)
            self.vae.load_state_dict(torch.load(os.path.join(model_path, "audiovae.pth"))["state_dict"])
            for module in self.vae.decoder.modules():
                if hasattr(module, "weight_g"):
                    remove_weight_norm(module)
            torch.set_default_dtype(torch.bfloat16)

        def run(self, seqs, is_prefill):
            """VoxCPM2Runner.run up to the model's outputs: every request unseeded, so one draw of
            noise per step serves the batch."""
            inputs = {"positions": self.prepare_prefill_context(seqs) if is_prefill
                      else self.prepare_decode_context(seqs)}
            payloads = [seq.custom_payload for seq in seqs]
            if is_prefill:
                inputs["text_only_prefill"] = not any(p.feat_masks.any() for p in payloads)

            def cuda(arrays):
                return torch.from_numpy(np.concatenate(arrays, axis=0)).cuda(non_blocking=True)

            inputs["text_tokens"] = cuda([p.text_tokens for p in payloads])
            inputs["feat"] = cuda([p.feats for p in payloads]).to(self.dtype)
            inputs["feat_mask"] = cuda([p.feat_masks for p in payloads])
            for name in ("temperature", "cfg_value"):
                inputs[name] = torch.tensor([getattr(p, name) for p in payloads], dtype=torch.float32,
                                            pin_memory=True).cuda(non_blocking=True).to(self.dtype)
            inputs["z_noise"] = torch.randn((len(seqs), self.feat_dim, self.patch_size), dtype=self.dtype,
                                            device="cuda")
            outputs = self.run_model(inputs, is_prefill)
            latents = outputs["latents"].to(torch.float32).cpu().numpy()
            stops = outputs["stop_flag"].cpu().tolist()
            return [{"latents": latents[i], "stop_flag": stops[i]} for i in range(len(seqs))]

    class Engine(VoxCPM2Engine):
        def __init__(self, config):
            # VoxCPM2Engine.__init__, with the runner above
            self.n_decode_pad_frames = 12
            self.feat_dim = config.model_config.feat_dim
            self.patch_size = config.model_config.patch_size
            self.audio_start_token, self.ref_audio_start_token, self.ref_audio_end_token = 101, 103, 104
            self.block_size = config.kvcache_block_size
            self.max_model_len = config.max_model_len
            from nanovllm_voxcpm.models.voxcpm2.utils import mask_multichar_chinese_tokens
            from transformers import LlamaTokenizerFast

            self.tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(config.model))
            LLMEngineBase.__init__(self, Runner, config, config.tensor_parallel_size)

        def postprocess_seq(self, seq, outputs, is_prefill):
            payload = seq.custom_payload
            patch = outputs["latents"]  # [patch, feat]
            seq.append_token(patch.tobytes())
            payload.feats.append(patch[None])
            payload.text_tokens.append(0)
            payload.feat_masks.append(True)
            payload.generated_waveforms.append(patch)  # here the row's patches, decoded when it ends
            n = len(payload.generated_waveforms)
            if (outputs["stop_flag"] == 1 and n >= MIN_PATCHES) or n >= payload.max_generate_length:
                seq.stoped = True

    model_config = VoxCPM2Config.model_validate_json(open(os.path.join(path, "config.json")).read())
    model_config.inference_timesteps = TIMESTEPS
    return Engine(Config(model=path, max_num_batched_tokens=16384, max_num_seqs=max_seqs,
                         max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=gpu_memory,
                         model_config=model_config, devices=[0]))


class Voicer:
    """Voices requests `concurrency` at a time; finished rows come back in batches, as audio."""

    def __init__(self, concurrency: int, gpu_memory: float):
        import torch

        from training.paths import cpu_budget

        # Thread pools default to the host's CPUs (224 on the box), past the container's limit
        # on threads; the tokenizer's reads this when it first runs.
        os.environ.setdefault("RAYON_NUM_THREADS", "4")
        torch.set_num_threads(max(1, int(cpu_budget() // 4)))
        self.torch = torch
        self.concurrency = concurrency
        self.engine = build_engine(snapshot(*MODEL), concurrency, gpu_memory)
        self.vae = self.engine.model_runner.vae
        self.hop = self.vae.encoder_chunk_size  # 640 samples at 16 kHz per frame
        self.align = self.hop * self.engine.patch_size
        self.out_hop = self.vae.decoder_chunk_size  # 1920 samples at 48 kHz per frame
        # Memory-bound depthwise convs and Snake activations at 48 kHz: half the bytes and fused
        # elementwise kernels. One compile serves every batch shape.
        self.decoder = torch.compile(self.vae.decoder.half(), dynamic=True)
        self.active: dict[str, Request] = {}  # by engine sequence id
        self.waiting: dict[int, tuple[float, list[tuple[Request, np.ndarray]]]] = {}  # by decode width
        began = time.perf_counter()
        with torch.inference_mode():  # every shape a run meets, so none is first met mid-run
            for width in range(BUCKET, WARM_FRAMES + 1, BUCKET):
                self.decoder(torch.zeros(shape(width)[0], self.vae.latent_dim, width, device="cuda",
                                         dtype=torch.float16))
                if width * self.hop <= 2 * ROW_S[1] * ENCODE_RATE:
                    self.vae.encoder(torch.zeros(shape(width)[0], 1, width * self.hop, device="cuda"))
        torch.cuda.synchronize()
        log(f"VAE warmed in {time.perf_counter() - began:.0f} s")

    def tokens(self, text: str) -> int:
        return len(self.engine.tokenizer(text))

    def _padded(self, arrays: list[np.ndarray], size: int, width: int, dtype):
        """[size, C, width] on the GPU, row i holding arrays[i] ([C, n]) from the left, zeros
        elsewhere."""
        x = self.torch.zeros(size, arrays[0].shape[0], width, device="cuda", dtype=dtype)
        for i, a in enumerate(arrays):
            x[i, :, : a.shape[1]] = self.torch.from_numpy(a)
        return x

    def encode(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        """16 kHz clips -> their VAE frames [F, 64], each left-padded to whole patches (voxcpm pads a
        prompt on the left)."""
        clips = [np.pad(c, ((-len(c)) % self.align, 0))[None] for c in clips]
        groups: dict[int, list[int]] = {}
        for i, c in enumerate(clips):
            groups.setdefault(shape(c.shape[1] // self.hop)[1], []).append(i)
        out: list[np.ndarray | None] = [None] * len(clips)
        for width, members in groups.items():
            size = shape(width)[0]
            for k in range(0, len(members), size):
                batch = members[k: k + size]
                x = self._padded([clips[i] for i in batch], size, width * self.hop, self.torch.float32)
                with self.torch.inference_mode():
                    z = self.vae.encoder(x)["mu"].transpose(1, 2).cpu().numpy()  # [B, width, 64]
                for j, i in enumerate(batch):
                    out[i] = np.ascontiguousarray(z[j, : clips[i].shape[1] // self.hop])
        return out

    def decode(self, width: int, rows: list[tuple[Request, np.ndarray]]) -> list[tuple[Request, np.ndarray]]:
        """(request, generated patches) whose frames fit `width` -> each with the 48 kHz audio of its
        generated part, decoded after the prompt's last frames as context, a batch at a time."""
        size, out = shape(width)[0], []
        for k in range(0, len(rows), size):
            batch = rows[k: k + size]
            z = [np.concatenate([req.latents[-CONTEXT_FRAMES:], patches.reshape(-1, req.latents.shape[1])]).T
                 for req, patches in batch]
            with self.torch.inference_mode():
                wav = self.decoder(self._padded(z, size, width, self.torch.float16))[:, 0].float().cpu().numpy()
            for i, (req, _) in enumerate(batch):
                out.append((req, wav[i, min(CONTEXT_FRAMES, len(req.latents)) * self.out_hop: z[i].shape[1] * self.out_hop]))
        return out

    @property
    def room(self) -> int:
        return self.concurrency - len(self.active)

    def submit(self, req: Request) -> None:
        text = re.sub(r"\s+", " ", req.text.replace("\n", " "))  # as voxcpm's generate
        seq_id = uuid.uuid4().hex
        # voxcpm's own cap on the patches: 6 per token of the text to say, plus 10
        self.engine.add_request(seq_id=seq_id, target_text=text, prompt_text=req.prompt_text,
                                prompt_latents=req.latents, ref_audio_latents=req.latents,
                                max_generate_length=self.tokens(text) * 6 + 10, temperature=1.0, cfg_value=CFG)
        self.active[seq_id] = req

    def step(self) -> list[tuple[Request, np.ndarray]]:
        """One engine step; returns the rows of every decode shape whose batch filled, or whose
        oldest row has waited MAX_WAIT_S (so a rare shape cannot hold a shard back), with audio. (A
        decode thread on its own CUDA stream gained nothing: the engine's steps keep the GPU busy.)"""
        out = []
        for seq in self.engine.step():
            if seq.is_finished:
                patches = np.stack(seq.custom_payload.generated_waveforms)
                width = shape(CONTEXT_FRAMES + patches.shape[0] * self.engine.patch_size)[1]
                self.waiting.setdefault(width, (time.monotonic(), []))[1].append((self.active.pop(seq.seq_id), patches))
        now = time.monotonic()
        for width, (since, rows) in list(self.waiting.items()):
            if len(rows) >= shape(width)[0] or now - since > MAX_WAIT_S:
                out += self.decode(width, self.waiting.pop(width)[1])
        return out

    def flush(self) -> list[tuple[Request, np.ndarray]]:
        out = []
        for width, (_, rows) in self.waiting.items():
            out += self.decode(width, rows)
        self.waiting.clear()
        return out

    def voice(self, requests: Iterable[Request]) -> Iterator[list[tuple[Request, np.ndarray]]]:
        """Every request, `concurrency` at a time; yields finished rows with their audio."""
        pending = iter(requests)
        while True:
            while self.room and (req := next(pending, None)) is not None:
                self.submit(req)
            if not self.active:
                break
            if done := self.step():
                yield done
        yield self.flush()


def shape(frames: int) -> tuple[int, int]:
    """The batch size and width for VAE inputs of `frames`: widths round up to BUCKET frames and
    each width has one batch size, so cuDNN plans each conv shape once (~0.9 s per new shape)."""
    width = -(-frames // BUCKET) * BUCKET
    return max(1, BATCH_FRAMES // width), width


# --- bench -------------------------------------------------------------------------------------


def load_16k(path: str) -> np.ndarray:
    import librosa

    return librosa.load(path, sr=ENCODE_RATE, mono=True)[0]


def bench(args: argparse.Namespace) -> None:
    import soundfile as sf

    rows = [json.loads(line) for line in open(args.rows / "clips.jsonl")][: args.limit]
    voicer = Voicer(args.concurrency, args.gpu_memory)
    began = time.perf_counter()
    latents = voicer.encode([load_16k(r["ref"]) for r in rows])
    log(f"encoded {len(rows)} prompts in {time.perf_counter() - began:.1f} s")
    out = args.out / "out" / args.system
    out.mkdir(parents=True, exist_ok=True)
    if not (args.out / "clips.jsonl").exists():
        (args.out / "clips.jsonl").symlink_to(args.rows / "clips.jsonl")
    requests = [Request(f"{rep}/{r['id']}", r["text_en"], r["ref_text"], lat)
                for rep in range(args.repeats) for r, lat in zip(rows, latents)]
    run = time.strftime("%Y%m%dT%H%M%S")
    began, start = time.perf_counter(), time.time()
    audio_s, timing = 0.0, []
    for batch in voicer.voice(requests):
        now = time.time()
        for req, wav in batch:
            audio_s += len(wav) / OUT_RATE
            rep, row_id = req.key.split("/")
            if args.write and rep == "0":
                sf.write(out / f"{row_id}.wav", np.clip(wav, -1, 1), OUT_RATE, subtype="PCM_16")
                timing.append({"id": row_id, "run": run, "workers": args.concurrency,
                               "audio_s": round(len(wav) / OUT_RATE, 3), "end": now})
    wall = time.perf_counter() - began
    if args.write:  # the bake-off's timing: one run spanning the whole pass
        with (out / "timing.jsonl").open("w") as f:
            for t in timing:
                f.write(json.dumps(t | {"seconds": round(t["end"] - start, 3)}) + "\n")
    log(f"{args.system}: {len(requests)} rows, {audio_s:.0f} s of audio in {wall:.1f} s: RTF {wall / audio_s:.4f} "
        f"at concurrency {args.concurrency}")


# --- run ---------------------------------------------------------------------------------------


def decode_16k(data: bytes) -> np.ndarray:
    """A segment's Ogg Opus -> mono float32 at 16 kHz, the VAE encoder's rate."""
    import io

    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        stream.codec_context.thread_count = 1  # ffmpeg's default is a thread per host CPU (224)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=ENCODE_RATE)
        chunks = [f.to_ndarray()[0] for frame in container.decode(stream) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray()[0] for f in resampler.resample(None)]
    return np.concatenate(chunks)


@dataclass
class Shard:
    source: str
    name: str  # the segments file's name
    dst: Path  # <root>/<dest>/<source>/en_audio/<name>
    ids: list[str]
    texts: list[str]  # Vietnamese transcripts: the prompts' texts
    texts_en: list[str]
    clips: list[np.ndarray] | None  # 16 kHz, until encoded
    results: dict = None  # id -> the best attempt so far: {"wer", "broken", "asr", "audio", "duration", "tries"}


class Checker:
    """Hears each output with Qwen3-ASR (`qwen-asr-serve`, language forced) and scores it as the
    bake-off scored: Whisper's English normalizer over both sides (the line unbracketed first), WER,
    and the pace against the bake-off's median. Each check also encodes the 24 kHz Ogg Opus kept."""

    def __init__(self, url: str):
        import httpx
        from transformers import WhisperTokenizer

        self.url = url
        self.client = httpx.Client(timeout=300, limits=httpx.Limits(max_connections=ASR_CALLS))
        from huggingface_hub import snapshot_download

        repo, revision = WHISPER_NORMALIZER
        path = snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.txt"])
        self.norm = WhisperTokenizer.from_pretrained(path).normalize

    def check(self, text: str, wav: np.ndarray) -> dict:
        import io

        import httpx
        import jiwer
        import soundfile as sf
        from scipy.signal import resample_poly

        from training.tranche.ingest import encode_opus

        buf = io.BytesIO()
        sf.write(buf, resample_poly(wav, 1, 3).astype(np.float32), ENCODE_RATE, subtype="PCM_16", format="WAV")
        for attempt in range(3):  # the server drops an idle keep-alive connection now and then
            try:
                reply = self.client.post(self.url, files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                                         data={"model": "asr", "language": "en", "to_language": "en"})
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
        reply.raise_for_status()
        asr = reply.json()["text"]
        ref = self.norm(re.sub(r"[()\[\]{}<>]", " ", text))
        wer = jiwer.wer(ref, self.norm(asr)) if ref.strip() else 0.0
        duration = len(wav) / OUT_RATE
        pace = duration / max(len(text), 1)
        broken = wer > BROKEN_WER or not 0.5 * PACE <= pace <= 2 * PACE
        return {"wer": round(wer, 4), "broken": broken, "asr": asr, "duration": round(duration, 3),
                "audio": encode_opus(resample_poly(wav, 1, 2).astype(np.float32), OUT_RATE // 2)}


def load_shard(source: str, name: str, root: Path, pool) -> Shard:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from training.tranche.ingest import REPOS, SOURCES

    dest = SOURCES[source].dest
    get = lambda part: pq.read_table(hf_hub_download(REPOS[dest], f"{source}/{part}/{name}", repo_type="dataset"))  # noqa: E731
    seg = get("segments").select(["id", "text", "audio"])
    en = dict(zip(*get("text_en").select(["id", "text_en"]).to_pydict().values()))
    ids = seg.column("id").to_pylist()
    if missing := [i for i in ids if not en.get(i)]:
        raise SystemExit(f"{source}/{name}: {len(missing)} rows without text_en, e.g. {missing[0]}")
    audio = [a["bytes"] for a in seg.column("audio").to_pylist()]
    clips = list(pool.map(decode_16k, audio))
    return Shard(source, name, root / dest / source / "en_audio" / name, ids, seg.column("text").to_pylist(),
                 [en[i] for i in ids], clips, {})


def write_shard(shard: Shard) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    r = [shard.results[i] for i in shard.ids]
    table = pa.table({
        "id": shard.ids,
        "en_audio": [{"bytes": x["audio"], "path": f"{i.replace('/', '_')}.opus"} for i, x in zip(shard.ids, r)],
        "en_duration_s": [x["duration"] for x in r],
        "en_asr": [x["asr"] for x in r],
        "en_wer": pa.array([x["wer"] for x in r], pa.float32()),
        "en_tries": pa.array([x["tries"] for x in r], pa.int8()),
        "en_broken": [x["broken"] for x in r],
    })
    shard.dst.parent.mkdir(parents=True, exist_ok=True)
    part = shard.dst.with_name(shard.dst.name + ".part")
    pq.write_table(table, part)
    os.replace(part, shard.dst)


def run(args: argparse.Namespace) -> None:
    """Every shard of each source in turn, the next one loading while the engine voices; the engine
    is kept full with retries first, then new rows. A shard is written once every row has a kept
    attempt, and each source's written shards are pushed every PUSH_S and at its end."""
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    from training.paths import cpu_budget
    from training.tranche import ingest

    api = HfApi()
    todo = []
    for source in args.sources:
        repo = ingest.REPOS[ingest.SOURCES[source].dest]
        files = api.list_repo_files(repo, repo_type="dataset")
        done = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/en_audio/")}
        for f in sorted(f for f in files if f.startswith(f"{source}/segments/")):
            name = f.rsplit("/", 1)[1]
            dst = args.root / ingest.SOURCES[source].dest / source / "en_audio" / name
            if name not in done and not dst.exists():
                todo.append((source, name))
    todo = todo[: args.limit]
    log(f"{len(todo)} shards to voice: " + ", ".join(f"{s} {sum(t[0] == s for t in todo)}" for s in args.sources))
    if not todo:
        return

    pool = ThreadPoolExecutor(max(1, int(cpu_budget()) - 6))  # PyAV decodes without the GIL
    loaded: queue.Queue = queue.Queue(maxsize=1)  # one shard ahead

    def loader():
        try:
            for source, name in todo:
                loaded.put(load_shard(source, name, args.root, pool))
        except BaseException as exc:  # raised again in the main thread
            loaded.put(exc)
        loaded.put(None)

    threading.Thread(target=loader, name="shard-loader", daemon=True).start()
    voicer = Voicer(args.concurrency, args.gpu_memory)
    checker = Checker(args.asr)
    checks = ThreadPoolExecutor(ASR_CALLS)
    inflight = set()
    retries: list[Request] = []
    fresh: Iterator[Request] = iter(())
    shards: dict[str, Shard] = {}  # by source/name, until written
    owner: dict[str, Shard] = {}  # row id -> its shard
    remaining: dict[str, int] = {}  # shard -> rows without a kept attempt
    loading_done = False
    left = {s: sum(t[0] == s for t in todo) for s in args.sources}  # shards not yet written
    pushes = ThreadPoolExecutor(1)  # uploads run beside the voicing
    pushed = {s: time.monotonic() for s in args.sources}
    uploads = []
    began, voiced, attempts = time.monotonic(), 0.0, 0

    def push(source: str) -> None:
        pushed[source] = time.monotonic()
        uploads.append(pushes.submit(ingest.push, argparse.Namespace(root=args.root, source=source, part="en_audio")))

    def next_shard() -> bool:
        nonlocal fresh, loading_done
        shard = loaded.get()
        if isinstance(shard, BaseException):
            raise shard
        if shard is None:
            loading_done = True
            return False
        latents = voicer.encode(shard.clips)
        shard.clips = None
        key = f"{shard.source}/{shard.name}"
        shards[key], remaining[key] = shard, len(shard.ids)
        for i in shard.ids:
            owner[i] = shard
        fresh = (Request(i, t_en, t_vi, lat) for i, t_en, t_vi, lat in zip(shard.ids, shard.texts_en, shard.texts, latents))
        return True

    def settle(req: Request, result: dict) -> None:
        nonlocal attempts, voiced
        attempts += 1
        voiced += result["duration"]
        shard = owner[req.key]
        best = shard.results.get(req.key)
        result["tries"] = (best["tries"] if best else 0) + 1
        if best is None or result["wer"] < best["wer"] or (best["broken"] and not result["broken"]):
            shard.results[req.key] = result
        else:
            best["tries"] = result["tries"]
        if result["broken"] and result["tries"] < MAX_TRIES and shard.results[req.key]["broken"]:
            retries.append(req)
            return
        key = f"{shard.source}/{shard.name}"
        remaining[key] -= 1
        if remaining[key] == 0:
            write_shard(shard)
            kept = list(shard.results.values())
            log(f"{key}: {len(kept)} rows, {sum(x['tries'] for x in kept) / len(kept):.2f} tries a row, "
                f"{sum(x['broken'] for x in kept) / len(kept):.1%} still broken; overall RTF "
                f"{(time.monotonic() - began) / voiced:.4f} over {voiced / 3600:.1f} h voiced ({attempts} attempts)")
            for i in shard.ids:
                del owner[i]
            del shards[key], remaining[key]
            left[shard.source] -= 1
            if left[shard.source] == 0 or time.monotonic() - pushed[shard.source] > PUSH_S:
                push(shard.source)

    while True:
        while voicer.room:
            req = retries.pop() if retries else next(fresh, None)
            if req is None:
                if loading_done or len(shards) >= MAX_OPEN or not next_shard():
                    break  # a few shards open: those finishing and the next
                continue
            voicer.submit(req)
        out = voicer.step() if voicer.active else voicer.flush()
        for req, wav in out:
            inflight.add(checks.submit(lambda r=req, w=wav: (r, checker.check(r.text, w))))
        for fut in [f for f in inflight if f.done()]:
            inflight.discard(fut)
            settle(*fut.result())
        for fut in [f for f in uploads if f.done()]:
            uploads.remove(fut)
            fut.result()  # a failed upload stops the run; a rerun pushes what is written
        if loading_done and not voicer.active and not voicer.waiting and not inflight and not retries:
            break
        if not voicer.active and not out:
            time.sleep(0.05)  # waiting on checks: their retries refill the engine
    checks.shutdown()
    pool.shutdown()
    pushes.shutdown()
    for fut in uploads:
        fut.result()
    log(f"done: {voiced / 3600:.1f} h voiced in {(time.monotonic() - began) / 3600:.2f} h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("bench")
    p.add_argument("--rows", type=Path, default=BAKEOFF, help="a folder with the bake-off's clips.jsonl")
    p.add_argument("--out", type=Path, default=BENCH)
    p.add_argument("--system", default="voxcpm2-batched")
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--gpu-memory", type=float, default=0.5)
    p.add_argument("--write", action="store_true")
    p = steps.add_parser("run")
    p.add_argument("sources", nargs="+", choices=["pho", "vivoice", "giga"])
    p.add_argument("--root", type=Path, default=Path("/workspace/tranche/repo"))
    p.add_argument("--asr", default="http://localhost:8100/v1/audio/transcriptions")
    p.add_argument("--concurrency", type=int, default=512)
    p.add_argument("--gpu-memory", type=float, default=0.45)
    p.add_argument("--limit", type=int, help="voice only this many shards (a staged start)")
    args = parser.parse_args()
    {"bench": bench, "run": run}[args.step](args)


if __name__ == "__main__":
    main()

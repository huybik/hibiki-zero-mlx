#!/usr/bin/env python
"""English TTS for the real-Vietnamese tranche: VoxCPM2 with the reference's transcript (`ult` in the
bake-off), batched on nano-vllm-voxcpm's engine ([tranche-tts](../../docs/agent-memory/tranche-tts.md)).

The engine runs in this process. Its runner returns each step's latent patch and stop flag
without decoding audio: nano-vllm's own streaming VAE decoder loops over every stream in Python in
each causal conv layer, which took ~70% of the engine loop at 256 streams. Each row's audio is
decoded once the row ends, as voxcpm's own non-streaming path decodes it, many rows per batched
pass of the VAE decoder in fp16 under torch.compile (3.4x fp32's speed; 57 dB SNR against fp32,
which voxcpm runs). The VAE is causal throughout, so a batch's right padding leaves every row's
audio as decoded alone. Prompts are encoded in fp16 batches the same way.

Env /workspace/envs/nanovox (training/tranche/envs.sh). From the repo root, with HF_TOKEN_AN (pushes) in
the environment and the ASR server up:
  /workspace/envs/asr/bin/qwen-asr-serve <Qwen3-ASR-1.7B snapshot> --served-model-name asr \
      --gpu-memory-utilization 0.2 --port 8100 --max-num-seqs 256 --api-server-count 8 &
      (one API process decodes and featurizes ~11 requests/s on the CPU, below the TTS's ~13 attempts/s)
  /workspace/envs/nanovox/bin/python -m training.tranche.voice run pho vivoice giga
      every segments shard (each has its text_en and asr_vi) on the Hub without en_audio yet, but
      those held out for evaluation (`ingest.SOURCES[...].heldout`): each
      row voiced with its own clip and transcript as prompt and reference, heard by the ASR, voiced
      again while broken (WER over 20% or pace off by 2x) up to MAX_TRIES, the best try kept; writes
      <root>/<source>/en_audio/<shard> (id, en_audio as Ogg Opus from 24 kHz, decoded at 48 kHz,
      en_duration_s, en_asr, en_wer, en_tries, en_broken) and pushes each source's folder every
      30 min and at its end. Rows the cache build drops on the Vietnamese side (`filters.vi_gate`
      on asr_vi) are not voiced: en_tries 0, the other columns null. A rerun skips the shards
      already written or pushed, and first pushes any written but not yet pushed.
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
# Flow-matching steps: 8 from 2026-11-08 (tranche-tts.md, "Quality against speed": the diffusion head is
# ~92% of a decode step; 8 was within noise of 10 on broken share, kept-row WER and voice match, for 8.5%
# less GPU per kept row; voxcpm runs 10)
TIMESTEPS = 8
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
# Attempts per row, the best by WER kept. 1 from the scale-up on (2026-11-06): on GigaSpeech 2 a retry
# costs more TTS time than its rows give back (tranche-tts.md); the first tranche ran 3.
MAX_TRIES = 1
ASR_TIMEOUT_S = 60  # a request takes well under a second
# The server stalls past a try now and then and recovers within minutes: 3 tries ended a run three
# times in two days (2026-11-06, 11-07 17:22 and 23:17 UTC+7), each losing the shards in flight.
ASR_TRIES = 10
ASR_MAX_TOKENS = 512  # as checks.py: a looping transcript comes back long and scores broken
CHECKERS = 32  # checking processes, each with one request to the ASR server at a time
DECODER_SHARE = 4  # processes decoding the next shard's clips: a quarter of the CPU quota (6 of 26)
OPUS_AT_ONCE = 4
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
            torch.set_default_dtype(torch.float32)  # weights folded in fp32, then halved by Voicer
            self.vae = AudioVAEV2(config=model_config.audio_vae_config)
            self.vae.load_state_dict(torch.load(os.path.join(model_path, "audiovae.pth"))["state_dict"])
            for module in self.vae.modules():
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
        # Memory-bound depthwise convs and Snake activations: half the bytes and fused elementwise
        # kernels. A graph per batch shape (each width has one), all warmed below: static shapes
        # run the decoder 24% and the encoder 35% faster than one dynamic graph. The fp16 encoder's
        # latents are 51 dB from fp32's; the quality test saw no change (tranche-tts.md, "Speed").
        torch._dynamo.config.recompile_limit = 64  # the default 8 would fall back to eager
        self.decoder = torch.compile(self.vae.decoder.half(), dynamic=False)
        self.encoder = torch.compile(self.vae.encoder.half(), dynamic=False)
        self.active: dict[str, Request] = {}  # by engine sequence id
        self.waiting: dict[int, tuple[float, list[tuple[Request, np.ndarray]]]] = {}  # by decode width
        began = time.perf_counter()
        # Every shape a run meets, so none compiles mid-run: inputs made in inference mode, as the
        # runs make theirs (Dynamo's guards tell inference tensors from others).
        with torch.inference_mode():
            for width in range(BUCKET, WARM_FRAMES + 1, BUCKET):
                self.decoder(torch.zeros(shape(width)[0], self.vae.latent_dim, width, device="cuda",
                                         dtype=torch.float16))
                if (width - BUCKET) * self.hop < ROW_S[1] * ENCODE_RATE:  # the prompts: rows' clips
                    self.encoder(torch.zeros(shape(width)[0], 1, width * self.hop, device="cuda", dtype=torch.float16))
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
                with self.torch.inference_mode():
                    x = self._padded([clips[i] for i in batch], size, width * self.hop, self.torch.float16)
                    z = self.encoder(x)["mu"].float().transpose(1, 2).cpu().numpy()  # [B, width, 64]
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
    """A segment's Ogg Opus -> mono float32 at 16 kHz, the VAE encoder's rate. Resampled by scipy:
    PyAV's AudioResampler builds an ffmpeg filter graph whose threads (one per host CPU) pile up."""
    import io

    import av
    from scipy.signal import resample_poly

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        stream.codec_context.thread_count = 1  # ffmpeg's default is a thread per host CPU (224)
        frames = [f.to_ndarray().astype(np.float32) for f in container.decode(stream)]
        rate = stream.codec_context.sample_rate
    pcm = np.concatenate(frames, axis=-1)
    pcm = pcm.mean(axis=0) if pcm.ndim == 2 and frames[0].shape[0] > 1 else pcm.reshape(-1)
    g = np.gcd(ENCODE_RATE, rate)
    return resample_poly(pcm, ENCODE_RATE // g, rate // g).astype(np.float32)


@dataclass
class Shard:
    source: str
    name: str  # the segments file's name
    dst: Path  # <root>/<source>/en_audio/<name>
    ids: list[str]
    texts: list[str]  # Vietnamese transcripts: the prompts' texts
    texts_en: list[str]
    clips: list[np.ndarray] | None  # the voiced rows' clips at 16 kHz, until encoded
    results: dict = None  # id -> the best attempt so far: {"wer", "broken", "asr", "audio", "duration", "tries"}
    skipped: set[str] = None  # rows voice.py skips: those filters.vi_gate drops or missing text_en; not voiced, their English columns null


class Checker:
    """Hears each output with Qwen3-ASR (`qwen-asr-serve`, language forced) and scores it as the
    bake-off scored: Whisper's English normalizer over both sides (the line unbracketed first), WER,
    and the pace against the bake-off's median. Each check also encodes the 24 kHz Ogg Opus kept."""

    def __init__(self, url: str, encoding):
        import httpx
        from transformers import WhisperTokenizer

        self.url = url
        self.client = httpx.Client(timeout=ASR_TIMEOUT_S)
        # PyAV's encoder resamples into libopus's 20 ms frames through an ffmpeg filter graph that
        # holds a thread per host CPU (224 on the box) while it lives: a semaphore shared by the
        # checking processes keeps a few encodes at a time, under the container's thread limit.
        self.encoding = encoding
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
        for attempt in range(ASR_TRIES):  # it also drops an idle keep-alive connection now and then
            try:
                reply = self.client.post(self.url, files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                                         data={"model": "asr", "language": "en", "to_language": "en",
                                               "max_completion_tokens": ASR_MAX_TOKENS})
            except httpx.TransportError:
                if attempt == ASR_TRIES - 1:
                    raise
                time.sleep(10)  # a refused connection (the server restarting) fails at once
                continue
            reply.raise_for_status()
            asr = reply.json()["text"]
            break
        ref = self.norm(re.sub(r"[()\[\]{}<>]", " ", text))
        wer = jiwer.wer(ref, self.norm(asr)) if ref.strip() else 0.0
        duration = len(wav) / OUT_RATE
        pace = duration / max(len(text), 1)
        broken = wer > BROKEN_WER or not 0.5 * PACE <= pace <= 2 * PACE
        pcm = resample_poly(wav, 1, 2).astype(np.float32)
        with self.encoding:
            audio = encode_opus(pcm, OUT_RATE // 2)
        return {"wer": round(wer, 4), "broken": broken, "asr": asr, "duration": round(duration, 3), "audio": audio}


_checker: Checker | None = None


def _start_checker(url: str, encoding) -> None:
    global _checker
    _checker = Checker(url, encoding)


def _check(text: str, wav: np.ndarray) -> dict:
    return _checker.check(text, wav)


def load_shard(source: str, name: str, root: Path, pool) -> Shard:
    import pyarrow.parquet as pq

    from training.tranche.filters import vi_gate
    from training.tranche.ingest import part_file

    get = lambda part: pq.read_table(part_file(root, source, part, name))  # noqa: E731
    seg = get("segments").select(["id", "text", "audio"])
    en = dict(zip(*get("text_en").select(["id", "text_en"]).to_pydict().values()))
    ids, texts = seg.column("id").to_pylist(), seg.column("text").to_pylist()
    no_en = {i for i in ids if not en.get(i)}
    if no_en:
        log(f"{source}/{name}: {len(no_en)} rows without text_en, e.g. {next(x for x in ids if x in no_en)}")
    vi = get("asr_vi").select(["id", "asr_vi", "language", "cer"]).to_pylist()
    if [r["id"] for r in vi] != ids:
        raise SystemExit(f"{source}/{name}: its asr_vi does not hold the segments' rows in their order")
    # Skipped: rows missing text_en (left null in en_audio) and those the Vietnamese gates drop.
    # The cache build rejects the vi_gate rows as "not voiced" and the no_en rows gracefully.
    skipped = no_en | {i for i, text, r in zip(ids, texts, vi) if vi_gate(r | {"text": text})}
    audio = [a["bytes"] for i, a in zip(ids, seg.column("audio").to_pylist()) if i not in skipped]
    clips = list(pool.map(decode_16k, audio, chunksize=32))
    return Shard(source, name, root / source / "en_audio" / name, ids, texts, [en.get(i, "") for i in ids], clips, {}, skipped)


def write_shard(shard: Shard) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    r = [shard.results.get(i) for i in shard.ids]  # None: skipped (no text_en or Vietnamese gates)
    col = lambda f, kind: pa.array([None if x is None else f(x) for x in r], kind)
    table = pa.table({
        "id": shard.ids,
        "en_audio": pa.array([None if x is None else {"bytes": x["audio"], "path": f"{i.replace('/', '_')}.opus"}
                              for i, x in zip(shard.ids, r)], pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        "en_duration_s": col(lambda x: x["duration"], pa.float64()),
        "en_asr": col(lambda x: x["asr"], pa.string()),
        "en_wer": col(lambda x: x["wer"], pa.float32()),
        "en_tries": pa.array([0 if x is None else x["tries"] for x in r], pa.int8()),
        "en_broken": col(lambda x: x["broken"], pa.bool_()),
    })
    shard.dst.parent.mkdir(parents=True, exist_ok=True)
    part = shard.dst.with_name(shard.dst.name + ".part")
    pq.write_table(table, part)
    os.replace(part, shard.dst)


def run(args: argparse.Namespace) -> None:
    """Every shard of each source in turn, the next one loading while the engine voices; the engine
    is kept full with retries first, then new rows. A shard is written once every voiced row has a
    kept attempt, and each source's written shards are pushed every PUSH_S and at its end."""
    import queue
    import threading
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from huggingface_hub import HfApi

    from training.paths import cpu_budget
    from training.tranche import ingest

    import pyarrow.parquet as pq

    ingest.an_token()
    api = HfApi()
    todo, waiting = [], 0
    for source in args.sources:
        files = api.list_repo_files(ingest.REPO, repo_type="dataset")
        done = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/en_audio/")}
        # A shard waits for its translation, pushed or still only here (translate runs beside voice),
        # and while a row of it has none (translate's next pass fills it).
        translated = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/text_en/")}
        for p in (args.root / source / "text_en").glob("*.parquet"):
            if pq.read_table(p, columns=["text_en"]).column("text_en").null_count:
                translated.discard(p.name)
            else:
                translated.add(p.name)
        # and for its Vietnamese ASR (filters.py, running ahead), whose gates skip rows.
        heard = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/asr_vi/")}
        heard |= {p.name for p in (args.root / source / "asr_vi").glob("*.parquet")}
        for f in sorted(f for f in files if f.startswith(f"{source}/segments/")):
            name = f.rsplit("/", 1)[1]
            if name in ingest.SOURCES[source].heldout:
                continue  # an evaluation set's shard: never trained on, so never voiced
            dst = args.root / source / "en_audio" / name
            if name in translated and name not in done and not dst.exists():
                if name in heard:
                    todo.append((source, name))
                else:
                    waiting += 1  # filters.py has not heard it yet: a later run takes it
        local = {p.name for p in (args.root / source / "en_audio").glob("*.parquet")}
        if local - done:  # written by an earlier run that stopped before pushing them
            ingest.push_part(args.root, source, "en_audio")
    todo = todo[: args.limit]
    log(f"{len(todo)} shards to voice: " + ", ".join(f"{s} {sum(t[0] == s for t in todo)}" for s in args.sources)
        + f"; {waiting} translated but waiting for their asr_vi")
    if not todo:
        return

    # Decoding and checking run in processes: as threads, their Python (PyAV's frame loop, the
    # normalizer, scipy, Opus) held the GIL from the engine loop and starved the GPU. Their thread
    # pools stay at one thread each.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    spawn = get_context("spawn")
    pool = ProcessPoolExecutor(max(1, int(cpu_budget() // DECODER_SHARE)), mp_context=spawn)
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
    checks = ProcessPoolExecutor(CHECKERS, mp_context=spawn, initializer=_start_checker,
                                 initargs=(args.asr, spawn.BoundedSemaphore(OPUS_AT_ONCE)))
    checking: dict = {}  # future -> its request
    inflight = set()
    retries: list[Request] = []
    fresh: Iterator[Request] = iter(())
    shards: dict[str, Shard] = {}  # by source/name, until written
    owner: dict[str, Shard] = {}  # row id -> its shard
    remaining: dict[str, int] = {}  # shard -> rows without a kept attempt
    loading_done = False
    left = {s: sum(t[0] == s for t in todo) for s in args.sources}  # shards not yet written
    pushes = ingest.Pushes(args.root, "en_audio")  # uploads run beside the voicing
    pushed = {s: time.monotonic() for s in args.sources}
    began, voiced_s, attempts = time.monotonic(), 0.0, 0

    def push(source: str) -> None:
        pushed[source] = time.monotonic()
        pushes.push(source)

    def next_shard() -> bool:
        nonlocal fresh, loading_done
        shard = loaded.get()
        if isinstance(shard, BaseException):
            raise shard
        if shard is None:
            loading_done = True
            return False
        began_encode = time.monotonic()
        latents = voicer.encode(shard.clips)
        shard.clips = None
        key = f"{shard.source}/{shard.name}"
        log(f"{key}: encoded {len(latents)} prompts in {time.monotonic() - began_encode:.0f} s")
        rows = [(i, t_en, t_vi) for i, t_en, t_vi in zip(shard.ids, shard.texts_en, shard.texts) if i not in shard.skipped]
        shards[key], remaining[key] = shard, len(rows)
        for i, _, _ in rows:
            owner[i] = shard
        fresh = (Request(i, t_en, t_vi, lat) for (i, t_en, t_vi), lat in zip(rows, latents, strict=True))
        if not rows:
            finish(shard)
        return True

    def finish(shard: Shard) -> None:
        """Writes a shard whose voiced rows all have a kept attempt, and pushes as due."""
        key = f"{shard.source}/{shard.name}"
        write_shard(shard)
        kept = list(shard.results.values())
        log(f"{key}: {len(shard.ids)} rows, {len(shard.skipped)} skipped by voice.py (Vietnamese gates or missing text_en), "
            f"{sum(x['tries'] for x in kept) / max(len(kept), 1):.2f} tries a row, "
            f"{sum(x['broken'] for x in kept) / max(len(kept), 1):.1%} still broken; overall RTF "
            f"{(time.monotonic() - began) / max(voiced_s, 1):.4f} over {voiced_s / 3600:.1f} h voiced ({attempts} attempts)")
        for i in shard.results:
            del owner[i]
        del shards[key], remaining[key]
        left[shard.source] -= 1
        if left[shard.source] == 0 or time.monotonic() - pushed[shard.source] > PUSH_S:
            push(shard.source)

    def settle(req: Request, result: dict) -> None:
        nonlocal attempts, voiced_s
        attempts += 1
        voiced_s += result["duration"]
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
            finish(shard)

    try:
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
                fut = checks.submit(_check, req.text, wav)
                checking[fut] = req
                inflight.add(fut)
            for fut in [f for f in inflight if f.done()]:
                inflight.discard(fut)
                settle(checking.pop(fut), fut.result())
            pushes.check()  # a failed upload stops the run; a rerun pushes what is written
            if loading_done and not voicer.active and not voicer.waiting and not inflight and not retries:
                break
            if not voicer.active and not out:
                time.sleep(0.05)  # waiting on checks: their retries refill the engine
    finally:  # after a crash, queued checks are dropped, not run, so the GPU frees promptly
        checks.shutdown(wait=False, cancel_futures=True)
        pool.shutdown(wait=False, cancel_futures=True)
    pushes.wait()
    log(f"done: {voiced_s / 3600:.1f} h voiced in {(time.monotonic() - began) / 3600:.2f} h")


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

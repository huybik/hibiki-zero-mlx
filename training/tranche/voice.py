#!/usr/bin/env python
"""English TTS for the real-Vietnamese tranche: VieNeu-TTS v3 Turbo in each row's own voice, its SDK's steps
batched on the GPU ([tranche-tts-scaleup](../../docs/agent-memory/tranche-tts-scaleup.md#optimizing-vieneu-and-voxcpm2)).

Per row, as vieneu 3.7.1's `generate_batch` after `_enroll_reference(denoise=True, use_ref_codes=True)`:
- the reference: the row's clip trimmed of silence at 30 dB, its first 8 s, cut to whole quarter seconds so the
  clips batch at 32 lengths, at 44.1 kHz (the loader's processes, a shard ahead); on the GPU, ENROL_PIECE rows at a
  time as the voicer takes them, shortest first: the resemble-enhance denoiser (onnx_graph.py, fp16, compiled per
  length; clean references skip it), the speaker encoder (onnx_graph.py) on a Kaldi fbank of the clean clip at
  16 kHz, and MOSS's codes of it at 48 kHz (fp16);
- the English line in chunks of at most 256 characters, phonemized (the loader's processes);
- each chunk generated in one of SLOTS continuous-batching slots (vieneu_frame.py) and made again while shorter
  than the SDK's minimum (twice at most, the longest kept); decoded by MOSS in fp16, in batches of chunks of
  similar length; made again while the SDK's babble guard suspects it (twice at most, its preference kept);
- the chunks joined with the SDK's pauses, at 48 kHz.

Env /workspace/envs/vieneu (training/tranche/envs.sh). From the repo root, with HF_TOKEN_AN (pushes) in the
environment and the ASR server up (`restart.sh asr`):
  /workspace/envs/vieneu/bin/python -m training.tranche.voice run yodas
      every segments shard (each has its text_en and asr_vi) on the Hub without en_audio yet, but
      those held out for evaluation (`ingest.SOURCES[...].heldout`): each
      row voiced from its own clip, heard by the ASR, voiced again while broken (WER over 20% or pace
      off by 2x) up to MAX_TRIES, the best try kept; writes
      <root>/<source>/en_audio/<shard> (id, en_audio as Ogg Opus from 24 kHz, decoded at 48 kHz,
      en_duration_s, en_asr, en_wer, en_tries, en_broken) and pushes each source's folder every
      30 min and at its end. Rows the cache build drops on the Vietnamese side (`filters.vi_gate`
      on asr_vi), and those whose text_en is missing or has nothing speakable, are not voiced:
      en_tries 0, the other columns null. A rerun skips the shards
      already written or pushed, and first pushes any written but not yet pushed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import types
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL = ("pnnbao-ump/VieNeu-TTS-v3-Turbo", "61b85e3d937fbbacb387714180e8182823512523")
MODEL_FILES = ["config.json", "update/*", "denoiser.onnx", "speaker_encoder.onnx"]  # the PyTorch path's
CODEC = ("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano", "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68")
ASR_RATE, DENOISE_RATE, SPEAKER_RATE, OUT_RATE = 16000, 44100, 16000, 48000
# The reference, as _enroll_reference cuts it, then to whole BUCKET_S: equal lengths batch, and each length is one
# compiled denoiser graph (32 of them)
TRIM_DB, REF_S, BUCKET_S = 30, 8.0, 0.25
DENOISE_BATCH = 16  # a batch padded to it with copies of its last clip (instance norms keep clips apart)
# References at least this clean (`level_spread`) skip the denoiser: YODAS's cleanest ~30%. On the bench's paired runs
# with and without the denoiser, those rows broke no more undenoised (5.5% against 5.7%) and matched the voice a
# little better; past ~40% skipped, broken rows grow. Without the denoiser at all, 11.9% broke against 6.8%.
CLEAN_SPREAD_DB = 40.0
# Inductor's default: the first compile of the 32 lengths takes ~14 min, later runs of the same code read its cache
# (/tmp/torchinductor_root) in about a minute. max-autotune-no-cudagraphs ran the bench 4% faster but compiles ~54 min
# cold, and this module's first run on the box missed the bench's cache.
DENOISE_MODE = "default"
ENROL_BATCH = 64  # references per speaker-encoder and codec batch
# Rows enrolled at once, as the voicer takes them (a multiple of both batches): their denoised, 16 kHz and 48 kHz
# copies (~3.4 MB a row) live for a piece. A whole shard at once peaked at ~59 GB, which glibc then kept in the
# process, and paused the frames and the ASR for 131-156 s.
ENROL_PIECE = 256
CHUNK_CHARS = 256  # the SDK's chunk length
SAMPLING = dict(temperature=0.8, top_k=25, top_p=0.95, repetition_penalty=1.2)  # generate_batch's defaults
# A chunk's frame cap: max_expected_frames of its phonemes up to MAX_NEW_FRAMES (the bench's; the SDK's default is
# 300), and in a slot also V3TurboBatchEngine.FUSED_MAX_FRAMES (512, 41 s) and the cache's room after the prompt
MAX_NEW_FRAMES = 1000
SHORT_RETRIES = 2  # generate_batch's max_retries
SLOTS = 512  # 3% faster than 256 on the bench, 4% than 768
ADMIT_MIN = 32  # free slots that make a prefill (in groups of PREFILL)
PREFILL = 64
QUEUE = 512  # chunks waiting for a slot, their prompt embeddings built ahead on a thread
DECODE_POOL = 512  # chunks decoded together, sorted by length into batches of DECODE_BATCH
DECODE_BATCH = 64
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
LOADER_SHARE = 4  # processes preparing the next shard's references and lines: a quarter of the CPU quota
OPUS_AT_ONCE = 4
PUSH_S = 1800
MAX_OPEN = 3  # shards in the voicer at once: those finishing their slowest rows, and the next


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T', time.gmtime())} {msg}", flush=True)


def snapshot(repo: str, revision: str, **kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, **kwargs)


# --- the loader's processes --------------------------------------------------------------------


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """A segment's Ogg Opus -> mono float32 at its own rate. Without PyAV's AudioResampler, whose ffmpeg filter
    graph holds a thread per host CPU (they pile up)."""
    import io

    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.audio[0]
        stream.codec_context.thread_count = 1  # ffmpeg's default is a thread per host CPU (224)
        frames = [f.to_ndarray().astype(np.float32) for f in container.decode(stream)]
        rate = stream.codec_context.sample_rate
    pcm = np.concatenate(frames, axis=-1)
    return (pcm.mean(axis=0) if pcm.ndim == 2 and frames[0].shape[0] > 1 else pcm.reshape(-1)), rate


def decode_16k(data: bytes) -> np.ndarray:
    """A segment's Ogg Opus -> mono float32 at 16 kHz, the ASR server's rate (filters.py), resampled by scipy."""
    from scipy.signal import resample_poly

    pcm, rate = decode(data)
    g = np.gcd(ASR_RATE, rate)
    return resample_poly(pcm, ASR_RATE // g, rate // g).astype(np.float32)


def reference(data: bytes) -> np.ndarray:
    """A segment's Ogg Opus -> its reference before the denoiser: silence trimmed at TRIM_DB, the first REF_S, cut to
    whole BUCKET_S (one under a bucket padded with silence to one: a new length would compile mid-run, and under 25 ms
    the speaker's fbank has no frame), at 44.1 kHz by soxr (the SDK's resampler)."""
    import librosa
    import soxr

    pcm, rate = decode(data)
    w = librosa.effects.trim(pcm, top_db=TRIM_DB)[0][: int(REF_S * rate)]
    q = int(BUCKET_S * rate)
    w = w[: len(w) // q * q] if len(w) >= q else np.pad(w, (0, q - len(w)))
    return soxr.resample(w, rate, DENOISE_RATE).astype(np.float32)


def level_spread(x):
    """Equal-length 44.1 kHz clips [B, N] on the GPU -> each one's spread between its loud and quiet 20 ms frames,
    dB (the 95th minus the 10th percentile of their RMS levels): a clean clip's pauses sit far below its speech."""
    import torch

    hop = DENOISE_RATE // 50
    frames = x[:, : x.shape[1] // hop * hop].unflatten(1, (-1, hop))
    db = 10 * torch.log10(frames.float().square().mean(-1) + 1e-12)
    lo, hi = torch.quantile(db, torch.tensor([0.10, 0.95], device=x.device), dim=1)
    return hi - lo


def kaldi_mel(bins: int, fft: int, rate: int) -> np.ndarray:
    """Kaldi's triangular mel filters, 20 Hz to Nyquist, over an rfft's fft // 2 + 1 bins (the last, Nyquist, unused):
    [bins, fft // 2 + 1]."""
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
    lo, hi = mel(20.0), mel(rate / 2)
    delta = (hi - lo) / (bins + 1)
    at = mel(np.arange(fft // 2) * rate / fft)[None]
    left = lo + np.arange(bins)[:, None] * delta
    w = np.maximum(0.0, np.minimum((at - left) / delta, (left + 2 * delta - at) / delta))
    return np.pad(w, ((0, 0), (0, 1))).astype(np.float32)


def speaker_fbank(w16):
    """The SDK's extract_speaker_fbank (kaldi-native-fbank: 80 mel bins of 25 ms povey frames every 10 ms, edges
    snipped, DC removed, pre-emphasis 0.97, power spectrum, log floored at float32's epsilon, mean-normalised over
    time) of equal-length 16 kHz clips [B, N] on the GPU: [B, frames, 80]."""
    import torch

    n, hop, fft = SPEAKER_RATE // 40, SPEAKER_RATE // 100, 512
    f = w16.float().unfold(1, n, hop)
    f = f - f.mean(-1, keepdim=True)
    f = torch.cat([f[..., :1] * (1 - 0.97), f[..., 1:] - 0.97 * f[..., :-1]], -1)
    f = f * torch.hann_window(n, periodic=False, device=f.device).pow(0.85)
    power = torch.fft.rfft(f, n=fft).abs().pow(2)
    mel = power @ torch.from_numpy(kaldi_mel(80, fft, SPEAKER_RATE)).to(f.device).T
    mel = mel.clamp(min=torch.finfo(torch.float32).eps).log()
    return mel - mel.mean(1, keepdim=True)


def window_attention(self, x, input_lengths):
    """MOSS's non-streaming attention (`MossAudioTokenizerMultiheadAttention._forward_non_streaming_sdpa`) as the
    causal sliding window it is, through flex_attention: the tiles outside the window are skipped, where the SDK's
    dense T x T mask computed them all. Batches are padded only on the right and every layer is causal, so the
    padding never reaches a real frame and `input_lengths` is not needed."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    B, T, _ = x.shape
    q, k, v = self._project_qkv(x)
    q, k = self._apply_dense_rope(q, k)
    W = self.context
    mask = create_block_mask(lambda b, h, i, j: (i >= j) & (i - j < W), None, None, T, T, device=x.device)
    return flex_attention(q, k, v, block_mask=mask).transpose(1, 2).reshape(B, T, self.embed_dim)


def phonemize(text: str) -> tuple[list[str], list]:
    """An English line -> its chunks phonemized and the SDK's pauses between them; no chunk when nothing in it is
    speakable (the row is then skipped)."""
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions

    chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=CHUNK_CHARS)
    return [phonemize_text_with_emotions(c) for c in chunks], gaps


# --- the voicer --------------------------------------------------------------------------------


@dataclass(eq=False)
class Request:
    key: str
    text: str  # the English line, which the check hears against
    chunks: list[str]  # its phonemized chunks
    gaps: list  # the pauses between them
    speaker: np.ndarray  # the row's speaker embedding
    codes: np.ndarray  # its reference codes [T, n_vq]


@dataclass(eq=False)
class Chunk:
    req: Request
    k: int  # its place in the line
    cap: int  # generate_batch's frame cap for it, as babble_suspect takes it
    min_frames: int  # shorter, it is made again
    embed: object = None  # its prompt embedding's future, until it takes a slot
    codes: object = None  # the longest generation so far [T, n_vq] on the GPU, until decoded
    tries: int = 0  # generations before its first decode
    wav: np.ndarray | None = None  # the audio kept
    verdict: tuple | None = None  # babble_suspect's on it
    rerolls: int = 0  # generations the babble guard asked for


class Voicer:
    """Rows in, their 48 kHz audio out: `enrol` a shard's references, `submit` its rows while there is `room`, `step`
    until each comes back."""

    def __init__(self):
        import torch
        from concurrent.futures import ThreadPoolExecutor

        from vieneu._v3_turbo_engine import VieNeuTTSv3Turbo
        from vieneu._v3_turbo_engine.onnx_denoiser import N_FFT
        from vieneu._v3_turbo_engine.rep_history import DEFAULT_REP_WINDOW
        from vieneu.v3_turbo_serve import V3TurboBatchEngine

        from training.paths import cpu_budget
        from training.tranche.onnx_graph import OnnxGraph
        from training.tranche.vieneu_frame import ContinuousFrame

        began = time.perf_counter()
        torch.set_num_threads(max(1, int(cpu_budget() // 4)))
        path = snapshot(*MODEL, allow_patterns=MODEL_FILES)
        tts = VieNeuTTSv3Turbo(checkpoint_path=path, model_subfolder="update", moss_tokenizer_path=snapshot(*CODEC),
                               device="cuda")
        tts.speaker_encoder = tts.denoiser = None  # its onnxruntime sessions, each with a thread per core: run here on the GPU
        self.codec = tts.audio_tokenizer
        self.codec.set_compute_dtype("fp16")  # encode and decode: 59 dB from fp32 on the bench
        # Encode and decode whole (no streaming chunks), their attention windowed (window_attention), each compiled
        # once for every length and batch of at least two (PyTorch specializes a batch of one: a recompile of ~25 s,
        # so such a batch is padded to two): decode 7.7x and encode 2.8x the SDK's (decode 54.6 dB from fp32's
        # one-row decode, as fp16's 55.1; 98.4% of the codes equal)
        attention = sys.modules[type(self.codec).__module__].MossAudioTokenizerMultiheadAttention
        for m in self.codec.modules():
            if isinstance(m, attention):
                if not (m.causal and m.context):
                    raise SystemExit("MOSS attention without a causal window: window_attention does not apply")
                m._forward_non_streaming_sdpa = types.MethodType(window_attention, m)
        self.codec._encode_frame = torch.compile(self.codec._encode_frame, dynamic=True)  # batch_encode's whole path
        self.decode_codes = torch.compile(lambda codes: self.codec._decode_frame(codes).audio, dynamic=True)
        self.prompt = V3TurboBatchEngine(tts)._prompt_embeds
        self.denoiser = OnnxGraph(f"{path}/denoiser.onnx", "cuda", torch.float16, compile=True, mode=DENOISE_MODE)
        self.speaker = OnnxGraph(f"{path}/speaker_encoder.onnx", "cuda")
        self.window = torch.hann_window(N_FFT, periodic=True, device="cuda")
        self.frame = ContinuousFrame(tts.model, SLOTS, tts.config.max_position_embeddings,
                                     V3TurboBatchEngine.FUSED_MAX_FRAMES, repetition_window=DEFAULT_REP_WINDOW,
                                     **SAMPLING)
        self.ahead = ThreadPoolExecutor(1)  # prompt embeddings, built while the frames run
        self.threads = ThreadPoolExecutor(max(1, int(cpu_budget() // 4)))  # soxr
        self.active: dict[str, Request] = {}  # by key, until its audio is returned
        self.queue: deque[Chunk] = deque()  # chunks waiting for a slot; retries first
        self.slots: dict[int, Chunk] = {}
        self.decoding: list[tuple[Chunk, object]] = []  # (chunk, the codes to decode)
        self.parts: dict[str, list] = {}  # key -> its chunks' audio, as each finishes
        self.done: list[tuple[Request, np.ndarray]] = []
        log(f"VieNeu loaded and its frame captured in {time.perf_counter() - began:.0f} s")
        began = time.perf_counter()
        q = int(BUCKET_S * DENOISE_RATE)
        for n in range(1, int(REF_S / BUCKET_S) + 1):  # every reference length
            self.denoise([np.zeros(n * q, np.float32)])
        torch.cuda.synchronize()
        log(f"denoiser compiled for {len(self.denoiser.plans)} lengths in {time.perf_counter() - began:.0f} s")
        began = time.perf_counter()
        with torch.no_grad():
            self.codec.batch_encode([torch.zeros(2, int(BUCKET_S * OUT_RATE), device="cuda")] * 2)
            self.decode_codes(torch.zeros(tts.config.n_vq, 2, 16, dtype=torch.long, device="cuda"))
        torch.cuda.synchronize()
        log(f"codec encode and decode compiled in {time.perf_counter() - began:.0f} s")

    # enrolment

    def denoise(self, clips: list[np.ndarray]) -> list[np.ndarray]:
        """OnnxDenoiser.denoise of equal-length 44.1 kHz clips, batched: the STFT in fp32, the UNet in fp16."""
        import torch
        from vieneu._v3_turbo_engine.onnx_denoiser import HOP, N_FFT

        n = len(clips)
        clips = clips + [clips[-1]] * (-n % DENOISE_BATCH)
        with torch.no_grad():
            x = torch.from_numpy(np.stack(clips)).cuda()
            scale = x.abs().amax(1, keepdim=True).clamp(min=1e-7)
            w = torch.nn.functional.pad(x / scale, (0, 441))  # resemble's npad
            w = w / (w.abs().amax(1, keepdim=True) + 1e-7)
            s = torch.stft(w, N_FFT, HOP, window=self.window, center=True, pad_mode="reflect",
                           return_complex=True)[..., :-1]
            phi = s.angle()
            mag, cos, sin = (t.float() for t in self.denoiser(*(t.half() for t in (s.abs(), phi.cos(), phi.sin()))))
            spec = torch.complex(mag * cos, mag * sin)
            spec = torch.cat([spec, spec[..., -1:]], -1)
            out = torch.istft(spec, N_FFT, HOP, window=self.window, center=True, length=w.shape[1])
            return list((out[:n, : x.shape[1]] * scale[:n]).float().cpu().numpy())

    def enrol(self, clips: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """44.1 kHz references (`reference`) -> each one's (speaker embedding, reference codes [T, n_vq]), ENROL_PIECE
        rows at a time, those clean enough (`level_spread`) not denoised; each reference is dropped from `clips` once
        through the denoiser."""
        import soxr
        import torch

        groups: dict[int, list[int]] = {}
        for i, c in enumerate(clips):
            groups.setdefault(len(c), []).append(i)
        out: list = [None] * len(clips)
        for members in groups.values():
            for p in range(0, len(members), ENROL_PIECE):
                piece = members[p: p + ENROL_PIECE]
                with torch.no_grad():
                    spread = level_spread(torch.from_numpy(np.stack([clips[i] for i in piece])).cuda()).tolist()
                noisy = [i for i, db in zip(piece, spread) if db < CLEAN_SPREAD_DB]
                clean = {i: clips[i] for i in piece}
                for k in range(0, len(noisy), DENOISE_BATCH):
                    part = noisy[k: k + DENOISE_BATCH]
                    clean.update(zip(part, self.denoise([clips[i] for i in part])))
                for i in piece:
                    clips[i] = None
                ins = list(self.threads.map(lambda c: (soxr.resample(c, DENOISE_RATE, SPEAKER_RATE).astype(np.float32),
                                                       soxr.resample(c, DENOISE_RATE, OUT_RATE).astype(np.float32)),
                                            [clean[i] for i in piece]))
                del clean
                with torch.no_grad():
                    for k in range(0, len(piece), ENROL_BATCH):
                        part = range(k, min(k + ENROL_BATCH, len(piece)))
                        w16 = torch.from_numpy(np.stack([ins[j][0] for j in part])).cuda()
                        emb = self.speaker(speaker_fbank(w16))[0].float().cpu().numpy()
                        wavs = [torch.from_numpy(ins[j][1]).cuda().expand(2, -1) for j in part]
                        enc = self.codec.batch_encode(wavs + wavs[:1] * (len(wavs) == 1))  # at least two
                        lengths = enc.audio_codes_lengths.tolist()
                        codes = enc.audio_codes.permute(1, 2, 0).cpu().numpy()  # [B, T, n_vq]
                        for b, j in enumerate(part):
                            out[piece[j]] = (emb[b].copy(), np.ascontiguousarray(codes[b, : lengths[b]]))
        return out

    # generation

    @property
    def room(self) -> int:
        return max(0, QUEUE - len(self.queue))

    def submit(self, req: Request) -> None:
        from vieneu.v3_turbo_serve.engine import _min_expected_frames
        from vieneu_utils.core_utils import max_expected_frames

        self.active[req.key] = req
        self.parts[req.key] = [None] * len(req.chunks)
        for k, ph in enumerate(req.chunks):
            chunk = Chunk(req, k, min(MAX_NEW_FRAMES, max_expected_frames(ph)), _min_expected_frames(ph))
            self._queue(chunk)

    def _queue(self, chunk: Chunk, first: bool = False) -> None:
        req = chunk.req
        prompt = {"phonemes": req.chunks[chunk.k], "speaker_emb": req.speaker, "ref_codes": req.codes,
                  "use_ref_codes": True}
        chunk.embed = self.ahead.submit(self.prompt, prompt)
        (self.queue.appendleft if first else self.queue.append)(chunk)

    def step(self) -> list[tuple[Request, np.ndarray]]:
        """Fills free slots, runs one frame of every slot, decodes when DECODE_POOL chunks wait (or all that wait
        once nothing generates); returns the rows whose every chunk is done, with their audio."""
        free = self.frame.free
        if self.queue and len(free) >= min(ADMIT_MIN, len(self.queue)):
            self._admit(free)
        if self.slots:
            for slot, codes in self.frame.step():
                self._generated(self.slots.pop(slot), codes)
        if len(self.decoding) >= DECODE_POOL or (self.decoding and not self.slots and not self.queue):
            self._decode()
        out, self.done = self.done, []
        return out

    def _admit(self, free: list[int]) -> None:
        take = [self.queue.popleft() for _ in range(min(len(free), len(self.queue)))]
        free = free[: len(take)]  # a draining queue fills fewer slots than are free
        for k in range(0, len(take), PREFILL):
            part, slots = take[k: k + PREFILL], free[k: k + PREFILL]
            self.frame.admit(slots, [c.embed.result() for c in part], [c.req.speaker for c in part],
                             [c.cap for c in part])
            for s, c in zip(slots, part):
                c.embed = None
                self.slots[s] = c

    def _generated(self, chunk: Chunk, codes) -> None:
        if chunk.wav is not None:  # the babble guard's new try: decoded, then weighed against the kept one
            self.decoding.append((chunk, codes))
            return
        chunk.tries += 1
        if chunk.codes is None or len(codes) > len(chunk.codes):
            chunk.codes = codes
        if len(chunk.codes) < chunk.min_frames and chunk.tries <= SHORT_RETRIES:
            self._queue(chunk, first=True)
        else:
            self.decoding.append((chunk, chunk.codes))
            chunk.codes = None

    def _decode(self) -> None:
        """MOSS's decode in fp16, batched by length (each batch's codes padded on the right), then the babble
        guard."""
        import torch
        from vieneu_utils.core_utils import BABBLE_MAX_RETRIES, babble_prefer, babble_suspect

        pending = sorted(self.decoding, key=lambda x: len(x[1]))
        self.decoding = []
        for b in range(0, len(pending), DECODE_BATCH):
            batch = pending[b: b + DECODE_BATCH]
            with torch.no_grad():
                codes = torch.zeros(batch[0][1].shape[1], max(len(batch), 2), len(batch[-1][1]), dtype=torch.long,
                                    device="cuda")  # at least two rows
                for j, (_, c) in enumerate(batch):
                    codes[:, j, : len(c)] = c.T
                audio = self.decode_codes(codes).float().mean(1).cpu().numpy()
            for (chunk, codes), wav in zip(batch, audio):
                wav = wav[: len(codes) * self.codec.downsample_rate].copy()  # not a view holding the batch's array
                verdict = babble_suspect(wav, OUT_RATE, chunk.req.chunks[chunk.k], chunk.cap, len(codes))
                if chunk.wav is None:
                    chunk.wav, chunk.verdict = wav, verdict
                else:
                    chunk.rerolls += 1
                    if babble_prefer(verdict, chunk.verdict):
                        chunk.wav, chunk.verdict = wav, verdict
                if chunk.verdict[0] and chunk.rerolls < BABBLE_MAX_RETRIES:
                    self._queue(chunk, first=True)
                else:
                    self._finished(chunk)

    def _finished(self, chunk: Chunk) -> None:
        from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks

        req = chunk.req
        parts = self.parts[req.key]
        parts[chunk.k] = chunk.wav
        if all(p is not None for p in parts):
            del self.parts[req.key], self.active[req.key]
            self.done.append((req, join_audio_chunks(parts, OUT_RATE, silence_ps=gaps_to_silence(req.gaps))))


# --- run ---------------------------------------------------------------------------------------


@dataclass
class Shard:
    source: str
    name: str  # the segments file's name
    dst: Path  # <root>/<source>/en_audio/<name>
    ids: list[str]
    texts_en: list[str]
    clips: list[np.ndarray] | None  # the voiced rows' references (`reference`), until enrolled
    lines: list[tuple[list[str], list]]  # the voiced rows' chunks and pauses (`phonemize`)
    results: dict = None  # id -> the best attempt so far: {"wer", "broken", "asr", "audio", "duration", "tries"}
    skipped: set[str] = None  # rows voice.py skips: filters.vi_gate's, and those missing or with nothing speakable in text_en; not voiced, their English columns null


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
        sf.write(buf, resample_poly(wav, 1, 3).astype(np.float32), ASR_RATE, subtype="PCM_16", format="WAV")
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
    voiced = [(i, a["bytes"]) for i, a in zip(ids, seg.column("audio").to_pylist()) if i not in skipped]
    clips = pool.map(reference, [a for _, a in voiced], chunksize=32)  # both maps run at once
    lines = pool.map(phonemize, [en[i] for i, _ in voiced], chunksize=32)
    clips, lines = list(clips), list(lines)
    mute = [k for k, (chunks, _) in enumerate(lines) if not chunks]  # skipped too: en_tries 0, as the SDK says nothing
    if mute:
        log(f"{source}/{name}: {len(mute)} rows whose text_en has nothing speakable, e.g. {voiced[mute[0]][0]}")
        skipped |= {voiced[k][0] for k in mute}
        keep = [k for k in range(len(voiced)) if lines[k][0]]
        clips, lines = [clips[k] for k in keep], [lines[k] for k in keep]
    return Shard(source, name, root / source / "en_audio" / name, ids, [en.get(i, "") for i in ids], clips, lines, {},
                 skipped)


def write_shard(shard: Shard) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    r = [shard.results.get(i) for i in shard.ids]  # None: skipped (no text_en or Vietnamese gates)
    col = lambda f, kind: pa.array([None if x is None else f(x) for x in r], kind)  # noqa: E731
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
    """Every shard of each source in turn, the next one loading while the voicer voices; the voicer is kept full
    with retries first, then new rows. A shard is written once every voiced row has a kept attempt, and each
    source's written shards are pushed every PUSH_S and at its end."""
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

    # Loading and checking run in processes: as threads, their Python (PyAV's frame loop, the trim,
    # the phonemizer, the normalizer, scipy, Opus) would hold the GIL from the frame loop and starve
    # the GPU. Their thread pools stay at one thread each.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    spawn = get_context("spawn")
    pool = ProcessPoolExecutor(max(1, int(cpu_budget() // LOADER_SHARE)), mp_context=spawn)
    # A shard's references are ~17 GB (8 s at 44.1 kHz in float32 a row), freed as they enrol: the next shard loads
    # once the one before it starts enrolling, so at most two are held. Its ~80 s load hides behind a shard's voicing.
    loaded: queue.Queue = queue.Queue()
    ahead = threading.Semaphore(1)

    def loader():
        try:
            for source, name in todo:
                ahead.acquire()
                began_load = time.monotonic()
                shard = load_shard(source, name, args.root, pool)
                log(f"{source}/{name}: loaded {len(shard.clips)} references and lines in "
                    f"{time.monotonic() - began_load:.0f} s")
                loaded.put(shard)
        except BaseException as exc:  # raised again in the main thread
            loaded.put(exc)
        loaded.put(None)

    threading.Thread(target=loader, name="shard-loader", daemon=True).start()
    voicer = Voicer()
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

    def requests(key: str, shard: Shard, rows: list) -> Iterator[Request]:
        """The shard's rows, enrolled ENROL_PIECE at a time as the voicer takes them, shortest references first (a
        piece's references share few lengths): the frames and the ASR keep running between pieces."""
        if len(rows) != len(shard.clips) or len(rows) != len(shard.lines):
            raise SystemExit(f"{key}: {len(rows)} rows to voice, {len(shard.clips)} references, {len(shard.lines)} lines")
        order = sorted(range(len(rows)), key=lambda k: len(shard.clips[k]))
        spent = 0.0
        for p in range(0, len(order), ENROL_PIECE):
            piece = order[p: p + ENROL_PIECE]
            began_enrol = time.monotonic()
            refs = voicer.enrol([shard.clips[k] for k in piece])
            spent += time.monotonic() - began_enrol
            for k in piece:
                shard.clips[k] = None
            for k, (spk, codes) in zip(piece, refs):
                (i, t_en), (chunks, gaps) = rows[k], shard.lines[k]
                yield Request(i, t_en, chunks, gaps, spk, codes)
        log(f"{key}: enrolled {len(rows)} references in {spent:.0f} s")

    def next_shard() -> bool:
        nonlocal fresh, loading_done
        shard = loaded.get()
        if isinstance(shard, BaseException):
            raise shard
        if shard is None:
            loading_done = True
            return False
        ahead.release()  # the next shard loads while this one's references enrol, piece by piece
        key = f"{shard.source}/{shard.name}"
        rows = [(i, t_en) for i, t_en in zip(shard.ids, shard.texts_en) if i not in shard.skipped]
        shards[key], remaining[key] = shard, len(rows)
        for i, _ in rows:
            owner[i] = shard
        fresh = requests(key, shard, rows)
        if not rows:
            finish(shard)
        return True

    def finish(shard: Shard) -> None:
        """Writes a shard whose voiced rows all have a kept attempt, and pushes as due."""
        key = f"{shard.source}/{shard.name}"
        write_shard(shard)
        kept = list(shard.results.values())
        log(f"{key}: {len(shard.ids)} rows, {len(shard.skipped)} skipped by voice.py (Vietnamese gates, text_en missing or unspeakable), "
            f"{sum(x['tries'] for x in kept) / max(len(kept), 1):.2f} tries a row, "
            f"{sum(x['broken'] for x in kept) / max(len(kept), 1):.1%} still broken; overall "
            f"{attempts / (time.monotonic() - began):.1f} rows/s, RTF {(time.monotonic() - began) / max(voiced_s, 1):.5f} "
            f"over {voiced_s / 3600:.1f} h voiced ({attempts} attempts)")
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
            out = voicer.step()
            for req, wav in out:
                fut = checks.submit(_check, req.text, wav)
                checking[fut] = req
                inflight.add(fut)
            for fut in [f for f in inflight if f.done()]:
                inflight.discard(fut)
                settle(checking.pop(fut), fut.result())
            pushes.check()  # a failed upload stops the run; a rerun pushes what is written
            if loading_done and not voicer.active and not inflight and not retries:
                break
            if not voicer.active and not out:
                time.sleep(0.05)  # waiting on checks: their retries refill the voicer
    finally:  # after a crash, queued checks are dropped, not run, so the GPU frees promptly
        checks.shutdown(wait=False, cancel_futures=True)
        pool.shutdown(wait=False, cancel_futures=True)
    pushes.wait()
    log(f"done: {voiced_s / 3600:.1f} h voiced in {(time.monotonic() - began) / 3600:.2f} h")


def main() -> None:
    from training.tranche import ingest

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("run")
    p.add_argument("sources", nargs="+", choices=sorted(ingest.SOURCES))
    p.add_argument("--root", type=Path, default=Path("/workspace/tranche/repo"))
    p.add_argument("--asr", default="http://localhost:8100/v1/audio/transcriptions")
    p.add_argument("--limit", type=int, help="voice only this many shards (a staged start)")
    # As the bench ran: the prefill, decode and enrolment batches of many shapes fragment PyTorch's cached blocks.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

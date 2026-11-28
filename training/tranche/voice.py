#!/usr/bin/env python
"""VieNeu-TTS v3 Turbo voicing both ways, each row in its own speaker's voice, its SDK's steps batched on the GPU
([tranche-tts](../../docs/agent-memory/tranche-tts.md); speed: [tranche-tts-scaleup](../../docs/agent-memory/tranche-tts-scaleup.md#optimizing-vieneu)):
- English for the real-Vietnamese tranche (`run <source>`): each segments row's text_en from its Vietnamese clip;
- Vietnamese for the EN-VI dataset (`run en_speech`): each en_speech train row's text_vi from its English clip, where
  the Vietnamese the dataset holds (the Mac's MLX voicing, training/prep/tts_en_speech.py) is broken.

Per row, as vieneu 3.7.1's `generate_batch` after `_enroll_reference(denoise=True, use_ref_codes=True)`:
- the reference: the row's clip trimmed of silence at 30 dB, its first 8 s, cut to whole quarter seconds so the
  clips batch at 32 lengths, at 44.1 kHz (the loader's processes, a shard ahead); on the GPU, ENROL_PIECE rows at a
  time as the voicer takes them, shortest first: the resemble-enhance denoiser (onnx_graph.py, fp16, compiled per
  length; clean references skip it), the speaker encoder (onnx_graph.py) on a Kaldi fbank of the clean clip at
  16 kHz, and MOSS's codes of it at 48 kHz (fp16);
- the line in chunks of at most 256 characters, phonemized (the loader's processes);
- each chunk generated in one of SLOTS continuous-batching slots (vieneu_frame.py) and made again while shorter
  than the SDK's minimum (twice at most, the longest kept); decoded by MOSS in fp16, in batches of chunks of
  similar length; made again while the SDK's babble guard suspects it (twice at most, its preference kept);
- the chunks joined with the SDK's pauses, at 48 kHz;
- heard by the ASR server (Qwen3-ASR, its language forced) and broken when its error is over BROKEN (the English's
  WER under Whisper's normalizer, the Vietnamese's CER over `filters.spoken`, as ingest scores its ASR) or its pace
  is off PACE by 2x; voiced again while broken up to MAX_TRIES, the best try kept.

Env /workspace/envs/vieneu (box/envs.sh). From the repo root, with HF_TOKEN_AN (pushes) in the environment and the
ASR server up (`restart.sh asr`):
  /workspace/envs/vieneu/bin/python -m training.tranche.voice run yodas
      every segments shard (each has its text_en and asr_vi) on the Hub without en_audio yet, but those held out
      for evaluation (`ingest.SOURCES[...].heldout`); writes <root>/<source>/en_audio/<shard> (id, en_audio as Ogg
      Opus from 24 kHz, decoded at 48 kHz, en_duration_s, en_asr, en_wer, en_tries, en_broken). Rows the cache build
      drops on the Vietnamese side (`filters.vi_gate` on asr_vi), and those whose text_en is missing or has nothing
      speakable, are not voiced: en_tries 0, the other columns null.
  /workspace/envs/vieneu/bin/python -m training.tranche.voice run en_speech
      every en_speech train shard of the tranche repo (the EN-VI dataset's copy) without vi_check yet: each row's
      Vietnamese as the dataset holds it heard first, the broken ones (and any without audio) voiced again; writes
      <root>/en_speech/vi_check/<shard> (id, vi_audio as Ogg Opus from 48 kHz where a new try is kept and null where
      the dataset's stays, then of the row's kept audio vi_duration_s, vi_asr, vi_cer, vi_tries (0: the dataset's,
      heard only) and vi_broken). A row without text_vi is neither heard nor voiced: all null but vi_tries 0. Each
      shard's download is deleted once read.
  Each source's folder is pushed every 30 min and at its end. A rerun skips the shards already written or pushed,
  and first pushes any written but not yet pushed.
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
DECODE_MIN_FRAMES = 33  # shorter, a decode stage of MOSS's fits one flex_attention block: compiled apart (Voicer)
EN_SPEECH = "en_speech"  # the EN-VI dataset's rows in the tranche repo, voiced in Vietnamese; every other source in English
WHISPER_NORMALIZER = ("openai/whisper-large-v3", "06f233f")  # its English normalizer only
# A try is broken over these: the English's WER under Whisper's normalizer (as the TTS bake-off scored), the
# Vietnamese's CER over filters.spoken at the gate's filters.MAX_CER.
BROKEN = {"en": 0.2, "vi": 0.15}
# Seconds of audio per character of the line: the bake-off's median English, within the Mac's Vietnamese medians on
# en_speech (0.060-0.068, 2026-11-28); a try off it by 2x either way is broken.
PACE = 0.0636
# The kept Opus's input rate: the tranche's English at Mimi's 24 kHz, the EN-VI dataset's Vietnamese at its 48 kHz.
KEPT_RATE = {"en": 24000, "vi": 48000}
# Attempts per row, the best kept. 1 from the scale-up on (2026-11-06): on GigaSpeech 2 a retry
# costs more TTS time than its rows give back (tranche-tts.md); the first tranche ran 3.
MAX_TRIES = 1
ASR_TIMEOUT_S = 60  # a request takes well under a second
# The server stalls past a try now and then and recovers within minutes: 3 tries ended a run three
# times in two days (2026-11-06, 11-07 17:22 and 23:17 UTC+7), each losing the shards in flight.
ASR_TRIES = 10
ASR_MAX_TOKENS = 512  # a looping transcript comes back long and scores broken
CHECKERS = 32  # checking processes
# rows a checking process hears at once, each on its own thread: 256 requests in flight, the server's --max-num-seqs.
# The server's time is its decode steps, mostly a read of its weights: one request a process (~30 in flight) held
# the whole run to ~28 rows/s beside the voicer (2026-11-28).
CHECK_ROWS = 8
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
    """A line, English or Vietnamese -> its chunks phonemized and the SDK's pauses between them; no chunk when nothing
    in it is speakable (the row is then skipped)."""
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps, phonemize_text_with_emotions

    chunks, gaps = normalize_to_chunks_v3_with_gaps(text, max_chars=CHUNK_CHARS)
    return [phonemize_text_with_emotions(c) for c in chunks], gaps


# --- the voicer --------------------------------------------------------------------------------


@dataclass(eq=False)
class Request:
    key: str
    text: str  # the line, which the check hears against
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
        # Encode and decode whole (no streaming chunks), their attention windowed (window_attention), compiled:
        # decode 7.7x and encode 2.8x the SDK's (decode 54.6 dB from fp32's one-row decode, as fp16's 55.1; 98.4% of
        # the codes equal). Every layer is causal, so right padding never reaches a real frame. The encode runs at one
        # shape, ENROL_BATCH references padded to the longest; the decode at any length of at least
        # DECODE_MIN_FRAMES and batch of at least two. Smaller, PyTorch specializes it (a batch of one; a stage of a
        # single flex_attention block): a recompile of 25-50 s each, mid-run.
        attention = sys.modules[type(self.codec).__module__].MossAudioTokenizerMultiheadAttention
        windowed = 0
        for m in self.codec.modules():
            if isinstance(m, attention):
                if not (m.causal and m.context):
                    raise SystemExit("MOSS attention without a causal window: window_attention does not apply")
                m._forward_non_streaming_sdpa = types.MethodType(window_attention, m)
                windowed += 1
        if not windowed:  # the SDK's dense decode would build a T x T bias and run out of memory
            raise SystemExit("no MOSS attention layer found to window")
        hop = self.codec.downsample_rate  # samples a code frame
        self.encode_len = -(-int(REF_S * OUT_RATE) // hop) * hop
        self.encode = torch.compile(lambda wav: self.codec._encode_frame(wav).audio_codes, dynamic=False)
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
            self.encode(torch.zeros(ENROL_BATCH, 2, self.encode_len, device="cuda"))
            # sizes unlike each other and the codebooks' 16, so none is tied to another
            self.decode_codes(torch.zeros(tts.config.n_vq, 3, DECODE_MIN_FRAMES + 7, dtype=torch.long, device="cuda"))
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

        from training.prep import kaldi

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
                        # the SDK's extract_speaker_fbank (kaldi-native-fbank), batched: embeddings at cosine 0.999999
                        emb = self.speaker(kaldi.fbank(w16, SPEAKER_RATE, "povey"))[0].float().cpu().numpy()
                        wav = torch.zeros(ENROL_BATCH, 2, self.encode_len, device="cuda")
                        for b, j in enumerate(part):
                            wav[b, :, : len(ins[j][1])] = torch.from_numpy(ins[j][1]).cuda()
                        codes = self.encode(wav).permute(1, 2, 0).cpu().numpy()  # [B, T, n_vq]
                        for b, j in enumerate(part):  # batch_encode's lengths: whole frames of the clip
                            n = len(ins[j][1]) // self.codec.downsample_rate
                            out[piece[j]] = (emb[b].copy(), np.ascontiguousarray(codes[b, :n]))
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
                codes = torch.zeros(batch[0][1].shape[1], max(len(batch), 2), max(len(batch[-1][1]), DECODE_MIN_FRAMES),
                                    dtype=torch.long, device="cuda")
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
    name: str  # its file's name: a segments shard's, or an en_speech train shard's
    lang: str  # the language it is voiced in: "en", or "vi" for en_speech
    dst: Path  # <root>/<source>/en_audio/<name>, or <root>/en_speech/vi_check/<name>
    ids: list[str]
    texts: list[str]  # each row's line, in `lang`
    clips: list[np.ndarray] | None  # the voiced rows' references (`reference`), until enrolled
    lines: list[tuple[list[str], list]]  # the voiced rows' chunks and pauses (`phonemize`)
    # id -> the best try so far: {"err", "broken", "asr", "audio", "duration", "tries"}; an en_speech row's Vietnamese
    # as the dataset holds it is try 0, its "audio" None
    results: dict
    skipped: set[str]  # rows not voiced (`load_segments`, `load_en_speech`)


class Checker:
    """Hears a try with Qwen3-ASR (`qwen-asr-serve`, its language forced) and scores it against its line: the
    English's WER under Whisper's normalizer (the line unbracketed first), the Vietnamese's CER over `filters.spoken`,
    and its pace (BROKEN, PACE). A voiced try is also encoded as the Ogg Opus kept (KEPT_RATE)."""

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

    def error(self, lang: str, text: str, asr: str) -> float:
        import jiwer

        if lang == "en":
            ref = self.norm(re.sub(r"[()\[\]{}<>]", " ", text))
            return jiwer.wer(ref, self.norm(asr)) if ref.strip() else 0.0
        from training.tranche.filters import spoken

        ref, hyp = " ".join(spoken(text)), " ".join(spoken(asr))
        return jiwer.cer(ref, hyp) if ref else float(bool(hyp))

    def check(self, lang: str, text: str, wav: np.ndarray, keep: bool = True) -> dict:
        """A 48 kHz try of `text` in `lang` -> its score, and with `keep` the Ogg Opus kept of it."""
        import io

        import httpx
        import soundfile as sf
        from scipy.signal import resample_poly

        from training.tranche.ingest import encode_opus

        buf = io.BytesIO()
        sf.write(buf, resample_poly(wav, 1, 3).astype(np.float32), ASR_RATE, subtype="PCM_16", format="WAV")
        for attempt in range(ASR_TRIES):  # it also drops an idle keep-alive connection now and then
            try:
                reply = self.client.post(self.url, files={"file": ("a.wav", buf.getvalue(), "audio/wav")},
                                         data={"model": "asr", "language": lang, "to_language": lang,
                                               "max_completion_tokens": ASR_MAX_TOKENS})
            except httpx.TransportError:
                if attempt == ASR_TRIES - 1:
                    raise
                time.sleep(10)  # a refused connection (the server restarting) fails at once
                continue
            reply.raise_for_status()
            asr = reply.json()["text"]
            break
        err = self.error(lang, text, asr)
        duration = len(wav) / OUT_RATE
        pace = duration / max(len(text), 1)
        broken = err > BROKEN[lang] or not 0.5 * PACE <= pace <= 2 * PACE
        audio = None
        if keep:
            rate = KEPT_RATE[lang]
            pcm = wav if rate == OUT_RATE else resample_poly(wav, rate // np.gcd(rate, OUT_RATE), OUT_RATE // np.gcd(rate, OUT_RATE))
            with self.encoding:
                audio = encode_opus(pcm.astype(np.float32), rate)
        return {"err": round(err, 4), "broken": broken, "asr": asr, "duration": round(duration, 3), "audio": audio}


_checker: Checker | None = None
_hearing = None  # the checking process's threads, one a row of a part


def _start_checker(url: str, encoding) -> None:
    from concurrent.futures import ThreadPoolExecutor

    global _checker, _hearing
    _checker = Checker(url, encoding)
    _hearing = ThreadPoolExecutor(CHECK_ROWS)


def _check(rows: list[tuple[str, str, np.ndarray]]) -> list[dict]:
    """Voiced tries (lang, text, 48 kHz audio) -> their scores and Opus."""
    return list(_hearing.map(lambda row: _checker.check(*row), rows))


def _hear_held(rows: list[tuple[str, bytes]]) -> list[dict]:
    """The EN-VI dataset's Vietnamese (text_vi, its Ogg Opus) -> their scores; nothing encoded."""
    def hear(row):
        pcm, rate = decode(row[1])
        if rate != OUT_RATE:  # libopus decodes at 48 kHz whatever rate it encoded
            raise ValueError(f"Vietnamese audio decoded at {rate} Hz")
        return _checker.check("vi", row[0], pcm, keep=False)

    return list(_hearing.map(hear, rows))


def prepare(key: str, pool, rows: list[tuple[str, bytes, str]]) -> tuple[list, list, set[str]]:
    """The references and lines of rows to voice (id, their clip, their line), in the loader's processes; those
    whose line has nothing speakable (the SDK would say nothing) come back apart, as ids."""
    clips = pool.map(reference, [c for _, c, _ in rows], chunksize=32)  # both maps run at once
    lines = pool.map(phonemize, [t for _, _, t in rows], chunksize=32)
    clips, lines = list(clips), list(lines)
    keep = [k for k, (chunks, _) in enumerate(lines) if chunks]
    mute = {rows[k][0] for k in set(range(len(rows))) - set(keep)}
    if mute:
        log(f"{key}: {len(mute)} rows whose line has nothing speakable, e.g. {min(mute)}")
    return [clips[k] for k in keep], [lines[k] for k in keep], mute


def load_segments(source: str, name: str, root: Path, pool) -> Shard:
    """A tranche segments shard, voiced in English: all its rows but those the cache build drops on the Vietnamese
    side (`filters.vi_gate` on asr_vi), those missing text_en and those with nothing speakable in it."""
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
    # The cache build rejects the vi_gate rows as "not voiced" and the no_en rows gracefully.
    skipped = no_en | {i for i, text, r in zip(ids, texts, vi) if vi_gate(r | {"text": text})}
    voiced = [(i, a["bytes"], en[i]) for i, a in zip(ids, seg.column("audio").to_pylist()) if i not in skipped]
    clips, lines, mute = prepare(f"{source}/{name}", pool, voiced)
    return Shard(source, name, "en", root / source / "en_audio" / name, ids, [en.get(i, "") for i in ids], clips,
                 lines, {}, skipped | mute)


def load_en_speech(name: str, root: Path, pool, checks) -> Shard:
    """An en_speech train shard, voiced in Vietnamese: every row's Vietnamese as the dataset holds it heard first (in
    the checking processes), then the rows it is broken on, or missing for, voiced again. Downloaded beside <root>
    (as ingest's corpus shards), not into the HF cache, and deleted once read: the 1,832 shards are ~447 GB."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from training.tranche.ingest import REPO

    path = Path(hf_hub_download(REPO, f"{EN_SPEECH}/train/{name}", repo_type="dataset",
                                local_dir=root / EN_SPEECH / ".download"))
    try:
        t = pq.read_table(path, columns=["id", "text_vi", "vi_audio", "en_audio"])
    finally:
        path.unlink()
    ids, texts = t.column("id").to_pylist(), [s or "" for s in t.column("text_vi").to_pylist()]
    held = [a and a["bytes"] for a in t.column("vi_audio").to_pylist()]
    heard = [k for k in range(len(ids)) if texts[k] and held[k]]
    parts = [heard[p: p + CHECK_ROWS] for p in range(0, len(heard), CHECK_ROWS)]
    futures = [checks.submit(_hear_held, [(texts[k], held[k]) for k in part]) for part in parts]
    del held
    results = {}
    for part, future in zip(parts, futures):
        for k, r in zip(part, future.result()):
            results[ids[k]] = r | {"tries": 0}
    voice = [k for k in range(len(ids)) if texts[k] and (ids[k] not in results or results[ids[k]]["broken"])]
    english = t.column("en_audio").take(pa.array(voice, pa.int64())).to_pylist()  # typed: an empty list too
    clips, lines, mute = prepare(f"{EN_SPEECH}/{name}", pool,
                                 [(ids[k], a["bytes"], texts[k]) for k, a in zip(voice, english)])
    log(f"{EN_SPEECH}/{name}: {len(heard)} of {len(ids)} rows heard, {len(voice)} to voice "
        f"({sum(r['broken'] for r in results.values())} broken, {sum(map(bool, texts)) - len(heard)} without audio)")
    return Shard(EN_SPEECH, name, "vi", root / EN_SPEECH / "vi_check" / name, ids, texts, clips, lines, results,
                 (set(ids) - {ids[k] for k in voice}) | mute)


ERROR = {"en": "wer", "vi": "cer"}  # the error column's name


def write_shard(shard: Shard) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    lang = shard.lang
    r = [shard.results.get(i) for i in shard.ids]  # None: neither voiced nor heard
    col = lambda f, kind: pa.array([None if x is None else f(x) for x in r], kind)  # noqa: E731
    table = pa.table({
        "id": shard.ids,
        f"{lang}_audio": pa.array([None if x is None or x["audio"] is None else
                                   {"bytes": x["audio"], "path": f"{i.replace('/', '_')}.opus"}
                                   for i, x in zip(shard.ids, r)],
                                  pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        f"{lang}_duration_s": col(lambda x: x["duration"], pa.float64()),
        f"{lang}_asr": col(lambda x: x["asr"], pa.string()),
        f"{lang}_{ERROR[lang]}": col(lambda x: x["err"], pa.float32()),
        f"{lang}_tries": pa.array([0 if x is None else x["tries"] for x in r], pa.int8()),
        f"{lang}_broken": col(lambda x: x["broken"], pa.bool_()),
    })
    shard.dst.parent.mkdir(parents=True, exist_ok=True)
    part = shard.dst.with_name(shard.dst.name + ".part")
    pq.write_table(table, part)
    os.replace(part, shard.dst)


def out_part(source: str) -> str:
    """The part a source's shards are voiced into."""
    return "vi_check" if source == EN_SPEECH else "en_audio"


def shards_to_voice(args: argparse.Namespace, files: list[str]) -> list[tuple[str, str]]:
    """(source, name) of every shard on the Hub to voice that neither the Hub nor <root> has the voicing of: for the
    tranche's sources, their segments shards but those held out for evaluation, once translated and heard (a later
    run takes the others); for en_speech, its train shards."""
    import pyarrow.parquet as pq

    from training.tranche import ingest

    todo = []
    for source in args.sources:
        part = out_part(source)
        done = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/{part}/")}
        done |= {p.name for p in (args.root / source / part).glob("*.parquet")}
        if source == EN_SPEECH:
            todo += [(source, n) for n in sorted(f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/train/"))
                     if n not in done]
            continue
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
        waiting = 0
        for f in sorted(f for f in files if f.startswith(f"{source}/segments/")):
            name = f.rsplit("/", 1)[1]
            if name in ingest.SOURCES[source].heldout or name not in translated or name in done:
                continue  # held out: an evaluation set's shard, never trained on, so never voiced
            if name in heard:
                todo.append((source, name))
            else:
                waiting += 1
        if waiting:
            log(f"{source}: {waiting} translated shards wait for their asr_vi (filters.py)")
    return todo[: args.limit]


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

    ingest.an_token()
    files = HfApi().list_repo_files(ingest.REPO, repo_type="dataset")
    for source in args.sources:
        part = out_part(source)
        hub = {f.rsplit("/", 1)[1] for f in files if f.startswith(f"{source}/{part}/")}
        if {p.name for p in (args.root / source / part).glob("*.parquet")} - hub:  # written by an earlier run
            ingest.push_part(args.root, source, part)                              # that stopped before pushing them
    todo = shards_to_voice(args, files)
    log(f"{len(todo)} shards to voice: " + ", ".join(f"{s} {sum(t[0] == s for t in todo)}" for s in args.sources))
    if not todo:
        return

    # Loading and checking run in processes: as threads, their Python (PyAV's frame loop, the trim,
    # the phonemizer, the normalizer, scipy, Opus) would hold the GIL from the frame loop and starve
    # the GPU. Their thread pools stay at one thread each.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    spawn = get_context("spawn")
    pool = ProcessPoolExecutor(max(1, int(cpu_budget() // LOADER_SHARE)), mp_context=spawn)
    # en_speech's loading hears its rows through these too, a shard ahead of the voicer.
    checks = ProcessPoolExecutor(CHECKERS, mp_context=spawn, initializer=_start_checker,
                                 initargs=(args.asr, spawn.BoundedSemaphore(OPUS_AT_ONCE)))
    # A shard's references are ~17 GB (8 s at 44.1 kHz in float32 a row), freed as they enrol: the next shard loads
    # once the one before it starts enrolling, so at most two are held. Its ~80 s load hides behind a shard's voicing.
    loaded: queue.Queue = queue.Queue()
    ahead = threading.Semaphore(1)

    def loader():
        try:
            for source, name in todo:
                ahead.acquire()
                began_load = time.monotonic()
                shard = (load_en_speech(name, args.root, pool, checks) if source == EN_SPEECH
                         else load_segments(source, name, args.root, pool))
                log(f"{source}/{name}: loaded {len(shard.clips)} references and lines in "
                    f"{time.monotonic() - began_load:.0f} s")
                loaded.put(shard)
        except BaseException as exc:  # raised again in the main thread
            loaded.put(exc)
        loaded.put(None)

    threading.Thread(target=loader, name="shard-loader", daemon=True).start()
    voicer = Voicer()
    checking: dict = {}  # future -> its part's requests
    inflight = set()
    retries: list[Request] = []
    fresh: Iterator[Request] = iter(())
    shards: dict[str, Shard] = {}  # by source/name, until written
    owner: dict[str, Shard] = {}  # row id -> its shard
    remaining: dict[str, int] = {}  # shard -> rows without a kept attempt
    loading_done = False
    left = {s: sum(t[0] == s for t in todo) for s in args.sources}  # shards not yet written
    pushes = {s: ingest.Pushes(args.root, out_part(s)) for s in args.sources}  # uploads run beside the voicing
    pushed = {s: time.monotonic() for s in args.sources}
    began, voiced_s, attempts = time.monotonic(), 0.0, 0

    def push(source: str) -> None:
        pushed[source] = time.monotonic()
        pushes[source].push(source)

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
                (i, line), (chunks, gaps) = rows[k], shard.lines[k]
                yield Request(i, line, chunks, gaps, spk, codes)
        log(f"{key}: enrolled {len(rows)} references in {spent:.0f} s")

    def next_shard() -> bool:
        """Opens the next loaded shard; waits for one only while the voicer is idle (en_speech's loads hear a whole
        shard, longer than voicing the last one's broken rows)."""
        nonlocal fresh, loading_done
        try:
            shard = loaded.get(block=not voicer.active)
        except queue.Empty:
            return False
        if isinstance(shard, BaseException):
            raise shard
        if shard is None:
            loading_done = True
            return False
        ahead.release()  # the next shard loads while this one's references enrol, piece by piece
        key = f"{shard.source}/{shard.name}"
        rows = [(i, line) for i, line in zip(shard.ids, shard.texts) if i not in shard.skipped]
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
        voiced = [shard.results[i] for i in shard.ids if i not in shard.skipped]
        log(f"{key}: {len(shard.ids)} rows, {len(voiced)} voiced, {len(shard.skipped)} not; "
            f"{sum(x['tries'] for x in voiced) / max(len(voiced), 1):.2f} tries a voiced row, "
            f"{sum(x['broken'] for x in voiced) / max(len(voiced), 1):.1%} of them still broken; overall "
            f"{attempts / (time.monotonic() - began):.1f} rows/s, RTF {(time.monotonic() - began) / max(voiced_s, 1):.5f} "
            f"over {voiced_s / 3600:.1f} h voiced ({attempts} attempts)")
        for i in shard.ids:
            owner.pop(i, None)
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
        if best is None or result["err"] < best["err"] or (best["broken"] and not result["broken"]):
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
            for p in range(0, len(out), CHECK_ROWS):
                part = out[p: p + CHECK_ROWS]
                fut = checks.submit(_check, [(owner[req.key].lang, req.text, wav) for req, wav in part])
                checking[fut] = [req for req, _ in part]
                inflight.add(fut)
            for fut in [f for f in inflight if f.done()]:
                inflight.discard(fut)
                for req, result in zip(checking.pop(fut), fut.result()):
                    settle(req, result)
            for p in pushes.values():
                p.check()  # a failed upload stops the run; a rerun pushes what is written
            if loading_done and not voicer.active and not inflight and not retries:
                break
            if not voicer.active and not out:
                time.sleep(0.05)  # waiting on checks: their retries refill the voicer
    finally:  # after a crash, queued checks are dropped, not run, so the GPU frees promptly
        checks.shutdown(wait=False, cancel_futures=True)
        pool.shutdown(wait=False, cancel_futures=True)
    for p in pushes.values():
        p.wait()
    log(f"done: {voiced_s / 3600:.1f} h voiced in {(time.monotonic() - began) / 3600:.2f} h")


def main() -> None:
    from training.tranche import ingest

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)
    p = steps.add_parser("run")
    p.add_argument("sources", nargs="+", choices=sorted([*ingest.SOURCES, EN_SPEECH]))
    p.add_argument("--root", type=Path, default=Path("/workspace/tranche/repo"))
    p.add_argument("--asr", default="http://localhost:8100/v1/audio/transcriptions")
    p.add_argument("--limit", type=int, help="voice only this many shards (a staged start)")
    # As the bench ran: the prefill, decode and enrolment batches of many shapes fragment PyTorch's cached blocks.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

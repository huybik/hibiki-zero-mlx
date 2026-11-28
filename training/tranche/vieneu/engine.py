"""VieNeu-TTS v3 Turbo's engine for voice.py, its SDK's steps batched on the GPU
([tranche-tts-scaleup](../../../docs/agent-memory/tranche-tts-scaleup.md#optimizing-vieneu)). Per row, as vieneu
3.7.1's `generate_batch` after `_enroll_reference(denoise=True, use_ref_codes=True)`:
- the reference: the row's clip trimmed of silence at 30 dB, its first 8 s, cut to whole quarter seconds so the
  clips batch at 32 lengths, at 44.1 kHz (`reference`, in voice.py's loader processes, a shard ahead); on the GPU,
  ENROL_PIECE rows at a time as the voicer takes them, shortest first: the resemble-enhance denoiser (onnx_graph.py,
  fp16, compiled per length; clean references skip it), the speaker encoder (onnx_graph.py) on a Kaldi fbank of the
  clean clip at 16 kHz, and MOSS's codes of it at 48 kHz (fp16);
- the line in chunks of at most 256 characters, phonemized (`phonemize`, in the loader's processes);
- each chunk generated in one of SLOTS continuous-batching slots (frame.py) and made again while shorter
  than the SDK's minimum (twice at most, the longest kept); decoded by MOSS in fp16, in batches of chunks of
  similar length; made again while the SDK's babble guard suspects it (twice at most, its preference kept);
- the chunks joined with the SDK's pauses, at 48 kHz.
"""
from __future__ import annotations

import sys
import time
import types
from collections import deque
from dataclasses import dataclass

import numpy as np

MODEL = ("pnnbao-ump/VieNeu-TTS-v3-Turbo", "61b85e3d937fbbacb387714180e8182823512523")
MODEL_FILES = ["config.json", "update/*", "denoiser.onnx", "speaker_encoder.onnx"]  # the PyTorch path's
CODEC = ("OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano", "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68")
DENOISE_RATE, SPEAKER_RATE, OUT_RATE = 44100, 16000, 48000
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


def log(msg: str) -> None:
    """The UTC date and time, then `msg`: voice.py's log line."""
    print(f"{time.strftime('%F %T', time.gmtime())} {msg}", flush=True)


def snapshot(repo: str, revision: str, **kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, **kwargs)


# --- the loader's processes --------------------------------------------------------------------


def reference(data: bytes) -> np.ndarray:
    """A segment's Ogg Opus -> its reference before the denoiser: silence trimmed at TRIM_DB, the first REF_S, cut to
    whole BUCKET_S (one under a bucket padded with silence to one: a new length would compile mid-run, and under 25 ms
    the speaker's fbank has no frame), at 44.1 kHz by soxr (the SDK's resampler)."""
    import librosa
    import soxr

    from training.audio.decode import decode

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
        from training.tranche.vieneu.frame import ContinuousFrame
        from training.tranche.vieneu.onnx_graph import OnnxGraph

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

        from training.audio import kaldi

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

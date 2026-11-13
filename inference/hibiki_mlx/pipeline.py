#!/usr/bin/env python
"""Pipelined MLX hibiki-zero inference on staged model directories (quantized or bf16).

Overlaps the CPU Mimi codec (rustymimi, GIL-released) with the GPU LM:
  - encoder thread streams encode_step over the whole file, running ahead
  - main thread runs the autoregressive LM step on the GPU
  - decoder thread streams decode_step on the audio tokens
FIFO queues preserve the streaming order, so output is bit-identical to the
sequential loop; we just stop letting the CPU and GPU idle on each other.

File mode feeds the source as training's free-running evaluation does
(`training/finetune/common.py` `generate_batch`): the levelled clip in whole frames, the source
EOS on the frame after it, then silence. It stops at the first text EOS the model writes once it
has heard the source EOS; an earlier one stops nothing, where training's evaluation would end
there and drop the rest of a long file.

Usage: python inference/tools/verify_mlx_q4.py  (or `from hibiki_mlx import load, run`)
"""

import codecs
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece
import sphn

from hibiki_mlx.level import Leveller
from moshi_mlx import models, utils

ROOT = Path(__file__).resolve().parents[2]  # repo root (inference/hibiki_mlx/ -> ../..)

W = ROOT / "weights"
# Upstream's 3B, staged from weights/ by `convert_mlx.py --bits 4` (README).
UPSTREAM = W / "hibiki-zero-3b-q4"
SAMPLE = ROOT / "inference" / "assets" / "leon.wav"  # French gate clip
SENTINEL = object()
FRAME = 1920  # samples at 24 kHz: one 12.5 Hz codec frame (80 ms)
# Silence after the source, the source EOS frame included: training's evaluation generates its
# longest clip plus 20 s (GenerationConfig.tail_s), since the text EOS came up to 14 s after a
# FLEURS row ended and 8 s cut a third of them.
TAIL_S = 20.0


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}\n{hint}")


def resolve_weights_dir(model: str | Path = "3b") -> Path:
    return UPSTREAM if str(model) == "3b" else Path(model)


def quantize(lm: nn.Module, record: dict) -> None:
    """Quantizes `lm` as a config.json `quantization` record says (convert_mlx.py writes it):
    every Linear and Embedding at its `group_size` and `bits`, unless its module path has an
    entry of its own."""

    def spec(path: str, module: nn.Module) -> bool | dict:
        return hasattr(module, "to_quantized") and (dict(record[path]) if path in record else True)

    nn.quantize(lm, record["group_size"], record["bits"], class_predicate=spec)


def text_special_ids(tokenizer) -> set[int]:
    return {tokenizer.unk_id(), tokenizer.eos_id(), tokenizer.pad_id()}


class TextDecoder:
    """The text of a stream of text tokens, one token at a time: for valid UTF-8, what the
    tokenizer decodes from the stream's non-special tokens. A byte-fallback piece (`<0xE1>`)
    carries one UTF-8 byte, so a character the vocabulary lacks spans several tokens: its text
    comes out once its sequence is complete, an invalid run of bytes as one U+FFFD (SentencePiece
    writes one per byte), and `flush` returns what is left when the stream ends. Special tokens
    add nothing and do not break a sequence."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.special = text_special_ids(tokenizer)
        self.utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def __call__(self, token: int) -> str:
        if token in self.special:
            return ""
        piece = self.tokenizer.id_to_piece(token)
        if self.tokenizer.is_byte(token):
            return self.utf8.decode(bytes([int(piece[1:-1], 16)]))
        return self.utf8.decode(piece.replace("▁", " ").encode())

    def flush(self) -> str:
        return self.utf8.decode(b"", final=True)


@dataclass(frozen=True)
class Model:
    """A loaded model directory: the LM, its tokenizer and how config.json's `lm_gen_config`
    samples it."""

    lm: models.Lm
    config: models.LmConfig
    tokenizer: sentencepiece.SentencePieceProcessor
    text_sampler: utils.Sampler
    audio_sampler: utils.Sampler
    mimi_path: Path

    def mimi(self) -> rustymimi.Tokenizer:
        """A fresh Mimi codec. Each thread needs its own (one instance cannot be borrowed by two
        at once), and each stream a fresh one: `reset()` leaves state behind (the first frame of
        the next clip came out different)."""
        nq = max(self.config.other_codebooks, self.config.generated_codebooks)
        return rustymimi.Tokenizer(str(self.mimi_path), num_codebooks=nq)

    def condition(self):
        if self.lm.condition_provider is None:
            return None
        return self.lm.condition_provider.condition_tensor("description", "very_good")


def load(weights_dir: Path) -> Model:
    """A staged model directory: config.json names the MLX weights (`moshi_name`), the tokenizer
    they are on (`tokenizer_name`), the sampling (`lm_gen_config`) and, for quantized weights,
    their quantization (`quantization`); the weights load strictly."""
    cfg_path = weights_dir / "config.json"
    _require_file(
        cfg_path,
        "Use a model directory staged by inference/tools/convert_mlx.py (README: upstream's 3B "
        "is staged into weights/hibiki-zero-3b-q4).",
    )
    cfg = json.loads(cfg_path.read_text())
    for key in ("moshi_name", "tokenizer_name", "lm_gen_config"):
        if key not in cfg:
            raise ValueError(f"{cfg_path} has no {key}: stage the model with convert_mlx.py")
    _require_file(weights_dir / cfg["moshi_name"], "The weights config.json names (moshi_name).")
    _require_file(
        weights_dir / cfg["tokenizer_name"],
        "A model directory holds the tokenizer its config.json names "
        "(convert_mlx.py copies it in).",
    )
    lm_config = models.LmConfig.from_config_dict(cfg)
    lm = models.Lm(lm_config)
    lm.set_dtype(mx.bfloat16)
    if "quantization" in cfg:
        quantize(lm, cfg["quantization"])
    lm.load_weights(str(weights_dir / cfg["moshi_name"]), strict=True)
    mx.eval(lm.parameters())
    tok = sentencepiece.SentencePieceProcessor(str(weights_dir / cfg["tokenizer_name"]))
    if tok.vocab_size() != cfg["text_card"]:
        raise RuntimeError("Tokenizer vocabulary does not match config.json")
    mimi_path = weights_dir / cfg.get("mimi_name", "mimi-pytorch-e351c8d8@125.safetensors")
    _require_file(mimi_path, "Download or copy the Mimi checkpoint into the model directory.")
    sampling = cfg["lm_gen_config"]
    return Model(
        lm=lm,
        config=lm_config,
        tokenizer=tok,
        text_sampler=utils.Sampler(top_k=sampling["top_k_text"], temp=sampling["temp_text"]),
        audio_sampler=utils.Sampler(top_k=sampling["top_k"], temp=sampling["temp"]),
        mimi_path=mimi_path,
    )


@dataclass(frozen=True)
class Translation:
    text: str
    ended: bool  # the model emitted the text EOS after hearing the source EOS
    frames: int  # LM steps run
    duration: float  # seconds of input audio
    last_word: int | None  # the step of the last content token (above PAD) before the EOS
    pcm: np.ndarray | None  # translated audio at 24 kHz, unless audio=False


def translate(model: Model, infile: str | Path, audio: bool = True) -> Translation:
    """Translates a file as training's free-running evaluation does: the first channel through
    the input level stage, zero-padded to whole frames, then one frame whose source codes are all
    the source EOS (`card`), then silence up to TAIL_S. It stops at the first text EOS of a step
    that heard the source EOS; a text EOS before it is kept going past, so a long file is
    translated to its end (training's evaluation stops at any EOS).

    Step t hears source frames before t, as training's forward does; moshi's LMGen in the
    evaluation hears one frame of silence more before the clip (80 ms later throughout)."""
    lm, cfg = model.lm, model.config
    mimi_enc = model.mimi()
    mimi_dec = model.mimi() if audio else None
    ct = model.condition()
    other_cb = cfg.other_codebooks
    gen_cb = cfg.generated_codebooks

    in_pcms, _ = sphn.read(str(infile), sample_rate=24000)
    source = Leveller()(in_pcms[0])  # the first channel
    duration = source.shape[-1] / 24000
    steps = -(-source.shape[-1] // FRAME)
    source = np.pad(source, (0, steps * FRAME - source.shape[-1]))
    tail = round(TAIL_S * 12.5)

    gen = models.LmGen(
        model=lm,
        max_steps=steps + tail + 8,
        text_sampler=model.text_sampler,
        audio_sampler=model.audio_sampler,
        cfg_coef=1.0,
        check=False,
    )
    lm.warmup(ct)
    enc_q: queue.Queue = queue.Queue(maxsize=64)  # encoder -> main
    dec_q: queue.Queue = queue.Queue(maxsize=64)  # main -> decoder
    out_pcm: list = []
    stop = threading.Event()

    def encoder():
        # Separate streaming state from the model; runs ahead of the LM. Queue numpy (not mx)
        # arrays: lazy mx graphs are bound to the creating thread's stream and can't be
        # evaluated from the LM thread.
        def codes(pcm_frame):
            out = mimi_enc.encode_step(pcm_frame)  # CPU, GIL released
            return np.transpose(out, (0, 2, 1))[0, :, :other_cb]

        silence = np.zeros((1, 1, FRAME), dtype=source.dtype)
        for idx in range(steps):
            if stop.is_set():
                break
            enc_q.put(codes(source[None, None, idx * FRAME : (idx + 1) * FRAME]))
        for idx in range(tail):
            if stop.is_set():
                break
            # Mimi still codes the silence under the EOS frame, as a whole-file encode does.
            silent = codes(silence)
            enc_q.put(np.full_like(silent, cfg.audio_eos_token) if idx == 0 else silent)
        enc_q.put(SENTINEL)

    def decoder():
        while True:
            item = dec_q.get()
            if item is SENTINEL:
                break
            out_pcm.append(mimi_dec.decode_step(item))  # CPU, GIL released

    enc_t = threading.Thread(target=encoder, daemon=True)
    enc_t.start()
    if audio:
        dec_t = threading.Thread(target=decoder, daemon=True)
        dec_t.start()

    text = TextDecoder(model.tokenizer)
    pieces: list[str] = []
    eos_token = model.tokenizer.eos_id()
    pad_token = model.tokenizer.pad_id()
    processed = 0
    ended = False
    last_word = None
    while True:
        oat = enc_q.get()
        if oat is SENTINEL:
            break
        tt = gen.step(mx.array(oat), ct)[0].item()  # sync this frame's LM
        processed += 1
        # The source EOS frame goes in at step `steps`, which step steps + 1 hears first.
        if tt == eos_token and processed > steps + 1:
            ended = True
            break
        if tt > pad_token:
            last_word = processed - 1
        pieces.append(text(tt))
        audio_tokens = gen.last_audio_tokens()
        if audio and audio_tokens is not None and gen_cb > 0:
            dec_q.put(np.array(audio_tokens[:, :, None]).astype(np.uint32))
    stop.set()
    # Drain the queue so the encoder isn't parked on a full put(), then shut down.
    while True:
        try:
            if enc_q.get(timeout=0.1) is SENTINEL:
                break
        except queue.Empty:
            if not enc_t.is_alive():
                break
    enc_t.join()
    if audio:
        dec_q.put(SENTINEL)
        dec_t.join()
    pcm = np.concatenate(out_pcm, axis=-1)[0, 0] if out_pcm else None
    return Translation(
        text=("".join(pieces) + text.flush()).strip(),
        ended=ended,
        frames=processed,
        duration=duration,
        last_word=last_word,
        pcm=pcm,
    )


def run(infile: str, outfile: str, model: Model, text_outfile: str | None = None):
    """Translates a file into `outfile` (24 kHz WAV) and its text, and prints both."""
    t0 = time.perf_counter()
    out = translate(model, infile)
    wall = time.perf_counter() - t0

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    if out.pcm is not None:
        sphn.write_wav(outfile, out.pcm, 24000)
    text_path = Path(text_outfile) if text_outfile else Path(outfile).with_suffix(".txt")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(out.text + "\n", encoding="utf-8")
    print(out.text)
    print(
        f"\n[{out.frames} frames for {out.duration:.2f}s of audio "
        f"({'ended' if out.ended else 'no EOS'}) in {wall:.2f}s -> {out.frames / wall:.1f} "
        f"frames/s ({out.duration / wall:.2f}x RT), out: {outfile}, text: {text_path}]"
    )


if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else str(SAMPLE)
    outfile = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "translations" / "leon_mlx_fast.wav")
    mx.random.seed(299792458)
    run(infile, outfile, load(UPSTREAM))

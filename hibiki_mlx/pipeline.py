#!/usr/bin/env python
"""Pipelined MLX hibiki-zero inference for q4 and bf16 weights.

Overlaps the CPU Mimi codec (rustymimi, GIL-released) with the GPU LM:
  - encoder thread streams encode_step over the whole file, running ahead
  - main thread runs the autoregressive LM step on the GPU
  - decoder thread streams decode_step on the audio tokens
FIFO queues preserve the streaming order, so output is bit-identical to the
sequential loop; we just stop letting the CPU and GPU idle on each other.

Usage: python scripts/verify_mlx_q4.py  (or `from hibiki_mlx import load, run`)
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import rustymimi
import sentencepiece
import sphn

from moshi_mlx import models, utils

ROOT = Path(__file__).resolve().parent.parent  # repo root (hibiki_mlx/ -> ..)

W = ROOT / "weights"
SENTINEL = object()
PAD_STOP = 12  # frames (~1 s) of sustained pad after audio ends => translation flushed


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing {path}\n{hint}")


def _resolve_audio_path(infile: str) -> str:
    path = Path(infile)
    if path.exists():
        return str(path)
    if len(path.parts) > 1 and path.parts[0] == "samples":
        packaged = ROOT / "assets" / path
        if packaged.exists():
            return str(packaged)
    return infile


def _q4_compatible(_: str, module: object) -> bool:
    weight = getattr(module, "weight", None)
    return weight is not None and hasattr(module, "to_quantized") and weight.shape[-1] % 32 == 0


def _q4_model_name(cfg: dict) -> str:
    name = cfg.get("moshi_name")
    if not (isinstance(name, str) and name.endswith(".q4.safetensors")):
        name = "hibiki.q4.safetensors"
    return name


def _model_name(cfg: dict) -> str:
    if cfg.get("weight_dtype") != "bfloat16":
        return _q4_model_name(cfg)
    name = cfg.get("moshi_name")
    if not (isinstance(name, str) and name.endswith(".bf16.safetensors")):
        raise ValueError("bfloat16 model config requires moshi_name ending in .bf16.safetensors")
    return name


def resolve_weights_dir(model: str | Path = "3b") -> Path:
    return W if str(model) == "3b" else Path(model)


def text_special_ids(tokenizer) -> set[int]:
    return {tokenizer.unk_id(), tokenizer.eos_id(), tokenizer.pad_id()}


def load(weights_dir: Path):
    cfg_path = weights_dir / "config.json"
    _require_file(
        cfg_path,
        "Use an MLX model directory containing config.json.",
    )
    cfg = json.loads(cfg_path.read_text())
    model_name = _model_name(cfg)
    tokenizer_name = cfg.get("tokenizer_name", "tokenizer_spm_48k_multi6_2.model")
    _require_file(
        weights_dir / model_name,
        "Use a staged MLX model directory, or run the matching conversion script.",
    )
    _require_file(
        weights_dir / tokenizer_name,
        "Download or copy the tokenizer into the MLX model directory.",
    )
    lm_config = models.LmConfig.from_config_dict(cfg)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if cfg.get("weight_dtype") != "bfloat16":
        nn.quantize(model, bits=4, group_size=32, class_predicate=_q4_compatible)
    model.load_weights(str(weights_dir / model_name), strict=True)
    mx.eval(model.parameters())
    tok = sentencepiece.SentencePieceProcessor(str(weights_dir / tokenizer_name))
    if "text_card" in cfg and tok.vocab_size() != cfg["text_card"]:
        raise RuntimeError("Tokenizer vocabulary does not match config.json")
    mimi_enc, mimi_dec = make_mimi(weights_dir, lm_config)
    return model, lm_config, tok, mimi_enc, mimi_dec


def make_mimi(weights_dir: Path, lm_config):
    # Separate codec instances per thread: a single rustymimi.Tokenizer can't be
    # borrowed by the encoder and decoder threads at once ("Already borrowed").
    # Fresh instances also reset the streaming state, so batch callers must make
    # a new pair per file rather than reuse one across files.
    cfg_path = weights_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    mimi_name = cfg.get("mimi_name", "mimi-pytorch-e351c8d8@125.safetensors")
    _require_file(
        weights_dir / mimi_name,
        "Download or copy the Mimi checkpoint into the model directory.",
    )
    mimi_path = str(weights_dir / mimi_name)
    nq = max(lm_config.other_codebooks, lm_config.generated_codebooks)
    return (
        rustymimi.Tokenizer(mimi_path, num_codebooks=nq),
        rustymimi.Tokenizer(mimi_path, num_codebooks=nq),
    )


def run(
    infile: str,
    outfile: str,
    weights_dir: Path = W,
    text_outfile: str | None = None,
    tail_s: float = 8.0,
    preloaded=None,
    text_temp: float = 0.4,
):
    infile = _resolve_audio_path(infile)
    model, lm_config, text_tok, mimi_enc, mimi_dec = preloaded or load(weights_dir)
    if model.condition_provider is not None:
        ct = model.condition_provider.condition_tensor("description", "very_good")
    else:
        ct = None
    other_cb = lm_config.other_codebooks
    gen_cb = lm_config.generated_codebooks

    in_pcms, _ = sphn.read(infile, sample_rate=24000)
    steps = in_pcms.shape[-1] // 1920
    # Hibiki translates simultaneously with a ~6 s lag, so when the input ends the
    # tail of the translation is still unspoken. Feed trailing silence to flush it
    # (mirroring the gen-duration padding of the PyTorch path); without this short
    # clips get their translation truncated. tail_s is an upper bound — generation
    # early-stops once the model goes quiet (see PAD_STOP below), so it doesn't sit
    # in silence long enough to start hallucinating.
    tail = int(round(tail_s * 12.5))

    gen = models.LmGen(
        model=model,
        max_steps=steps + tail + 8,
        text_sampler=utils.Sampler(top_k=25, temp=text_temp),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        cfg_coef=1.0,
        check=False,
    )
    model.warmup(ct)
    enc_q: queue.Queue = queue.Queue(maxsize=64)  # encoder -> main
    dec_q: queue.Queue = queue.Queue(maxsize=64)  # main -> decoder
    out_pcm: list = []

    stop = threading.Event()

    def encoder():
        # Separate streaming state from the model; runs ahead of the LM. Queue
        # numpy (not mx) arrays: lazy mx graphs are bound to the creating
        # thread's stream and can't be evaluated from the LM thread.
        def emit(pcm_frame):
            codes = mimi_enc.encode_step(pcm_frame)  # CPU, GIL released
            enc_q.put(np.transpose(codes, (0, 2, 1))[0, :, :other_cb])

        silence = np.zeros((1, 1, 1920), dtype=in_pcms.dtype)  # flush the lag tail
        for idx in range(steps):
            if stop.is_set():
                break
            emit(in_pcms[None, 0:1, idx * 1920 : (idx + 1) * 1920])
        for _ in range(tail):
            if stop.is_set():
                break
            emit(silence)
        enc_q.put(SENTINEL)

    def decoder():
        while True:
            item = dec_q.get()
            if item is SENTINEL:
                break
            out_pcm.append(mimi_dec.decode_step(item))  # CPU, GIL released

    enc_t = threading.Thread(target=encoder, daemon=True)
    dec_t = threading.Thread(target=decoder, daemon=True)
    enc_t.start()
    dec_t.start()

    text_pieces: list[str] = []
    special_text_tokens = text_special_ids(text_tok)
    eos_token = text_tok.eos_id()
    processed = 0
    pad_run = 0
    t0 = time.perf_counter()
    while True:
        oat = enc_q.get()
        if oat is SENTINEL:
            break
        text_token = gen.step(mx.array(oat), ct)
        tt = text_token[0].item()  # sync this frame's LM
        processed += 1
        if tt not in special_text_tokens:
            piece = text_tok.id_to_piece(tt).replace("▁", " ")
            text_pieces.append(piece)
        audio = gen.last_audio_tokens()
        if audio is not None and gen_cb > 0:
            dec_q.put(np.array(audio[:, :, None]).astype(np.uint32))
        # After the input audio ends, stop once the model goes quiet (sustained pad).
        # Sitting in silence longer makes it hallucinate/repeat and inflates WER.
        if processed > steps:
            if tt == eos_token:
                break
            pad_run = pad_run + 1 if tt in special_text_tokens else 0
            if pad_run >= PAD_STOP:
                break
    stop.set()
    # Drain the queue so the encoder isn't parked on a full put(), then shut down.
    while True:
        try:
            if enc_q.get(timeout=0.1) is SENTINEL:
                break
        except queue.Empty:
            if not enc_t.is_alive():
                break
    dec_q.put(SENTINEL)
    enc_t.join()
    dec_t.join()
    wall = time.perf_counter() - t0

    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    if out_pcm:
        pcm = np.concatenate(out_pcm, axis=-1)[0, 0]
        sphn.write_wav(outfile, pcm, 24000)
    text = "".join(text_pieces).strip()
    text_path = Path(text_outfile) if text_outfile else Path(outfile).with_suffix(".txt")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text + "\n")
    print(text)
    print(
        f"\n[{processed} frames ({steps} audio) in {wall:.2f}s -> {processed / wall:.1f} frames/s "
        f"({steps / wall / 12.5:.2f}x RT), out: {outfile}, text: {text_path}]"
    )


if __name__ == "__main__":
    infile = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "assets" / "samples" / "leon.wav")
    outfile = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "translations" / "leon_mlx_fast.wav")
    mx.random.seed(299792458)
    run(infile, outfile)

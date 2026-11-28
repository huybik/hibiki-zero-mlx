"""File mode: pipelined MLX hibiki-zero inference of a file, on staged model directories
(quantized or bf16).

Overlaps the CPU Mimi codec (rustymimi, GIL-released) with the GPU LM:
  - encoder thread streams encode_step over the whole file, running ahead
  - main thread runs the autoregressive LM step on the GPU
  - decoder thread streams decode_step on the audio tokens
FIFO queues preserve the streaming order, so output is bit-identical to the
sequential loop; we just stop letting the CPU and GPU idle on each other.

File mode feeds the source as training's free-running evaluation does
(`training/finetune/generation.py` `generate_batch`): the levelled clip in whole frames, the source
EOS on the frame after it, then silence. It stops at the first text EOS the model writes once it
has heard the source EOS; an earlier one stops nothing, where training's evaluation would end
there and drop the rest of a long file. `run` (main.py's file mode) also stops after QUIET_S
without a word once the source has ended; the gate does not.
"""

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import sphn

from hibiki_mlx.level import FRAME, Leveller
from hibiki_mlx.model import Model
from hibiki_mlx.primer import prime, primer_steps
from hibiki_mlx.text import TextDecoder

SENTINEL = object()
# Silence after the source, the source EOS frame included: training's evaluation generates its
# longest clip plus 20 s (GenerationConfig.tail_s), since the text EOS came up to 14 s after a
# FLEURS row ended and 8 s cut a third of them.
TAIL_S = 20.0
# File mode's stop once the model has written no word for this long after hearing the source EOS:
# the two-way model often writes no text EOS (49 of OneVoice's 200 short clips), and such a file
# would run the whole tail on silence.
QUIET_S = 5.0


@dataclass(frozen=True)
class Translation:
    text: str
    ended: bool  # the model emitted the text EOS after hearing the source EOS
    frames: int  # LM steps run
    duration: float  # seconds of input audio
    last_word: int | None  # the step of the last content token (above PAD) before the EOS
    pcm: np.ndarray | None  # translated audio at 24 kHz, unless audio=False


def translate(
    model: Model,
    infile: str | Path,
    audio: bool = True,
    quiet_s: float | None = None,
    speaking: str | None = None,
    voice_gate: bool = False,
) -> Translation:
    """Translates a file as training's free-running evaluation does: the first channel through
    the input level stage, zero-padded to whole frames, then one frame whose source codes are all
    the source EOS (`card`), then silence up to TAIL_S. It stops at the first text EOS of a step
    that heard the source EOS; a text EOS before it is kept going past, so a long file is
    translated to its end (training's evaluation stops at any EOS). With `quiet_s` it also stops
    once no word has come for that long since the later of the source EOS and the last word.
    With `speaking` (a PRIMERS key) the stream starts with that language's primer (`prime`), whose
    steps `frames` leaves out. With `voice_gate` the levelled source goes through the voice gate
    (`voice.VoiceGate`): silence wherever no speech is near.

    Step t hears source frames before t, as training's forward does; moshi's LMGen in the
    evaluation hears max_delay - 1 frames of silence first (one on the 3B), so its steps come
    later."""
    lm, cfg = model.lm, model.config
    mimi_enc = model.mimi()
    mimi_dec = model.mimi() if audio else None
    other_cb = cfg.other_codebooks
    gen_cb = cfg.generated_codebooks

    in_pcms, _ = sphn.read(str(infile), sample_rate=24000)
    source = Leveller()(in_pcms[0])  # the first channel
    duration = source.shape[-1] / 24000
    steps = -(-source.shape[-1] // FRAME)
    source = np.pad(source, (0, steps * FRAME - source.shape[-1]))
    if voice_gate and steps:
        from hibiki_mlx.voice import VoiceGate  # torch and Silero load only where a gate runs

        gate = VoiceGate()
        gated = [gate(frame) for frame in source.reshape(-1, FRAME)]
        source = np.concatenate([frame for frame in gated if frame is not None] + gate.flush())
    tail = round(TAIL_S * 12.5)
    quiet = None if quiet_s is None else round(quiet_s * 12.5)

    gen = model.generator(primer_steps(speaking) + steps + tail + 8)
    lm.warmup(model.text_sampler, model.audio_sampler)
    if speaking:
        prime(model, gen, mimi_enc, speaking)  # before the encoder thread takes mimi_enc
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

    def put_audio() -> None:
        # The frame whose last codebook the step just sampled. Audio lags text by max_delay
        # steps, so the first steps' audio is from before the clip: none at a stream's start, the
        # primer's after `prime`.
        audio_tokens = gen.last_audio_tokens()
        if audio and processed > gen.max_delay and audio_tokens is not None and gen_cb > 0:
            dec_q.put(np.array(audio_tokens[:, :, None]).astype(np.uint32))

    while True:
        oat = enc_q.get()
        if oat is SENTINEL:
            break
        tt = gen.step(mx.array(oat)).item()  # sync this frame's LM
        processed += 1
        put_audio()
        # The source EOS frame goes in at step `steps`, which step steps + 1 hears first.
        if tt == eos_token and processed > steps + 1:
            ended = True
            # Training's evaluation keeps the speech of every frame before the EOS step; the last
            # max_delay - 1 of them are complete only that many steps on (7 frames on a heads
            # student), so step on through the encoder's silence for their audio alone.
            for _ in range(gen.max_delay - 1 if audio else 0):
                oat = enc_q.get()
                if oat is SENTINEL:  # the tail ended first, where training's generation ends too
                    break
                gen.step(mx.array(oat))
                processed += 1
                put_audio()
            break
        if tt > pad_token:
            last_word = processed - 1
        pieces.append(text(tt))
        # The source EOS went in at step `steps`; step processed - 1 has just run.
        if quiet is not None and processed - 1 - max(steps, last_word or 0) >= quiet:
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


def run(
    infile: str,
    outfile: str,
    model: Model,
    text_outfile: str | None = None,
    speaking: str | None = None,
):
    """Translates a file into `outfile` (24 kHz WAV) and its text, and prints both; the model
    hears the source through the voice gate and stops after QUIET_S without a word once the
    source has ended. `speaking` primes it (`translate`)."""
    t0 = time.perf_counter()
    out = translate(model, infile, quiet_s=QUIET_S, speaking=speaking, voice_gate=True)
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


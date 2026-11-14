#!/usr/bin/env python
"""hibiki-zero MLX translation — realtime mic or a file, on quantized or bf16 weights.

  python inference/main.py path/to/audio.wav   # file -> translations/<stem>_translated.wav
  python inference/main.py --mic               # mic  -> speakers, live (Ctrl-C to stop)

You get streamed text + 24 kHz audio. Both modes use hibiki_mlx.pipeline
(load()/run()) and sample as the model directory's config.json says (lm_gen_config);
--model accepts 3b (upstream, staged into weights/hibiki-zero-3b-q4) or any staged
model directory. File mode is the 3-thread pipelined path and ends the input as
training's evaluation does (the source EOS, then silence), stopping at the text EOS or
once the model has said nothing for 5 s after the source ended (pipeline.QUIET_S); mic
mode never ends it, and pipelines encode->LM->decode across threads so the live
critical path is just the LM step (~22 ms on the M4 Pro for the 3B at 4 bits, 25 at 6;
budget 80 ms).

Mic mode cancels the speakers' echo (hibiki_mlx.echo) before the input level stage, so the
model does not hear and translate its own translation. It reaches echoes up to ~450 ms late:
the Mac's own speakers, not every Bluetooth or AirPlay speaker.

Both modes pass the source through the voice gate (hibiki_mlx.voice): the model hears
silence until speech starts and wherever no one speaks, never noise alone, 160 ms late.

--speaking vi|en says which language is spoken: both modes then start the stream with a
short primer in that language, unheard, so the two-way model keeps that direction instead
of guessing it from the voice (pipeline.PRIMERS).
"""
import argparse
import gc
import queue
import sys
import threading
from pathlib import Path

import mlx.core as mx
import numpy as np

from hibiki_mlx import pipeline as f
from hibiki_mlx.level import Leveller
from hibiki_mlx.voice import VoiceGate
from moshi_mlx import models

ROOT = Path(__file__).resolve().parent.parent  # repo root (inference/ -> ..)

FRAME = 1920  # samples @ 24 kHz = one 12.5 Hz codec frame (80 ms)
BLOCK = 480   # samples per sound-device callback (20 ms): small, so the echo comes back soon


def run_mic(max_steps: int, model: f.Model, speaking: str | None = None):
    import sounddevice as sd

    from hibiki_mlx.echo import EchoCanceller

    mimi_enc, mimi_dec = model.mimi(), model.mimi()
    text = f.TextDecoder(model.tokenizer)
    ct = model.condition()
    other_cb = model.config.other_codebooks
    gen_cb = model.config.generated_codebooks
    gen = models.LmGen(
        model=model.lm, max_steps=max_steps + f.primer_steps(speaking),
        text_sampler=model.text_sampler, audio_sampler=model.audio_sampler,
        cfg_coef=1.0, check=False,
    )
    model.lm.warmup(ct)
    if speaking:                                      # the primer, unheard, before the mic opens
        f.prime(model, gen, mimi_enc, ct, speaking)
    # A full collection over the loaded objects holds the GIL ~40 ms, enough to delay the 20 ms
    # sound callback, which shifts the echo's delay for good on the Mac's separate devices.
    gc.freeze()

    in_q: queue.Queue = queue.Queue()                 # (heard, played) block pairs
    enc_q: queue.Queue = queue.Queue()                # encoder -> LM
    dec_q: queue.Queue = queue.Queue()                # LM -> decoder
    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()
    failed: list[BaseException] = []                  # a worker's error, raised at the end

    playing = [np.zeros(0, np.float32)]               # decoded translation not yet played

    def on_audio(indata, outdata, frames, t, status):
        while playing[0].size < frames:
            try:
                playing[0] = np.concatenate([playing[0], out_q.get_nowait()])
            except queue.Empty:
                break                                  # not ready yet -> silence
        played = np.zeros(frames, np.float32)
        ready = playing[0][:frames]
        played[: ready.size] = ready
        playing[0] = playing[0][ready.size :]
        outdata[:, 0] = played
        in_q.put_nowait((indata[:, 0].copy(), played))  # what the mic heard while it played

    # Pipeline the codec off the main LM thread (same trick as the file path): encode
    # and decode are CPU (GIL-free) and independent of the LM recurrence, so the
    # live critical path collapses from encode+LM+decode (~58 ms) to just the LM
    # step (~24 ms on M4). Costs one frame (80 ms) of extra output latency.
    echo = EchoCanceller()                            # the session's echo canceller
    gate = VoiceGate()                                # and voice gate

    def encoder():
        # Queue numpy (not mx) arrays: lazy mx graphs are bound to the creating
        # thread's stream and can't be evaluated from the LM thread.
        leveller = Leveller()                                           # the session's level
        blocks = []                                                     # cancelled, toward a frame
        while not stop.is_set():
            try:
                heard, played = in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            blocks.append(echo(heard, played))
            if len(blocks) < FRAME // BLOCK:
                continue
            frame = gate(leveller(np.concatenate(blocks)))              # silence unless speech
            blocks = []
            if frame is None:                                           # its first LOOKAHEAD
                continue
            codes = mimi_enc.encode_step(frame[None, None, :])          # CPU, GIL free
            enc_q.put_nowait(np.transpose(codes, (0, 2, 1))[0, :, :other_cb])

    def decoder():
        while not stop.is_set():
            try:
                at = dec_q.get(timeout=0.1)
            except queue.Empty:
                continue
            out = mimi_dec.decode_step(at)                              # CPU, GIL free
            out_q.put_nowait(out[0, 0])                                 # (1920,) float32

    def guarded(loop):
        def run():                                                      # an error ends the session
            try:
                loop()
            except BaseException as error:
                failed.append(error)
            finally:
                stop.set()
        return run

    workers = [threading.Thread(target=guarded(fn), daemon=True) for fn in (encoder, decoder)]
    for worker in workers:
        worker.start()
    ended = ""
    live = 0                                          # steps run on the mic
    try:
        # One duplex stream, so each block played and the block heard beside it stay a fixed
        # time apart, the echo's delay the canceller finds; its small blocks keep that delay
        # (132 ms on the MacBook Pro, 282 at 80 ms blocks) well inside the canceller's reach (~450 ms).
        with sd.Stream(samplerate=24000, channels=1, blocksize=BLOCK,
                       dtype="float32", callback=on_audio):
            print(f"listening — the translation plays back. Ctrl-C to stop "
                  f"(cap {max_steps / 12.5 / 60:.0f} min)\n")
            while not stop.is_set():
                try:
                    codes = enc_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                tt = gen.step(mx.array(codes), ct)
                live += 1
                sys.stdout.write(text(tt[0].item()))                    # sync this frame
                sys.stdout.flush()
                audio = gen.last_audio_tokens()
                # Audio lags by max_delay steps: the first ones' is the primer's, if any.
                if live > gen.max_delay and audio is not None and gen_cb > 0:
                    dec_q.put_nowait(np.array(audio[:, :, None]).astype(np.uint32))
    except ValueError as e:                                             # reached max_steps
        ended = f"\n[reached cap: {e}]"
    except KeyboardInterrupt:
        ended = "\n[stopping]"
    finally:
        stop.set()
        for worker in workers:
            worker.join()
    print(text.flush() + ended)                         # the text's unfinished tail, then why
    if failed:
        raise failed[0]


def main():
    p = argparse.ArgumentParser(description="hibiki-zero MLX translation (mic or file)")
    p.add_argument("input", nargs="?", help="audio file to translate")
    p.add_argument("--mic", action="store_true", help="realtime mic -> speakers")
    p.add_argument("-o", "--out", help="output wav (file mode); default translations/<stem>_translated.wav")
    p.add_argument("--text-out", help="output text transcript (file mode); default matches output wav with .txt")
    p.add_argument("--model", default="3b", help="3b or a staged Hibiki-Zero model directory")
    p.add_argument("--minutes", type=float, default=30.0, help="mic session cap (default 30)")
    p.add_argument("--speaking", choices=sorted(f.PRIMERS),
                   help="the language spoken; primes the direction (default: the model guesses)")
    args = p.parse_args()

    mic = args.mic or args.input == "mic"
    if not (mic or args.input):
        p.error("give an audio file path, or --mic for realtime")
    print("loading MLX weights ...")
    model = f.load(f.resolve_weights_dir(args.model))
    mx.random.seed(299792458)
    if mic:
        run_mic(max_steps=int(args.minutes * 60 * 12.5) + 8, model=model, speaking=args.speaking)
    else:
        infile = args.input
        out = args.out or str(ROOT / "translations" / f"{Path(infile).stem}_translated.wav")
        f.run(infile, out, model, text_outfile=args.text_out, speaking=args.speaking)


if __name__ == "__main__":
    main()

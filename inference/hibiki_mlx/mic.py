"""main.py's mic mode: the microphone translated live, the translation played back."""

import gc
import queue
import sys
import threading

import mlx.core as mx
import numpy as np

from hibiki_mlx.level import FRAME, Leveller
from hibiki_mlx.model import Model
from hibiki_mlx.primer import prime, primer_steps
from hibiki_mlx.text import TextDecoder
from hibiki_mlx.voice import VoiceGate

BLOCK = 480   # samples per sound-device callback (20 ms): small, so the echo comes back soon


def run_mic(max_steps: int, model: Model, speaking: str | None = None):
    import sounddevice as sd

    from hibiki_mlx.echo import EchoCanceller

    mimi_enc, mimi_dec = model.mimi(), model.mimi()
    text = TextDecoder(model.tokenizer)
    other_cb = model.config.other_codebooks
    gen_cb = model.config.generated_codebooks
    gen = model.generator(max_steps + primer_steps(speaking))
    model.lm.warmup(model.text_sampler, model.audio_sampler)
    if speaking:                                      # the primer, unheard, before the mic opens
        prime(model, gen, mimi_enc, speaking)
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
                tt = gen.step(mx.array(codes))
                live += 1
                sys.stdout.write(text(tt.item()))                       # sync this frame
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

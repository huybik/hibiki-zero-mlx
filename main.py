#!/usr/bin/env python
"""hibiki-zero MLX translation — realtime mic or a file, on q4 + the fast path.

  python main.py path/to/audio.wav      # file  -> translations/<stem>_translated.wav
  python main.py --mic                  # mic   -> speakers, live (Ctrl-C to stop)

Speak/record FR/ES/PT/DE; you get streamed EN text + 24 kHz EN audio. Both modes
use the 4-bit weights via hibiki_mlx.pipeline (load()/run()); --model picks the
3B (default) or the Hibiki-M 1B (FR->EN only). File mode is the 3-thread
pipelined path; mic mode pipelines encode->LM->decode across threads so the live
critical path is just the LM step (3B ~22 ms / 1B ~15 ms on M4, budget 80 ms).
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from hibiki_mlx import pipeline as f
from moshi_mlx import models, utils

ROOT = Path(__file__).resolve().parent

FRAME = 1920  # samples @ 24 kHz = one 12.5 Hz codec frame (80 ms)


def run_mic(max_steps: int, weights_dir: Path = f.W, text_temp: float = 0.4):
    import sounddevice as sd

    print("loading q4 weights ...")
    model, lm_config, text_tok, mimi_enc, mimi_dec = f.load(weights_dir)
    ct = None
    if model.condition_provider is not None:
        ct = model.condition_provider.condition_tensor("description", "very_good")
    other_cb = lm_config.other_codebooks
    gen_cb = lm_config.generated_codebooks
    gen = models.LmGen(
        model=model, max_steps=max_steps,
        text_sampler=utils.Sampler(top_k=25, temp=text_temp),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        cfg_coef=1.0, check=False,
    )
    model.warmup(ct)
    mx.eval(model.parameters())

    in_q: queue.Queue = queue.Queue()
    enc_q: queue.Queue = queue.Queue()                # encoder -> LM
    dec_q: queue.Queue = queue.Queue()                # LM -> decoder
    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def on_input(indata, frames, t, status):
        in_q.put_nowait(indata[:, 0].copy())          # (1920,) float32 mic frame

    def on_output(outdata, frames, t, status):
        try:
            outdata[:, 0] = out_q.get_nowait()         # translated EN PCM
        except queue.Empty:
            outdata.fill(0)                            # not ready yet -> silence

    # Pipeline the codec off the LM thread (same trick as the file path): encode
    # and decode are CPU (GIL-free) and independent of the LM recurrence, so the
    # live critical path collapses from encode+LM+decode (~58 ms) to just the LM
    # step (~24 ms on M4). Costs one frame (80 ms) of extra output latency.
    def encoder():
        # Queue numpy (not mx) arrays: lazy mx graphs are bound to the creating
        # thread's stream and can't be evaluated from the LM thread.
        while not stop.is_set():
            try:
                pcm = in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            codes = mimi_enc.encode_step(pcm[None, None, :])             # CPU, GIL free
            enc_q.put_nowait(np.transpose(codes, (0, 2, 1))[0, :, :other_cb])

    def lm():
        try:
            while not stop.is_set():
                try:
                    codes = enc_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                tt = gen.step(mx.array(codes), ct)
                tok = tt[0].item()                                       # sync this frame
                if tok not in (0, 3):
                    sys.stdout.write(text_tok.id_to_piece(tok).replace("▁", " "))
                    sys.stdout.flush()
                audio = gen.last_audio_tokens()
                if audio is not None and gen_cb > 0:
                    dec_q.put_nowait(np.array(audio[:, :, None]).astype(np.uint32))
        except ValueError as e:                                         # reached max_steps
            print(f"\n[reached cap: {e}]")
        finally:
            stop.set()

    def decoder():
        while not stop.is_set():
            try:
                at = dec_q.get(timeout=0.1)
            except queue.Empty:
                continue
            out = mimi_dec.decode_step(at)                              # CPU, GIL free
            out_q.put_nowait(out[0, 0])                                 # (1920,) float32

    for fn in (encoder, lm, decoder):
        threading.Thread(target=fn, daemon=True).start()
    print(f"listening — speak FR/ES/PT/DE; translated EN plays back. Ctrl-C to stop "
          f"(cap {max_steps / 12.5 / 60:.0f} min)\n")
    try:
        with sd.InputStream(samplerate=24000, channels=1, blocksize=FRAME,
                            dtype="float32", callback=on_input), \
             sd.OutputStream(samplerate=24000, channels=1, blocksize=FRAME,
                             dtype="float32", callback=on_output):
            while not stop.is_set():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[stopping]")
    finally:
        stop.set()


def main():
    p = argparse.ArgumentParser(description="hibiki-zero MLX q4 translation (mic or file)")
    p.add_argument("input", nargs="?", help="audio file to translate (FR/ES/PT/DE)")
    p.add_argument("--mic", action="store_true", help="realtime mic -> speakers")
    p.add_argument("-o", "--out", help="output wav (file mode); default translations/<stem>_translated.wav")
    p.add_argument("--text-out", help="output text transcript (file mode); default matches output wav with .txt")
    p.add_argument("--model", default="3b", help="model size (3b|1b) or a q4 model directory (default 3b)")
    p.add_argument("--text-temp", type=float, default=0.4, help="text sampling temperature (default 0.4)")
    p.add_argument("--minutes", type=float, default=30.0, help="mic session cap (default 30)")
    args = p.parse_args()

    weights_dir = f.resolve_weights_dir(args.model)
    mx.random.seed(299792458)
    if args.mic or args.input == "mic":
        run_mic(
            max_steps=int(args.minutes * 60 * 12.5) + 8,
            weights_dir=weights_dir,
            text_temp=args.text_temp,
        )
    elif args.input:
        infile = args.input
        out = args.out or str(ROOT / "translations" / f"{Path(infile).stem}_translated.wav")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        f.run(
            infile,
            out,
            weights_dir=weights_dir,
            text_outfile=args.text_out,
            text_temp=args.text_temp,
        )
    else:
        p.error("give an audio file path, or --mic for realtime")


if __name__ == "__main__":
    main()

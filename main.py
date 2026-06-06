#!/usr/bin/env python
"""hibiki-zero MLX translation — realtime mic or a file, on q4 + the fast path.

  python main.py path/to/audio.wav      # file  -> translations/<stem>_translated.wav
  python main.py --mic                  # mic   -> speakers, live (Ctrl-C to stop)

Speak/record FR/ES/PT/DE; you get streamed EN text + 24 kHz EN audio. Both modes
use the 4-bit weights via src/infer_mlx_fast (load()/run()). File mode is the
3-thread pipelined path (~3x RT); mic mode runs encode->LM->decode per 80 ms
frame in one worker (~59 ms/frame < the 80 ms budget, so it keeps up live).
"""
import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import infer_mlx_fast as f  # noqa: E402
from moshi_mlx import models, utils  # noqa: E402

FRAME = 1920  # samples @ 24 kHz = one 12.5 Hz codec frame (80 ms)


def run_mic(max_steps: int, weights_dir: Path = f.W):
    import sounddevice as sd

    print("loading q4 weights ...")
    model, lm_config, text_tok, mimi_enc, mimi_dec = f.load(weights_dir)
    other_cb = lm_config.other_codebooks
    gen_cb = lm_config.generated_codebooks
    gen = models.LmGen(
        model=model, max_steps=max_steps,
        text_sampler=utils.Sampler(top_k=25, temp=0.8),
        audio_sampler=utils.Sampler(top_k=250, temp=0.8),
        cfg_coef=1.0, check=False,
    )
    model.warmup()
    mx.eval(model.parameters())

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def on_input(indata, frames, t, status):
        in_q.put_nowait(indata[:, 0].copy())          # (1920,) float32 mic frame

    def on_output(outdata, frames, t, status):
        try:
            outdata[:, 0] = out_q.get_nowait()         # translated EN PCM
        except queue.Empty:
            outdata.fill(0)                            # not ready yet -> silence

    def worker():
        try:
            while not stop.is_set():
                try:
                    pcm = in_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                codes = mimi_enc.encode_step(pcm[None, None, :])          # CPU, GIL free
                codes = mx.array(codes).transpose(0, 2, 1)[:, :, :other_cb]
                tt = gen.step(codes[0])
                tok = tt[0].item()                                       # sync this frame
                if tok not in (0, 3):
                    sys.stdout.write(text_tok.id_to_piece(tok).replace("▁", " "))
                    sys.stdout.flush()
                audio = gen.last_audio_tokens()
                if audio is not None and gen_cb > 0:
                    out = mimi_dec.decode_step(np.array(audio[:, :, None]).astype(np.uint32))
                    out_q.put_nowait(out[0, 0])                          # (1920,) float32
        except ValueError as e:                                         # reached max_steps
            print(f"\n[reached cap: {e}]")
        finally:
            stop.set()

    threading.Thread(target=worker, daemon=True).start()
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
    p.add_argument("--minutes", type=float, default=30.0, help="mic session cap (default 30)")
    args = p.parse_args()

    mx.random.seed(299792458)
    if args.mic or args.input == "mic":
        run_mic(max_steps=int(args.minutes * 60 * 12.5) + 8)
    elif args.input:
        infile = args.input
        out = args.out or str(ROOT / "translations" / f"{Path(infile).stem}_translated.wav")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        f.run(infile, out)
    else:
        p.error("give an audio file path, or --mic for realtime")


if __name__ == "__main__":
    main()

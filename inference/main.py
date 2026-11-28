#!/usr/bin/env python
"""hibiki-zero MLX translation — realtime mic or a file, on quantized or bf16 weights.

  python inference/main.py path/to/audio.wav   # file -> translations/<stem>_translated.wav
  python inference/main.py --mic               # mic  -> speakers, live (Ctrl-C to stop)

You get streamed text + 24 kHz audio. Both modes load the model with hibiki_mlx.model
(load()) and sample as the model directory's config.json says (lm_gen_config);
--model accepts 3b (upstream, staged into weights/hibiki-zero-3b-q4) or any staged
model directory. File mode (hibiki_mlx.file_mode) is the 3-thread pipelined path and ends
the input as training's evaluation does (the source EOS, then silence), stopping at the
text EOS or once the model has said nothing for 5 s after the source ended (QUIET_S); mic
mode (hibiki_mlx.mic) never ends it, and pipelines encode->LM->decode across threads so the live
critical path is just the LM step (~22 ms on the M4 Pro for the 3B at 4 bits, 25 at 6;
budget 80 ms).

Mic mode cancels the speakers' echo (hibiki_mlx.echo) before the input level stage, so the
model does not hear and translate its own translation. It reaches echoes up to ~450 ms late:
the Mac's own speakers, not every Bluetooth or AirPlay speaker.

Both modes pass the source through the voice gate (hibiki_mlx.voice): the model hears
silence until speech starts and wherever no one speaks, never noise alone, 160 ms late.

--speaking vi|en says which language is spoken: both modes then start the stream with a
short primer in that language, unheard, so the two-way model keeps that direction instead
of guessing it from the voice (hibiki_mlx.primer).
"""
import argparse
from pathlib import Path

import mlx.core as mx

from hibiki_mlx.file_mode import run
from hibiki_mlx.mic import run_mic
from hibiki_mlx.model import load, resolve_weights_dir
from hibiki_mlx.primer import PRIMERS

ROOT = Path(__file__).resolve().parent.parent  # repo root (inference/ -> ..)


def main():
    p = argparse.ArgumentParser(description="hibiki-zero MLX translation (mic or file)")
    p.add_argument("input", nargs="?", help="audio file to translate")
    p.add_argument("--mic", action="store_true", help="realtime mic -> speakers")
    p.add_argument("-o", "--out", help="output wav (file mode); default translations/<stem>_translated.wav")
    p.add_argument("--text-out", help="output text transcript (file mode); default matches output wav with .txt")
    p.add_argument("--model", default="3b", help="3b or a staged Hibiki-Zero model directory")
    p.add_argument("--minutes", type=float, default=30.0, help="mic session cap (default 30)")
    p.add_argument("--speaking", choices=sorted(PRIMERS),
                   help="the language spoken; primes the direction (default: the model guesses)")
    args = p.parse_args()

    if not (args.mic or args.input):
        p.error("give an audio file path, or --mic for realtime")
    print("loading MLX weights ...")
    model = load(resolve_weights_dir(args.model))
    mx.random.seed(299792458)
    if args.mic:
        run_mic(max_steps=int(args.minutes * 60 * 12.5) + 8, model=model, speaking=args.speaking)
    else:
        infile = args.input
        out = args.out or str(ROOT / "translations" / f"{Path(infile).stem}_translated.wav")
        run(infile, out, model, text_outfile=args.text_out, speaking=args.speaking)


if __name__ == "__main__":
    main()

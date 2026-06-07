#!/usr/bin/env python
"""Sweep text-sampling temperature over the full CoVoST2 manifest and score each.

Fixed seed per run for reproducibility. For each temp: translate all rows, score
BLEU/chrF/WER, and flag clips whose output starts with a likely spurious lead-in
(a sentence-ending '.','?','!' early in the text => the model committed to a wrong
opener, then restarted)."""
import csv
import re
import sys
from pathlib import Path

import mlx.core as mx
import sacrebleu

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import infer_mlx_fast as f

MANIFEST = ROOT / "remote_dataset" / "covost2_fr_en_test" / "manifest.csv"
SEED = 299792458


def lead_in_flag(text: str) -> bool:
    # spurious opener => an early sentence boundary within the first few words
    m = re.search(r"[.?!]", text)
    if not m:
        return False
    head = text[: m.start()]
    return 0 < len(head.split()) <= 4


def main():
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    pre = f.load(f.W)
    model, lm_config, text_tok, _, _ = pre
    for temp in (0.8, 0.4, 0.0):
        out_dir = ROOT / "translations" / f"covost2_temp{temp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        hyps, refs, flagged = [], [], []
        mx.random.seed(SEED)
        for row in rows:
            wav = row["audio_file"]
            stem = Path(wav).stem
            enc, dec = f.make_mimi(f.W, lm_config)
            txt_path = out_dir / f"{stem}.txt"
            f.run(wav, str(out_dir / f"{stem}.wav"), text_outfile=str(txt_path),
                  preloaded=(model, lm_config, text_tok, enc, dec), text_temp=temp)
            txt = txt_path.read_text().strip()
            hyps.append(txt); refs.append(row["translation_en"].strip())
            if lead_in_flag(txt):
                flagged.append((stem, txt))
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        print(f"\n##### text_temp={temp}  BLEU={bleu:.2f}  chrF={chrf:.2f}  "
              f"lead-in flags={len(flagged)} #####")
        for stem, txt in flagged:
            print(f"    {stem}: {txt}")


if __name__ == "__main__":
    main()

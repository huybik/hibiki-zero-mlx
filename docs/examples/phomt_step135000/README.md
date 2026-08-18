# PhoMT holdout examples — step 135,000

These are six row-ID-disjoint examples from the 1,068-row PhoMT holdout used in
the teacher-forced seen-versus-unseen evaluation. None of these IDs occurs in
the frozen training manifest. All six rows are marked
`cross_lingual_timbre_matched=true` in the grounded-v2 cache.

Free-running outputs were generated from `model_step135000.safetensors` with
the correct Vietnamese source, seed 42, audio temperature 0.8, text temperature
0.4, and top-k 250 for both streams. The supplied files include Vietnamese
source audio, English reference audio/text, and generated English audio/text.

## Aggregate result

| Metric | Result |
| --- | ---: |
| Nonempty | 6/6 |
| EOS found | 4/6 |
| BLEU | 2.07 |
| chrF | 18.85 |
| WER | 159.18% |
| Generation health | Fail |

The complete raw outputs are in [predictions.csv](generated/predictions.csv),
[metrics.json](generated/metrics.json), and [pairs.jsonl](pairs.jsonl).

## Pair 1 — `phomt_s01071r00154`

- Vietnamese: Tránh sử dụng tiếng lóng.
- English reference: Avoid using slang.
- Step-135k output: Avoid snoring , You've been in snooze. ":
- EOS: yes
- Audio: [Vietnamese source](inputs/phomt_s01071r00154_vi.wav) ·
  [English reference](inputs/phomt_s01071r00154_en_reference.wav) ·
  [generated English](generated/0000_phomt_s01071r00154_vi_step135k_mono.wav)

## Pair 2 — `phomt_s01071r00260`

- Vietnamese: Bạn có thể tìm kiếm bằng chứng chống lại suy nghĩ này.
- English reference: You can look for evidence to the contrary.
- Step-135k output: You can look for a katyl, taeky, "I" in "-no" ways. ""). ")))
- EOS: yes
- Audio: [Vietnamese source](inputs/phomt_s01071r00260_vi.wav) ·
  [English reference](inputs/phomt_s01071r00260_en_reference.wav) ·
  [generated English](generated/0001_phomt_s01071r00260_vi_step135k_mono.wav)

## Pair 3 — `phomt_s01071r00217`

- Vietnamese: Cố gắng không vội vã đưa ra kết luận.
- English reference: Try not to jump to conclusions.
- Step-135k output: Try to be sure you're not giving your ⌘ ☘ ⌘ ⌘ ⌘ ⌘ ⌘ ⌘ ⌘ ⌘
  ⌌ ⌌ The best of ⌘ ☌ people ⌘) but ⌄Well - ⌘) the other way around ⌘) See,
  you're doing it. ⌘. ⊞)
- EOS: no
- Audio: [Vietnamese source](inputs/phomt_s01071r00217_vi.wav) ·
  [English reference](inputs/phomt_s01071r00217_en_reference.wav) ·
  [generated English](generated/0002_phomt_s01071r00217_vi_step135k_mono.wav)

## Pair 4 — `phomt_s01071r00340`

- Vietnamese: Nếu muốn buộc tóc sau lưng, bạn chỉ nên buộc hoặc tết lỏng.
- English reference: When pulling your hair back, style it in a loose braid.
- Step-135k output: So if you want to get your hair's back, you'll say,
  "------------- -- or - - - - - - ------re-use"? ,'-------------
  'to-intro your yourself '" ----- - ----o- do - - - - - - '---------or - /
  - - - - -------or" ---
- EOS: no
- Audio: [Vietnamese source](inputs/phomt_s01071r00340_vi.wav) ·
  [English reference](inputs/phomt_s01071r00340_en_reference.wav) ·
  [generated English](generated/0003_phomt_s01071r00340_vi_step135k_mono.wav)

## Pair 5 — `phomt_s01071r00474`

- Vietnamese: Hàn the rất độc đối với con người và thú cưng.
- English reference: Borax is very toxic to humans and pets as well.
- Step-135k output: We're going to make a lot of money. ", "
  -----------sit-top- cold - - - - - and - - - -------knucks------------nation--
  - -------knose -----------
- EOS: yes
- Audio: [Vietnamese source](inputs/phomt_s01071r00474_vi.wav) ·
  [English reference](inputs/phomt_s01071r00474_en_reference.wav) ·
  [generated English](generated/0004_phomt_s01071r00474_vi_step135k_mono.wav)

## Pair 6 — `phomt_s01071r00299`

- Vietnamese: Không siết các cơ bụng, chân hoặc mông.
- English reference: Do n't squeeze other muscles like your abs, legs, or butt.
- Step-135k output: Do not sting , The muscles, , shiver , The muscles, sting.
  sting .. '"
- EOS: yes
- Audio: [Vietnamese source](inputs/phomt_s01071r00299_vi.wav) ·
  [English reference](inputs/phomt_s01071r00299_en_reference.wav) ·
  [generated English](generated/0005_phomt_s01071r00299_vi_step135k_mono.wav)

These examples reconcile the earlier measurements: step 135,000 has low
teacher-forced loss on unseen PhoMT, but its free-running translations remain
semantically weak and can fail to terminate.

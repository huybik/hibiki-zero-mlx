# Tranche translation (VI→EN)

Read when touching `training/tranche/translate.py` or the `mt` env of `training/tranche/box/envs.sh`,
or translating the tranche on the box.
Back to [CONTEXT.md](../../CONTEXT.md) · [real-vi-tranche](real-vi-tranche.md#pipeline) ·
[Running](#running) · [Bake-off](#bake-off)

## Running

`training/tranche/translate.py`, in the env `/workspace/envs/mt` that `box/envs.sh mt` builds (vLLM 0.34.0), runs
Index-Translate-35B-A3B in vLLM in its own process. It reads each segments shard's `id`, `doc` and
`text` from the Hub, translates each row as its own prompt, with up to two earlier rows of its doc
as context, and writes `<root>/<source>/text_en/<shard>.parquet` (`id`, `text_en`); `ingest push
<source> --part text_en` uploads it beside the segments.

- It takes the GPU memory voice and the ASR server hold (0.9 of it), so translation and voice
  take turns: stop voice and the ASR server, run `translate.py <source> --shards FIRST:` over the
  pushed shards, `ingest push <source> --part text_en`, then restart them. Eight shards
  (`GROUP = 8`, `--group`) go into each generate call, and the next group is read meanwhile.
- Engine & pipeline optimizations (2026-11-28, [translate_5090_optimization_plan](../translate_5090_optimization_plan.md)):
  - `box/envs.sh mt` pins `vllm==0.34.0` + `flashinfer-jit-cache`; `engine()` sets
    `VLLM_HAS_FLASHINFER_CUBIN=1`, passes `language_model_only=True` (falling back to
    `limit_mm_per_prompt`), `async_scheduling=True`, and `max_model_len=2048` (down from 4096).
  - On `<64 GiB` cards (RTX 5090 NVFP4), `engine()` defaults to **`mamba_ssm_cache_dtype="bfloat16"`**
    + **`kv_cache_dtype="fp8"`** + **`max_num_seqs=512`** (at `gpu_memory_utilization=0.95`,
    `max_num_batched_tokens=16384`). By default vLLM honors `config.json`'s `mamba_ssm_dtype: "float32"`
    (61.4 MiB/seq), which capped Mamba cache blocks at 317–339 and refused `max_num_seqs=512`; pairing
    `bfloat16` SSM (31.4 MiB/seq) with `fp8` KV keeps the hybrid attention block size aligned at 1,088
    tokens (1.49% padding) and gives 117,145 KV tokens at 512 concurrent sequences. Cards ≥64 GiB keep
    `auto` KV/SSM and `512` sequences; all knobs are overridable via CLI flags (`--max-num-seqs`,
    `--max-model-len`, `--max-num-batched-tokens`, `--kv-cache-dtype`, `--mamba-ssm-cache-dtype`,
    `--moe-backend`, `--mtp-tokens`, `--model`, `--quantization`).
  - Per-row `row_max_tokens()` tightens `SamplingParams.max_tokens` to `bad()`'s character gate
    (bucketed to multiples of 32, conservative 2.2 chars/token + 16-token margin) so runaway rows
    abort 2–5× sooner without clipping valid replies; prompts are sorted longest-first (LPT) before
    each generate call, deduplicated on greedy attempt 0, and pre-tokenized via
    `apply_chat_template` + parallel Rust batch tokenizer into `TokensPrompt`.
  - Optional `--context-rows` / `--max-context-chars` and multi-segment window packing (`--window K`,
    `--window-context-rows`) pack consecutive rows of the same `doc` into numbered `n|English`
    windows with all-or-nothing acceptance (`parse_window` + `bad()` on every line) and automatic
    fallback to `K=1` single-row translation on any failed window. Default stays `--window 1`
    until a full-shard YODAS sample audit verifies zero cross-line drift.
  - Measured on box `62513802` (RTX 5090, `vllm==0.34.0`, `Index-Translate-35B-A3B-preview-FP4`@`111ff5e9`,
    4,000 YODAS shard 0000 rows + 2,000 Archive 50 rows + 149 FLEURS dev; raw logs in
    `/Volumes/data/hibiki-zero-evals/box-62513802/translate-5090opt-2026-11-28/`):
    - Baseline (`main`, `max_model_len=4096`, `seqs=256`, `auto` KV/SSM): **42.5 rows/s** YODAS,
      **97.5 rows/s** Arch 50, FLEURS 64.75 / 37.33, Arch 50 70.07 / 41.60.
    - New 5090 default (`bfloat16` SSM + `fp8` KV + `seqs=512`, `FLASHINFER_CUTLASS` MoE, `K=1 ctx=2`):
      **50.5–51.8 rows/s** YODAS (**1.19–1.22×**), **110.0 rows/s** Arch 50 (**1.13×**), FLEURS
      64.47–64.67 / 36.74–36.94, Arch 50 **70.44 / 42.20**, YODAS vs baseline ref 87.58 chrF.
      (`kv_cache_dtype=fp8` alone without `bfloat16` SSM slowed `seqs=256` to 39.7 rows/s; `marlin`
      MoE was 47.8 rows/s, `b12x` 40.0 rows/s; `--mtp-tokens 1` failed at load because
      `model_mtp.safetensors` is BF16 without NVFP4 scales / ignore entries.)
    - Opt-in `--context-rows 1 --max-context-chars 120`: **59.6 rows/s** YODAS (**1.40×**), **121.9 rows/s**
      Arch 50 (**1.25×**), FLEURS **64.80 / 36.91**, Arch 50 69.96 / 41.53.
    - Opt-in `--window 4` / `6` / `8` (`--window-context-rows 1`): **98.9–99.7%** YODAS and
      **99.8–100.0%** Arch 50 window acceptance; **70.3–70.7 rows/s** YODAS (**1.66×** on 4k slice)
      and **168.6 / 179.1 / 189.0 rows/s** Arch 50 (**1.73–1.94×**), Arch 50 **70.76–71.38 / 45.22–46.52**
      (+0.7–1.3 chrF / +3.6–4.9 BLEU over baseline).
- Speed (2026-11-07, GigaSpeech 2, ~21,700 rows an archive): 204 rows/s, an archive in ~1.8 min,
  at 0.9 of the GPU, 512 sequences and 16k-token prefill steps; 125 rows/s at 0.75 beside the
  ASR server with 256 (PyTorch's allocator hitting out-of-memory). The GPU sits near 58% and
  430 W while vLLM's engine loop and the main process each hold a core: the engine is
  CPU-bound. Chat templates and tokenizing take 0.08 ms a row. py-spy cannot attach in this
  container (no ptrace).
- On a card under 64 GiB (`engine` reads its memory from NVML, not torch, which would start CUDA
  in the main process; a 48 GB card would keep ~8 GB beside the FP8 weights) it loads IndexTeam's
  own NVFP4 build (Blackwell only),
  `Index-Translate-35B-A3B-preview-FP4`@`111ff5e` (21 GB; the FP8 35B does not fit a 32 GB RTX 5090),
  at 0.95 of the GPU (`translate.py`'s default there) and 256 sequences: at 0.9 only 232 sequences'
  linear-attention state fits and the engine refuses 512. On the 5090 (2026-11-25): FLEURS dev chrF 64.5, BLEU 36.4
  (the FP8's 65.1 in the [bake-off](#bake-off), 149 sentences); GigaSpeech 2 archive 60, 21,810
  rows in 225 s (97 rows/s, the GPU 92% busy), none failed, chrF 90.3 against the FP8's English.
  The YODAS pilot (2026-11-25, 23 shards, 264,665 rows of 213 Vietnamese characters on average, ~3×
  GigaSpeech 2's): 44 rows/s, 99.8 min, the GPU ~89% busy at ~460 W; 10 rows failed every try, reruns translated
  9 and the last was blanked by hand ([tranche-yodas](tranche-yodas.md#voice)). Its
  KV cache holds ~137k tokens beside the 21 GB of weights, so longer rows run fewer at once.
- GigaSpeech 2's transcripts are all capitals: a transcript all in capitals goes in lowercased,
  because kept, the model answers in capitals (Muse did on 5.7% of archive 50's rows), and such
  rows broke in the TTS 59.8% of the time against 21.3% (archives 13–24).
- The first try is greedy with thinking off. A reply is tried again, sampled (temperature 0.7,
  top-p 0.8, top-k 20, a fresh seed each try: the engine's own is fixed), up to six tries in all
  (the later ones cover a few rows a group, in seconds). The second half of the tries, and every try
  of a row that failed in an earlier run, drop the context rows. A reply is tried again when it is:
  - cut at 512 tokens, or empty;
  - holding letters of a script its Vietnamese lacks. The 35B writes some Sino-Vietnamese names
    in Chinese ("King Tran Thai宗" for Trần Thái Tông), so a row's later tries may not emit the
    token ids its earlier replies wrote such letters with (`logit_bias` −100; exact ids, since a
    letter also hides in multi-letter tokens such as 太宗 and 江湖 and in byte pieces, which decode
    to U+FFFD; the newest 1,000, under vLLM 0.33's 1,024 a request. Its `bad_words` would block
    single-letter tokens only, and raises past 128). Burst 2's rerun translated all 15 such rows
    burst 1 had left null;
  - holding brackets its Vietnamese lacks ("[Name]", "[a specific time]");
  - left in Vietnamese: two of 25 frequent Vietnamese words, each with a diacritic. Words that
    are also name syllables stay out: ASCII ones (trong, khi: "Nguyen Phu Trong" is right), and
    như, lại, từ, which a name left with its diacritics holds ("Như Xuân"). Some rows come back as
    their own Vietnamese, capitalized, every try ("Bà Hằng kể lúc đó…"), so the row's later tries
    may not emit the tokens that reply wrote Vietnamese letters with;
  - on a try under such bans, or for a row that failed in an earlier run, holding any Vietnamese
    letter (after NFC) or combining mark, or two different ones of 23 of those words written
    without diacritics, lowercase and whole ("do" and "den" left out: they are English; Thi in a
    name, I've and co-founder don't count). The model dodges a
    ban with the word's first token and another ending: Chị is 'Ch'+'ị', so "ChỊ Huong", "Bá
    Hàng", "Cà khịa" for đá cầu (4 of the 42 rows of burst 2's rerun). Such a reply bans its
    Vietnamese-letter tokens too. Some dodges stay English-looking and pass ("Chia Hoa" for chị
    Hòa, "Lawyer Thang" for luật sư Thơm). Over all 3.65M local GigaSpeech 2 rows the folded
    words fired on 64 before "den" left and distinct words counted, about half correct English
    ("gambling den … gambling den", "de la Mora"); a Vietnamese letter is in 0.54% (names kept
    with diacritics), so only strict rows are held to it. 13 capital letters (Ẻ, Ỗ, Ũ…) have no
    token and decode as byte pieces, so they reject a reply but are never banned;
  - from a prompt with context rows, longer than 2.5× its Vietnamese's characters plus 20: the
    model now and then translates the context rows too (1 in 2,000 rows of the bake-off; English
    runs ~1.2× the Vietnamese). Without context rows, longer than 4× plus 40: only a reply that
    runs on.
- Dropping the context costs meaning on some rows: the tight gate rejects a right expansion with
  context ("nội ngoại sản nhi" as "internal medicine, surgery, obstetrics and gynecology,
  pediatrics", 65 characters against 62.5), and the context-free tries then write "internal and
  external, products and children" (2 rows); 6 of burst 2's 42 rerun rows lost meaning so
  ("Bác sĩ Bạch Thị | Chính cảnh báo" as "The main warning"). A looser gate (2.5× plus 50) would
  keep both expansions but let short rows' leaks through; at ~1 row in 50,000 it stays.
- Burst 1 (archives 51–156, three tries, 2026-11-07) left 118 rows null in 67 archives; a rerun
  with the Chinese-token ban left 42 (about half echoed Vietnamese, half context leaks or
  expansions); a rerun with context-free tries and the echo ban left none, but 6 of its 42 rows
  dodged the ban (above) and were set null again before the push. Their rerun on 4f81680 wrote
  all six in ASCII: "Shuttlecock kicking" for đá cầu, but names still guessed ("Chic Huong" for
  chị Hương, "Lawyer Throm"). Burst 2 translated archives 159–215 with one row failing every try
  (giga/185-325-10, "[crowned]" in brackets each time), so archive 185 is held from voice; the
  35B was deleted after the burst (203 archives translated, more than voice reaches by the
  rental's end).
- The 35B keeps the diacritics of 0.73% of its rows' names ("Đỗ Hoàng Việt"; Muse 0.05%) despite
  the prompt; only a row already retried under bans is rejected for them.
- A row failing every try keeps `text_en` null, is logged with its last reply, and holds its
  shard back from voice. A rerun tries only the null rows, sampled from the first try; a row
  already written is never translated again, as it may be voiced.
- A run stops when more than half the rows of a try over more than 5,000 fail it, or once more
  than 1% of the rows it has tried fail every try (checked after each group, past 5,000); a lock
  keeps a second run off the same folder. Shards are written `GROUP` at a time (default 8), once all
  their rows are done, so a crash loses the group in flight.
- Muse Spark 1.3 Contributor through OpenRouter translated the first tranche and GigaSpeech 2
  archives 13–50 (2026-11-05/06), in batches of up to 40 consecutive rows: all 840,520 rows of
  the first tranche in ~55 min for $9.75. It wrote Chinese mid-sentence in 22 of the first
  24,681 rows ("the keeper of the江湖 ledger").

## Bake-off

Bake-off, 2026-11-06, vLLM 0.33 beside the ASR server, greedy, thinking off. FLEURS dev: its 149
sentences against their English. Archive 50: its first 2,000 rows, lowercased, against Muse's
English, both lowercased (Muse's misplaced lines count against every system). chrF/BLEU:

| Model, prompt | FLEURS dev | Archive 50 |
|---|---:|---:|
| Index-Translate-35B-A3B, a row per prompt, short | 65.1 / 37.7 | 70.5 / 42.6 |
| Index-Translate-9B, a row per prompt | 64.7 / 36.8 | 69.1 / 40.4 |
| Index-Translate-9B, a row per prompt, short | 64.8 / 36.7 | 68.7 / 40.1 |
| Index-Translate-9B, the card's prompt (Chinese, no context) | 63.6 / 35.4 | 64.4 / 34.4 |
| Index-Translate-9B, Muse's 40 numbered lines | 63.8 / 36.4 | 66.2 / 41.1 |
| Hy-MT2-30B-A3B, a row per prompt | 63.1 / 35.1 | |
| Hy-MT2-30B-A3B, Muse's 40 numbered lines (39 replies split) | 58.8 / 29.3 | 54.6 / 26.4 |

- "A row per prompt": the fragment instruction (translate exactly, leave an unfinished sentence
  unfinished), the doc's two earlier rows as context, names without diacritics, no brackets.
  Without the last two rules 8.3% of rows kept Vietnamese diacritics on names (Muse 0.05%), and
  a few added `[he]`. In 40-line batches the 9B moved words between neighbouring lines.
- On FLEURS, Gemini 2.5 Flash scored 66.8, Muse 66.6, Gemini 3.1 Flash-Lite 66.3, DeepSeek V3.2
  64.8 and Claude Haiku 4.5 63.6 (OpenRouter, Muse's 40-line prompt).
- Speed on archive 51 (19,707 rows), served over HTTP to a Python client: the 9B 70 rows/s
  (prefix caching on: each sequence's aligned state took ~1,000 tokens of cache and capped the
  batch near 345), 84 rows/s with it off; the 35B 110 rows/s. The GPU sat at 46% and ~300 W while
  the client's 512 threads held a core: in-process generation replaced the server.


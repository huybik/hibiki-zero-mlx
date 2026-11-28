# VieNeu TTS both ways

Read when touching `training/tranche/voice.py`, its engine `training/tranche/vieneu/` (`engine.py`, `frame.py`,
`onnx_graph.py`; its attention kernel `training/decode_attention.py`), `training/tranche/asr.py` or the `vieneu` env of
`training/tranche/box/envs.sh`, or voicing a row in its own speaker's voice on the box, either way:
the tranche's English targets, or the EN-VI dataset's Vietnamese.
Back to [CONTEXT.md](../../CONTEXT.md) · part of [real-vi-tranche](real-vi-tranche.md) · [Choice](#choice) ·
[Which TTS voiced what](#which-tts-voiced-what) · [The check](#the-check) · [English](#english) ·
[Vietnamese](#vietnamese) · [Running](#running)

## Choice

- VieNeu-TTS v3 Turbo voices both ways on the box (user, 2026-11-28): `voice.py run <source>` the tranche's
  English, `voice.py run en_speech` the EN-VI dataset's Vietnamese where the dataset's is broken. Its GPU path and
  speed: [tranche-tts-scaleup](tranche-tts-scaleup.md#optimizing-vieneu). The MLX port that voiced the dataset
  on the Mac stays for the Mac's nights ([vi-tts](vi-tts.md)).
- How it got there:
  - Kyutai's TTS 1.6B, the paper's, is out: its voice conditioning is closed and only precomputed embeddings
    ship.
  - The first tranche's bake-off (2026-11-05, the H100; its code, `bakeoff.py`, is in git history): Apache
    models of at least 24 kHz that clone from any clip. Ruled out before it: non-commercial weights (F5-TTS,
    MaskGCT, Llasa, Spark-TTS, XTTS-v2, OpenAudio S1-mini, Fish S2), licences barring training on their
    output (IndexTTS-2, Higgs Audio v2), VibeVoice (an audible disclaimer), MegaTTS 3 (no cloning
    encoder), OpenVoice v2 (22.05 kHz), LongCat-AudioDiT (Chinese and English only). 281 real Vietnamese
    clips, each cloned to speak an English line, round-tripped through Mimi; voice match is WeSpeaker
    ResNet293-LM against the row's own clip (0 = another speaker, 1 = the speaker's other clip):

    | Mode | Broken | Voice match | RTF |
    |---|---:|---:|---:|
    | CosyVoice3 cross-lingual | 0.0% | 0.11 | 0.19 |
    | Qwen3-TTS x-vector | 1.4% | 0.21 | 0.16 |
    | VoxCPM2 reference only | 3.6% | 0.29 | 0.08 |
    | Qwen3-TTS with transcript | 5.0% | 0.31 | 0.16 |
    | VoxCPM2 with transcript | 8.2% | 0.34 | 0.08 |
    | dots.tts with transcript | 8.9% | 0.40 | 0.32 |
    | dots.tts x-vector | 8.2% | 0.43 | 0.20 |

    The user's blind listening on 20 rows (closest voice: VoxCPM2 11, Qwen3-TTS 7, dots.tts 2; the highest
    voice match lost most often, so the embedding ranks modes but does not decide) chose VoxCPM2 with
    transcript, run batched on nano-vllm-voxcpm.
  - For YODAS's ~20k h (2026-11-26), VieNeu-TTS v3 Turbo optimized ran 3.1× VoxCPM2 optimized, at 7.7% broken
    against 3.8–4.5% and voice match 0.60 against 0.69, and is the scale-up's TTS by the user's 3× rule;
    ZONOS3-Mini broke 31% ([tranche-tts-scaleup](tranche-tts-scaleup.md#finding)).

## Which TTS voiced what

- The tranche's English: PhoAudiobook, viVoice, GigaSpeech 2 and YODAS shards 0000–0003 by VoxCPM2 with
  transcript on nano-vllm-voxcpm 2.0.5, at 10 flow-matching steps until 2026-11-08 ~00:40 UTC+7 and 8 after
  (within noise of 10: broken +1.0 point, kept WER median +0.003, voice match unchanged); YODAS from shard 0004
  by VieNeu ([tranche-tts-scaleup](tranche-tts-scaleup.md#in-production)).
- The EN-VI dataset's Vietnamese: VieNeu in MLX on the Mac and on anquachdev's hardware, all 1,835 shards by
  2026-10-27 ([vi-tts](vi-tts.md)); where `voice.py run en_speech` has run, its broken rows by VieNeu on the box
  ([Vietnamese](#vietnamese)).

## The check

- Every try is heard by Qwen3-ASR-1.7B (`qwen-asr-serve` from env `asr`, `restart.sh asr`, in FP8:
  [tranche-tts-scaleup](tranche-tts-scaleup.md#in-production); the client `asr.Client`, YODAS's ingest's too), its
  language forced (`language` and `to_language` `en` or `vi`), and is broken when its error is over `BROKEN` or its
  pace (seconds per character of the line) is off `PACE` (0.0636) by 2× either way:
  - English: WER under Whisper large-v3's English normalizer, the line unbracketed first (the normalizer deletes
    text in brackets, which scored correct readings of a parenthetical broken), over 0.2;
  - Vietnamese: CER over `filters.spoken` (syllables, numbers written out; `asr.cer`, as ingest scores its ASR), over 0.15,
    the gate's `filters.MAX_CER`. The Mac's voicing ran 0.060–0.068 s a character (medians of cv27-00060,
    libriheavy-large-00012-00103, libriheavy-medium-00100), close enough to English's for one `PACE`.
- A row is voiced once, broken or not (from the scale-up on, 2026-11-06; the first tranche ran up to 3 tries, the
  one of lower error kept): time, not rows, bounds it, and a retry costs more TTS time than its rows give back. On
  six GigaSpeech 2 shards (VoxCPM2):

  | Tries | Rows kept | Kept Vietnamese h per h of English voiced |
  |---:|---:|---:|
  | 1 | 77.1% | 0.660 |
  | 2 | 85.8% | 0.607 |
  | 3 | 89.0% | 0.571 |

- At most 512 transcript tokens a check (`asr.MAX_TOKENS`): uncapped, greedy decoding looped on an output to the
  model's 65k context past any timeout (one in the first ~8,000); capped, it comes back long and scores broken.
- The server stalls past a 60 s try now and then and recovers within minutes; 3 tries ended a run three times in
  two days (2026-11-06/07), so a check tries 10 times, 10 s apart (`ASR_TRIES`); an HTTP error stops the run at once.
- Measured, not shipped (2026-11-07/08): Qwen3-ASR-0.6B served 122 requests/s against the 1.7B's 138 and agreed
  with it on broken on 89% of rows (the 1.7B with itself on the Opus round trip: 95%), flagging 3 points more.

## English

`voice.py run <source>` voices each tranche segments shard on the Hub that has its `text_en` and `asr_vi` but no
`en_audio`, except those held out for evaluation (`repo.SOURCES[...].heldout`,
[giga-heldout](sft-evaluation.md#evaluation-sets)); each row from its own Vietnamese clip, as the paper conditions
on the source speaker.

- Rows not voiced (`en_tries` 0, the other English columns null): those the cache build drops on the Vietnamese
  side alone (`filters.vi_gate` on asr_vi: language, CER over 0.15, a nghìn/triệu/tỷ the transcript lacks; 5.5%
  of GigaSpeech 2's rows), and those whose `text_en` is missing or has nothing speakable. So a shard waits for its
  asr_vi (YODAS's from ingest) as for its text_en.
- Still broken after the check: VoxCPM2 0.2–2.2% of PhoAudiobook's rows a shard, ~1.5% of viVoice's, 10.2–11.9%
  of GigaSpeech 2's at 3 tries; VieNeu 6.1–8.7% of YODAS's at 1. On GigaSpeech 2's shard 0, 2,007 of the 2,085
  broken rows fail on WER, 78 on pace; clips under 2 s break most (23%, against 7% from 3 to 8 s), where one
  Vietnamese name the ASR spells otherwise ("Binh Dinh" heard as "Benin") is over 20% of the words. Of 12 sampled,
  5 were that ASR noise and 7 real faults (a word repeated or dropped, a number changed, a Vietnamese clause), so
  the rule stays and the cache drops them (`en_broken`).
- Prompted with the Vietnamese transcript, VoxCPM2 sometimes spoke a Vietnamese clause before the English (~0.05%
  of rows pass the WER check, e.g. pho/00003/00588), so the cache build also drops rows whose `en_asr` has
  Vietnamese letters absent from `text_en` ([tranche-cache](tranche-cache.md#gates)).
- The kept audio is Ogg Opus from 24 kHz (Mimi's rate), which decodes at 48 kHz whatever its input: readers
  resample. VoxCPM2 overshot full scale on 3% of PhoAudiobook rows and 15% of viVoice's (at most 0.23% of
  samples, hard-clipped by the encode).

## Vietnamese

`voice.py run en_speech` checks the EN-VI dataset's Vietnamese, as the Mac voiced it, against its `text_vi` and
voices again the rows it is broken on, from the row's English clip; it reads the dataset's copy in the tranche
repo (`en_speech/train/`, 1,832 shards; the 3 validation shards are evaluation rows, left as they are).

- Each shard's rows are first heard as the dataset holds them (try 0, through the checking processes, a shard
  ahead of the voicer), then those broken, or without audio, voiced once; the better of the two is
  kept. The voicer is the English one's: the same reference cut, denoiser gate and sampling.
- It writes `en_speech/vi_check/<shard>` (id, `vi_audio` as Ogg Opus from 48 kHz where a new try is kept and null
  where the dataset's stays, then of the kept audio `vi_duration_s`, `vi_asr`, `vi_cer`, `vi_tries` (0: the
  dataset's, heard only) and `vi_broken`), pushed to the tranche repo every 30 min; a row without `text_vi` is
  neither heard nor voiced (all null, `vi_tries` 0). Each shard downloads beside the root and is deleted once
  read (~447 GB in all).
- Measured on two shards (2026-11-28, records in `/Volumes/data/hibiki-zero-evals/box-62513802/vieneu-both-2026-11-28/`,
  with clips of the Mac's broken Vietnamese against the box's):

  | Shard | Rows | Broken as the dataset holds them | Fixed by one re-voice | Broken after |
  |---|---:|---:|---:|---:|
  | cv27-00060 | 8,892 | 7.05% | 61% | 2.74% |
  | libriheavy-medium-00100 | 2,617 | 2.06% | 67% | 0.69% |

  Common Voice's ~180–220 rows/s were heard (4.6 s of Vietnamese a row); Libriheavy's ~12.5 s rows were not timed
  apart. The corpus is 1,745 Libriheavy shards (409 GB) and 87 Common Voice (34 GB). In the same run YODAS 0010's
  English came out as main voiced it: 6.40% broken on both, kept WER median 0.031 against 0.032.
- Its consumers: `cache.en_speech --vi-check <copy of en_speech/vi_check>` takes the new audio of the rows re-voiced
  and rejects those still broken or never heard (every train shard needs its part), and records the part in each
  shard's `source`; `cache.noisy` then fetches the same clips for a Vietnamese source
  ([sft-data-caches](sft-data-caches.md#builds), [noisy-source](noisy-source.md)).

## Running

- Env `/workspace/envs/vieneu` (`box/envs.sh vieneu`, with `asr` for the server): `restart.sh voice` (YODAS,
  rerun as shards arrive) or `restart.sh voice-en-speech` (to the end). Stop a run by its PID file
  (`/workspace/logs/voice.pid`, `voice-en-speech.pid`), never `pkill -f`: its spawned workers look like vLLM's.
- Loading and checking run in spawned processes: as threads their Python held the GIL and starved the GPU loop
  (GPU at 2%). Pools take shares of the container's quota (`cpu_budget()`, cgroup `cpu.max`, not `nproc`): the
  loader a quarter. The 32 checking processes stay fixed: they are the ASR requests in flight (8 each,
  `CHECK_ROWS`), not CPU work. One ASR API process featurizes ~11 requests/s, so the server runs a third of the
  quota's worth.
- The box's container allows 6,656 threads: ffmpeg defaults to a decoder thread per host CPU, so PyAV codecs get
  `thread_count = 1`; PyAV's `AudioResampler` holds a thread per host CPU while it lives (4,075 threads from 20
  decoding threads), so `voice.py` decodes without it (`training.audio.decode`, the cache build's decoder); its
  Opus encoder (`training.audio.decode.encode_opus`) builds one too, so a semaphore shared by
  the checking processes lets 4 encode at once.
- A crashed run drops its queued checks, so its GPU memory frees within minutes, except behind a check still
  trying a stalled server (up to `ASR_TRIES` × ~70 s): a hung server surfaces ~20 min after it hangs, one that
  refuses connections at once.

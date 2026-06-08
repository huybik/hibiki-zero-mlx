# Audio-NTREX-4L Data Spec

Dataset: `kyutai/Audio-NTREX-4L`

## Summary

`Audio-NTREX-4L` is a long-form multilingual speech translation dataset for
evaluating speech translation models from French, Spanish, Portuguese, and German
to English on multi-sentence utterances.

It is built from NTREX text translation data. The source-language text is grouped
into multi-sentence utterances, then synthesized into speech with commercial TTS
systems. The target side is English text only. Audio generation is conditioned
using voices from the multilingual CML-TTS dataset.

This dataset is useful for validation and evaluation. It is not directly enough
for Hibiki-Zero speech-to-speech fine-tuning because it does not include English
target audio.

## Dataset Shape

| Item | Value |
| --- | --- |
| Hugging Face dataset | `kyutai/Audio-NTREX-4L` |
| Original data | NTREX / NTREX-128 |
| Task | Speech translation evaluation |
| Source languages | French, Spanish, Portuguese, German |
| Target language | English |
| Source modality | Audio + source text |
| Target modality | Text |
| Splits | `valid`, `test` |
| Rows per split | 1,800 |
| Total rows | 3,600 |
| Unique source texts per language | 300 |
| Average source duration | About 45 seconds |
| Audio source | Synthetic TTS |
| TTS systems | ElevenLabs, Cartesia, Gradium |
| Voice conditioning | Multilingual CML-TTS voices |
| License | CC BY-NC-SA 4.0 |

## Dataset Construction

The dataset is built from these NTREX-128 text files:

| Language | File |
| --- | --- |
| English | `newstest2019-ref.eng-US.txt` |
| French | `newstest2019-ref.fra.txt` |
| Spanish | `newstest2019-ref.spa.txt` |
| Portuguese | `newstest2019-ref.por.txt` |
| German | `newstest2019-ref.deu.txt` |

Construction process:

1. Select 300 groups of consecutive English NTREX lines from the same original
   document.
2. Use the corresponding source-language NTREX lines to form multi-sentence
   source texts and aligned English target texts.
3. Define `id` as a hash of the ordered NTREX line indexes in each group.
4. Clean source and target text by removing parenthetical elements to make the
   text better suited for natural speech.
5. Synthesize each source text into 3 audio versions, each with a different TTS
   system and voice conditioning.
6. Transcribe synthesized audio with `openai/whisper-large-v3`.
7. Check Word Error Rate against `source_text` to validate TTS quality.
8. Split into balanced `valid` and `test` sets, keeping all pairs with the same
   `target_text` in the same split.

Each language keeps 150 different `id` values in each split.

## Load Pattern

Use streaming mode for inspection or data preparation. By default, the `datasets`
library tries to decode audio and may require `torchcodec`. To avoid that and
work with raw audio bytes, cast `source_audio` with `decode=False`.

```python
from datasets import Audio, load_dataset

ds = load_dataset("kyutai/Audio-NTREX-4L", split="valid", streaming=True)
ds = ds.cast_column("source_audio", Audio(decode=False))
example = next(iter(ds))
```

With `decode=False`, `example["source_audio"]` has this shape:

```python
{
    "bytes": b"...",
    "path": "3242e52a7803dc23_lang-fr.wav",
}
```

## Schema

| Column | Type | Meaning |
| --- | --- | --- |
| `id` | `string` | Stable hash-like ID for the grouped NTREX lines |
| `source_language` | `string` | Source language code, e.g. `fr`, `es`, `pt`, `de` |
| `target_language` | `string` | Target language code, usually `en` |
| `source_ntrex_file` | `string` | Source NTREX reference file |
| `target_ntrex_file` | `string` | English NTREX reference file |
| `ntrex_lines` | `list[int32]` | Consecutive NTREX line indexes grouped into the utterance |
| `tts` | `string` | TTS provider used for source audio |
| `source_audio` | `Audio` | Source-language synthesized speech |
| `source_text` | `string` | Source-language text used to synthesize audio |
| `source_aligned_transcript.text` | `list[string]` | Whisper transcript tokens/segments for the synthesized source audio |
| `source_aligned_transcript.timestamp` | `list[list[float64]]` | Timestamps aligned to `source_aligned_transcript.text` |
| `target_text` | `string` | English reference translation |

## Example Fields

```text
id: 3242e52a7803dc23
source_language: fr
target_language: en
tts: elevenlabs
ntrex_lines: [1065, 1066, 1067, 1068]
source_audio.path: 3242e52a7803dc23_lang-fr.wav
```

`source_text` is long-form source-language text. `target_text` is the English
reference translation for the same NTREX line group.

## Fine-Tune Implications

Hibiki-Zero speech-to-speech fine-tuning needs paired source speech and target
outputs. A practical training record should contain:

| Field | Required for S2ST fine-tune | Source |
| --- | --- | --- |
| `source_audio` | yes | Provided by Audio-NTREX-4L |
| `source_text` | useful | Provided by Audio-NTREX-4L |
| `target_text` | yes | Provided by Audio-NTREX-4L |
| `target_audio` | yes | Not provided; must be synthesized from `target_text` |
| `target_audio_tokens` | yes for cached training | Derived from synthesized `target_audio` with Mimi |

Therefore, `Audio-NTREX-4L` can be used in two ways:

1. Evaluation data: run Hibiki on `source_audio`, compare generated English text
   against `target_text`.
2. Synthetic fine-tune data: synthesize English `target_audio` from `target_text`,
   then cache source/target audio tokens for training.

For a new-language fine-tune, this dataset is best treated as a reference format:
source audio plus English text reference. If the new language is not FR/ES/PT/DE,
build an equivalent dataset by TTS-synthesizing source-language NTREX lines and
using the aligned English lines as `target_text`.

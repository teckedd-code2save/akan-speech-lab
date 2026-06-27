# Artifact Versioning

Status: active convention

All reviewable outputs should use a name that makes the experiment legible
without opening the model card.

## Prefix

Every artifact starts with:

```text
serendepify-gsl
```

`gsl` means Ghanaian Speech Lab.

## Pattern

```text
serendepify-gsl-{task}-{language}-{datasets}-{base}-{method}-v{major}.{minor}
```

Fields:

- `task`: `asr`, `tts`, `punct`, `eval`, `manifest`
- `language`: `ak`, `twi`, `fat`, `ak-multi`
- `datasets`: compact source tokens joined by `-`
- `base`: model family and size
- `method`: training or artifact method
- `version`: review version, not marketing version

## Dataset Tokens

| Token | Dataset |
|---|---|
| `waxal` | `google/WaxalNLP`, config `aka_asr` |
| `gnlp` | `ghananlpcommunity/twi-speech-text-multispeaker-16k` |
| `afys` | `AfriSpeech/youversion-african-speech` |
| `corr` | user correction data with consent |
| `owned` | explicitly consented owned recordings |

## Method Tokens

| Token | Meaning |
|---|---|
| `fullft` | full fine-tune |
| `lora-r{rank}` | LoRA adapter with rank |
| `replay-fullft` | full fine-tune with replay-mixed corpora |
| `replay-lora-r{rank}` | LoRA with replay-mixed corpora |
| `specaug` | SpecAugment used |
| `punct-tagger` | punctuation restoration tagger |
| `manifest-audit` | dataset manifest and audit artifact |

## First ASR Review Output

The first new model pass should be:

```text
serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1
```

Meaning:

- `serendepify-gsl`: Ghanaian Speech Lab artifact
- `asr`: speech-to-text model
- `ak`: Akan-family target
- `waxal-gnlp`: Waxal plus GhanaNLP, after harmonization
- `whisper-small`: base model
- `replay-fullft`: full fine-tuning with replay mixing, not GhanaNLP-only continuation
- `v0.1`: first external review candidate

Expected Hugging Face model repo:

```text
teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1
```

## Required Card Fields

Every published artifact must state:

- code name
- parent/base model
- exact dataset mixture and manifest hash
- preprocessing policy
- punctuation policy
- training method
- whether LoRA or full fine-tuning was used
- decoding settings
- WER/CER by held-out set
- known regressions
- failure examples
- whether the artifact is production, review, or diagnostic


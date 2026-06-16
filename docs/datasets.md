# Dataset Plan

## First ASR Datasets

### Waxal Akan ASR

- Dataset: `google/WaxalNLP`
- Config: `aka_asr`
- Purpose: reproduce and beat the existing Whisper/Waxal Akan baseline.
- Existing baseline: `teckedd/whisper_small-waxal_akan-asr-v1`, reported WER `34.2849`.
- Rule: this baseline is not the goal. It is the floor we should beat through cleaner preprocessing, better split discipline, and tokenizer strategy.

Prepare a speaker-safe manifest:

```bash
python scripts/prepare_waxal.py \
  --config aka_asr \
  --speaker-safe-split \
  --output data/manifests/waxal_aka_asr_speaker_safe.jsonl
```

### GhanaNLP Twi

- Dataset: `ghananlpcommunity/twi-speech-text-multispeaker-16k`
- Purpose: controlled Twi-specific ASR training and cross-dataset evaluation.
- Rule: run GhanaNLP-only experiments before mixing with Waxal.

## Candidate Supplemental Sources

- `AfriSpeech/youversion-african-speech`
- English-to-Twi translation models/datasets for future text normalization and semantic eval.

## Quality Checks

- License and consent.
- Audio sample rate and clipping.
- Transcript language and dialect labels.
- Speaker leakage across train/eval.
- Empty/English-only/noisy transcripts.
- Too-short or too-long clips.
- Spelling variants and diacritic consistency.
- Domain coverage for health and ecommerce.

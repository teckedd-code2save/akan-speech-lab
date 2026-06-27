# Dataset Plan

## ASR Dataset Understanding

The goal is not to rank datasets as good or bad before we have understood them.
The goal is to characterize each corpus, harmonize what can be harmonized, and
then train from a mixture whose risks are explicit.

### Waxal Akan ASR

- Dataset: `google/WaxalNLP`
- Config: `aka_asr`
- Purpose: reproduce the existing Whisper/Waxal behavior and provide a
  speaker-labeled anchor corpus for regression checks.
- Existing baseline: `teckedd/whisper_small-waxal_akan-asr-v1`, reported WER `34.2849`.
- Rule: this baseline is reference evidence, not the project goal.

Prepare a speaker-safe manifest:

```bash
python scripts/prepare_waxal.py \
  --config aka_asr \
  --speaker-safe-split \
  --output data/manifests/waxal_aka_asr_speaker_safe.jsonl
```

### GhanaNLP Twi

- Dataset: `ghananlpcommunity/twi-speech-text-multispeaker-16k`
- Purpose: controlled Twi-specific ASR training, corpus harmonization, and
  cross-dataset evaluation.
- Rule: do not dismiss GhanaNLP because a blunt continuation failed. Audit its
  transcript style, audio quality, duration distribution, duplicates, and
  spelling conventions, then mix it with replay and regression gates.
- Current Dataset Viewer: 15,560 rows in one train split; the card text says 21,138.
- Usable transcripts after empty-text filtering: 15,372 / 9.51 hours.
- Existing 0.4-30 second training window: 11,830 eligible; 3,542 clips are shorter than 0.4 seconds.
- Published columns: audio, text, duration. No speaker IDs are exposed, so speaker-safe evaluation cannot be claimed.
- Frozen split: normalized-transcript group hash, 90/5/5. This keeps all 1,057 duplicate transcript groups within one partition and yields zero cross-split transcript leakage.

The manifest stores stable `hf://dataset/config/split/row` references rather
than expiring Dataset Viewer audio URLs. The split is useful for transcript-group
isolation, but the lack of speaker IDs means speaker-safe evaluation cannot be
claimed from the public schema alone.

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

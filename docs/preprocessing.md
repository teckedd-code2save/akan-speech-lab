# Preprocessing Strategy

The central bet is that Akan ASR quality can improve substantially before model scale changes.

## Waxal Akan ASR Checks

- Re-split by `speaker_id` for speaker-independent evaluation.
- Drop empty transcripts.
- Flag English-only or wrong-language rows.
- Drop clips shorter than `0.4s` and longer than `30s` for first-pass training.
- Normalize punctuation/casing while preserving Akan diacritics.
- Keep original and normalized transcript in every manifest row.
- Track duration, speaker, source split, and language.

## Why This Matters

The earlier Whisper/Waxal run reached reported WER `34.2849`, but the qualitative output was better than expected. That suggests the model has useful latent capacity and the next improvement may come from cleaner data, stronger evaluation, and better tokenizer/language forcing choices.

## First Commands

```bash
python scripts/prepare_waxal.py \
  --config aka_asr \
  --speaker-safe-split \
  --output data/manifests/waxal_aka_asr_speaker_safe.jsonl
```

Use `--limit 1000` for a cheap local smoke pass before a full manifest build.

Inspect the manifest before training:

```bash
python scripts/inspect_manifest.py \
  data/manifests/waxal_aka_asr_speaker_safe.jsonl \
  --output evals/reports/waxal_aka_asr_manifest_quality.json
```

## Old Notebook Review

When the old notebook is available, save it under:

```text
references/old-whisper-waxal-finetune/whisper_small_waxal_akan_asr_v1_reference.ipynb
```

Then extract likely review clues:

```bash
python scripts/review_old_notebook.py \
  references/old-whisper-waxal-finetune/whisper_small_waxal_akan_asr_v1_reference.ipynb
```

The review should look for hidden issues: split leakage, wrong forced language token, unnormalized labels, no duration filtering, weak eval samples, and accidental train/eval contamination.

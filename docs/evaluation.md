# Evaluation Protocol

## ASR

Primary metrics:

- WER on normalized Akan text.
- CER for diacritic and spelling sensitivity.
- Qualitative examples by domain: health, ecommerce, family care, everyday speech.

Baseline comparisons:

- Existing Waxal Whisper fine-tune: `teckedd/whisper_small-waxal_akan-asr-v1`, reported WER `34.2849`.
- MMS ASR.
- New GhanaNLP-only run.

Promotion rule:

- Do not push a new “best” model unless it improves held-out WER and gives clearly better qualitative Akan output without regressions.
- First target is not to reproduce `34.2849`; it is to beat it materially with a cleaner Waxal Akan split.

Required ASR slices:

- WER/CER overall.
- WER/CER by split.
- WER/CER by duration bucket.
- WER/CER by speaker frequency bucket.
- Qualitative samples with reference, prediction, normalized reference, normalized prediction, and error notes.

## Tiny Local Baseline

Build a fixed preview manifest:

```bash
python scripts/build_waxal_viewer_manifest.py \
  --config aka_asr \
  --split test \
  --limit 5 \
  --output evals/samples/waxal_aka_asr_test_preview.jsonl
```

Materialize audio locally so model evaluation does not depend on streaming signed URLs:

```bash
python scripts/materialize_manifest_audio.py \
  --manifest evals/samples/waxal_aka_asr_test_preview.jsonl \
  --output-manifest evals/samples/waxal_aka_asr_test_preview_local.jsonl \
  --limit 3
```

Evaluate a model on a few rows locally:

```bash
python scripts/eval_asr_manifest.py \
  --model-id teckedd/whisper_small-waxal_akan-asr-v1 \
  --manifest evals/samples/waxal_aka_asr_test_preview_local.jsonl \
  --limit 2 \
  --language yoruba \
  --task transcribe
```

This is not the final benchmark. It is a wiring test that proves model loading, audio loading, normalization, prediction capture, and WER/CER reporting.

If a model is large, cache it explicitly before evaluation:

```bash
python scripts/cache_hf_model.py \
  --model-id teckedd/whisper_small-waxal_akan-asr-v1
```

Render a report:

```bash
python scripts/render_asr_report.py \
  --input-json evals/reports/asr_eval_report.json \
  --output-md evals/reports/asr_eval_report.md
```

## TTS

TTS needs a separate gate:

- speaker consent,
- consistent transcript/audio pairing,
- fixed prompt suite,
- human intelligibility review,
- naturalness review,
- failure examples.

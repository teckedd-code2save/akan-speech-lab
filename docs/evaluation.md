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

## TTS

TTS needs a separate gate:

- speaker consent,
- consistent transcript/audio pairing,
- fixed prompt suite,
- human intelligibility review,
- naturalness review,
- failure examples.

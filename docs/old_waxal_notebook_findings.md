# Old Waxal Whisper Notebook Findings

Reference file:

```text
references/old-whisper-waxal-finetune/serlabs_waxalnlp_twi_asr.py
```

## What Worked

- Used `google/WaxalNLP`, config `aka_asr`.
- Used `openai/whisper-small`.
- Used `WhisperTokenizer` and `WhisperProcessor` with `language="yoruba"` and `task="transcribe"`.
- Disabled `forced_decoder_ids` during generation.
- Reached reported WER `34.2849` and produced qualitatively useful Akan output.

## Potential Misdoings To Fix

- Loaded Waxal train/test directly, with no speaker-overlap check.
- Dropped `speaker_id`, `language`, and `gender` before deeper data diagnostics.
- Used streaming datasets, which makes reproducible sample inspection and deterministic preprocessing harder.
- Did not build a durable manifest with original text, normalized text, split, speaker, duration, and source.
- Used a simple punctuation regex and did not preserve a documented normalization policy.
- Did not compare tokenizer/language strategies against each other.
- Did not bucket WER by duration, speaker frequency, or transcript type.
- Did not preserve a clear baseline eval artifact independent of training logs.

## New Pipeline Response

- Build manifests first.
- Inspect speaker leakage before training.
- Keep speaker/language metadata.
- Run baseline eval before fine-tuning.
- Treat Yoruba proxy as one experiment arm, not a default truth.
- Save qualitative error reports alongside WER/CER.


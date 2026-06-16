# Training Plan

## ASR First

1. Prepare Waxal Akan ASR manifest with a speaker-safe split.
2. Evaluate `teckedd/whisper_small-waxal_akan-asr-v1` on the exact same held-out set.
3. Run preprocessing diagnostics before training.
4. Fine-tune Whisper small with controlled tokenizer/language strategies.
5. Compare WER/CER and qualitative Akan outputs.
6. Prepare GhanaNLP-only manifest and repeat.
7. Mix Waxal + GhanaNLP only after both single-dataset runs are understood.
8. Push to Hugging Face only if better.

## TTS Later

TTS follows ASR because it is easier to fool ourselves with pleasant audio that is not intelligible, faithful, or consent-safe.

## Tokenizer Strategy

Whisper does not officially target Akan as a first-class language token. We should test:

- keeping a stable proxy language token,
- no forced language token,
- tokenizer added tokens only if justified by evaluation.

The earlier successful baseline used a Yoruba-adjacent tokenizer strategy. We should keep that as one controlled arm, not a permanent assumption.

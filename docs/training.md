# Training Plan

## ASR First

1. Audit all 12,752 labeled Waxal Akan rows.
2. Freeze benchmark v1 from the original test split: 99 rows, 33 test-only speakers.
3. Evaluate `teckedd/whisper_small-waxal_akan-asr-v1` under no-language, Yoruba, and English decoder settings.
4. Select no forced language because it ties the stored Yoruba behavior without hard-coding a proxy language.
5. Run a two-step Modal smoke test on 32 train and 16 validation rows.
6. Prepare the full training set on CPU, then fine-tune Whisper small on an L4.
7. Evaluate the best validation checkpoint once on frozen benchmark v1.
8. Compare WER/CER, bootstrap interval, speaker slices, duration slices, and qualitative Akan outputs.
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

The published checkpoint stores Yoruba in its generation config, but clearing it produces identical benchmark predictions. The first clean candidate therefore uses no forced language token. Current references average 2.60 Whisper tokens per word; tokenizer extension remains a later controlled experiment because changing vocabulary and training policy simultaneously would hide which change helped.

## Candidate v2

- Base: `openai/whisper-small`
- Training: full model, FP16, non-reentrant gradient checkpointing
- Optimizer schedule: AdamW defaults, `1e-5`, 200 warmup steps, 1,200 total steps
- Effective batch: 32 (`8 x 4` accumulation)
- Validation batch: 8 on L4
- Selection: lowest validation WER every 200 steps
- Test discipline: benchmark v1 is never used for training or checkpoint selection
- Promotion gate: beat 33.62% benchmark WER and pass manual Ghanaian review

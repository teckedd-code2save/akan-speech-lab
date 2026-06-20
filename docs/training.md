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
9. Prepare GhanaNLP-only manifest and repeat.
10. Mix Waxal + GhanaNLP only after both single-dataset runs are understood.

## GhanaNLP-Only Arm

- Base: `teckedd/whisper-small-waxal-akan-continuation-v1`
- Training data: GhanaNLP only; Waxal is not mixed into this arm.
- Split: stable normalized-transcript groups, 90/5/5.
- Model selection: validation only; the GhanaNLP test partition remains untouched.
- First configuration: 400 steps, `3e-6`, 50 warmup steps, evaluate/save every 100 steps.
- Decoder variable held constant: Yoruba-compatible training prefix inherited from the prior checkpoint.
- Two-step smoke: passed on L4 with 32 train, 12 validation, and 17 untouched test rows.
- Smoke WER: 72.09% before and after two steps. This proves wiring and exposes cross-corpus domain shift; it is not model evidence.
- Required post-training checks: GhanaNLP held-out test, complete Waxal test for catastrophic forgetting, repetition guard, and Ghanaian listening review.

### GhanaNLP Round 1 Result

- Selected checkpoint: step 400 at 84.58% validation WER, down from 165.53% baseline.
- Untouched GhanaNLP test: 160.65% to 99.35% WER; 341 rows better, 163 tied, 67 worse.
- GhanaNLP test repetition collapses: 17 baseline to 1 candidate.
- Waxal full test: 32.77% to 37.80% WER; 407 rows better, 226 tied, 889 worse.
- Waxal paired 95% WER-change interval: +3.67 to +7.10 points.
- Promotion decision: **failed** because absolute GhanaNLP quality remains poor and Waxal regression is material.
- Next experiment: mixed Waxal replay or a GhanaNLP adapter; do not continue GhanaNLP-only full-model training.
11. Push to Hugging Face only if better.

## TTS Later

TTS follows ASR because it is easier to fool ourselves with pleasant audio that is not intelligible, faithful, or consent-safe.

## Tokenizer Strategy

Whisper does not officially target Akan as a first-class language token. We should test:

- keeping a stable proxy language token,
- no forced language token,
- tokenizer added tokens only if justified by evaluation.

The published checkpoint stores Yoruba in its generation config, but clearing it produces identical benchmark predictions. The first clean candidate therefore uses no forced language token. Current references average 2.60 Whisper tokens per word; tokenizer extension remains a later controlled experiment because changing vocabulary and training policy simultaneously would hide which change helped.

## Candidate Results

### Arm A: From base, no language prefix

- Base: `openai/whisper-small`
- Labels: raw Waxal transcripts
- Decoder: no forced language
- Step 200 validation WER: 88.88%
- Decision: stopped; the trajectory was far behind the published run

This is evidence that clearing a trained checkpoint's inference prefix is not equivalent to training Whisper from base without a language prefix.

### Arm B: Continued Yoruba-prefix training

- Base: `teckedd/whisper_small-waxal_akan-asr-v1`
- Labels: conservative lowercase/punctuation normalization matching the successful notebook
- Decoder during training: Yoruba proxy; inference is separately tested with the prefix cleared
- Training: full model, FP16, non-reentrant gradient checkpointing
- Optimizer schedule: AdamW defaults, `5e-6`, 50 warmup steps, 400 continuation steps
- Effective batch: 32 (`8 x 4` accumulation)
- Validation batch: 8 on L4
- Selection: lowest validation WER every 100 steps
- Baseline validation WER: 32.69%
- Step 100: 31.87%
- Step 200: 31.58%
- Step 300: 31.45% (selected)
- Step 400: 31.83% (regressed)
- Test discipline: benchmark v1 is never used for training or checkpoint selection
- Frozen benchmark: 32.65% WER / 12.26% CER versus 33.62% / 12.37% baseline
- Paired bootstrap: 94.1% probability of improvement; 95% difference interval -2.22 to +0.20 points
- Full 1,522-row test: 32.77% WER / 12.47% CER versus 33.84% / 12.74% baseline
- Full-test paired bootstrap: 99.86% probability of improvement; 95% difference interval
  -1.90 to -0.33 points
- Full-test row outcomes: 524 better, 603 tied, 395 worse, 2 severe repetition loops
- Published candidate: `teckedd/whisper-small-waxal-akan-continuation-v1`
- Promotion status: experimental until manual Ghanaian review and validation of the guarded retry policy; repetition-collapse detection is implemented

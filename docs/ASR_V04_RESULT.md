# ASR v0.4 Result: Waxal-Only Run Not Promoted

Artifact: `serendepify-gsl-asr-ak-waxal-whisper-small-only-lowlr-freezeenc-fullft-v0.4`

Hub model: [teckedd/serendepify-gsl-asr-ak-waxal-whisper-small-only-lowlr-freezeenc-fullft-v0.4](https://huggingface.co/teckedd/serendepify-gsl-asr-ak-waxal-whisper-small-only-lowlr-freezeenc-fullft-v0.4)

Modal call: `fc-01KW8F1D116A4T1H6C2Z8QACBZ`

Status: complete. Published for traceability only. This is not a promoted model.

## Configuration

- Base model: `teckedd/whisper-small-waxal-round2-specaug-v1`
- Train rows: 9,138 Waxal rows
- Dev rows: 1,024 Waxal rows
- Method: full fine-tune with frozen encoder
- Learning rate: `1e-6`
- Effective batch size: 16
- Max steps: 600
- SpecAugment: enabled
- Regression stop: enabled
- Best checkpoint emitted by Trainer: `checkpoint-200`
- Actual decision: not promoted because checkpoint 200 regressed against the base model on the same dev slice

## Metrics

| Slice | Base model WER | v0.4 checkpoint WER | Decision |
| --- | ---: | ---: | --- |
| Waxal dev, 1,024 rows | 31.88% | 34.09% | Regression |

Training stopped after the first eval point because `eval_wer` was worse than
the base model's baseline WER on the same Waxal-only dev slice.

## What This Means

This run answered the direct question: a Waxal-only low-learning-rate
continuation from the Round 2 checkpoint does not improve the current ASR path.
The loss improved, but WER got worse, which means the model became less useful
for word recognition despite fitting the training objective.

The useful engineering fixes from this run are:

1. the Waxal-only training harness is now isolated from the mixed GhanaNLP path,
2. Modal job state records the single durable call ID,
3. the loader was hardened with fixed `aka_asr` parquet URLs after the dataset
   builder hit a transient Hugging Face 504,
4. Modal scaled down to zero active containers after completion.

The next ASR step should not be another blind continuation run. The course is:

1. build correction capture from real microphone tests,
2. audit failure slices by speaker, duration, punctuation loss, code-switching,
   and repeated-token failures,
3. create a reviewed correction dataset,
4. train against that correction dataset while replaying Waxal for regression
   protection,
5. keep punctuation and expressive marks as a separate restoration task unless
   the corpus labels can support them directly.

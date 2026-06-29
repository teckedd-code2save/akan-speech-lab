# ASR v0.3 Result: Not Promoted

Artifact: `serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-broad-lowlr-freezeenc-fullft-v0.3`

Hub model: [teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-broad-lowlr-freezeenc-fullft-v0.3](https://huggingface.co/teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-broad-lowlr-freezeenc-fullft-v0.3)

Modal call: `fc-01KW8BJ3F3KFXWJKRXZA232NSY`

Status: complete. Published for traceability only. This is not a promoted model.

## Configuration

- Base model: `teckedd/whisper-small-waxal-round2-specaug-v1`
- Train rows: 12,000, balanced as 6,000 GhanaNLP + 6,000 Waxal
- Dev rows: 512, balanced as 256 GhanaNLP + 256 Waxal
- Method: full fine-tune with frozen encoder
- Learning rate: `3e-6`
- Effective batch size: 16
- Max steps: 800
- SpecAugment: enabled
- Early regression stop: enabled
- Best checkpoint: `checkpoint-200`
- Stopped after checkpoint 400 showed regression

## Metrics

| Slice | Baseline WER | Final/best WER | Decision |
| --- | ---: | ---: | --- |
| Mixed dev | 45.44% | 45.18% | Tiny gain; not meaningful enough |
| Waxal dev | 34.54% | 35.84% | Regression |
| GhanaNLP dev | 105.85% | 96.92% | Better, but still unusable |

Training loss was `0.9432`; final eval loss was `1.1400`.

## What This Means

This run did not create a better Akan ASR model. The broader low-learning-rate
recipe reduced GhanaNLP damage a little, but it hurt the Waxal slice that the
current best model is meant to protect. The mixed WER movement is too small to
justify promotion.

The result argues against more blind raw Waxal+GhanaNLP full fine-tunes. The
next ASR work should happen before GPU training:

1. audit GhanaNLP alignment, dialect coverage, punctuation, and transcript
   normalization row by row,
2. collect correction pairs from real microphone tests,
3. separate punctuation and expressive-tag restoration from core word ASR,
4. train only after per-corpus manifests and review slices are clean,
5. require that any future candidate protects the Waxal slice while improving
   GhanaNLP or correction-data slices.

Round 2 remains the best published model for Waxal-style Akan word recognition.

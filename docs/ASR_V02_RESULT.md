# ASR v0.2 Result: Not Promoted

Artifact: `serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-balanced-fullft-v0.2`

Hub model: `teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-balanced-fullft-v0.2`

Modal call: `fc-01KW5PRVHMDCKZQ7BBTGRZPKKJ`

## Training Setup

- Base model: `teckedd/whisper-small-waxal-round2-specaug-v1`
- Method: full fine-tuning
- Train rows: 4,000, balanced as 2,000 Waxal + 2,000 GhanaNLP
- Dev rows: 256, balanced as 128 Waxal + 128 GhanaNLP
- Steps: 1,200
- Effective batch size: 16
- Learning rate: `1e-5`
- Warmup steps: 100
- SpecAugment: enabled
- GPU: NVIDIA L4

## Outcome

This run is a failed candidate. It is published for traceability, but it must
not be treated as a better ASR model.

| Metric | Value |
|---|---:|
| Baseline dev WER before training | 44.32% |
| Best loaded checkpoint dev WER | 52.46% |
| Training loss | 0.4703 |
| Best checkpoint | `checkpoint-200` |

Later evaluation checkpoints degraded further:

| Epoch | Eval WER |
|---:|---:|
| 2.4 | 68.87% |
| 3.2 | 77.05% |
| 4.0 | 71.02% |
| 4.8 | 73.57% |

## What This Teaches

The pipeline worked mechanically: it avoided the earlier GhanaNLP unlabeled
split waste, trained, saved, published, and scaled Modal back to zero. The
model result is poor because the training recipe was wrong.

The main failure is likely small-set overfitting. With 4,000 training rows and
an effective batch size of 16, 1,200 steps means about 4.8 epochs over a mixed
subset. The loss collapsed, but held-out WER worsened, which is the pattern we
should expect when the model memorizes the subset instead of improving
generalization.

The next run must not repeat this shape.

## Next Run Requirements

The next candidate must:

- train on a much broader slice of the accepted manifest, not a tiny balanced
  subset;
- run fewer effective epochs over the selected data;
- report WER by corpus before promotion;
- keep the immutable Waxal test separate from tuning;
- stop early if dev WER is worse than the starting checkpoint after the first
  evaluation;
- publish as a candidate only with a model card that says whether it improved
  or failed.


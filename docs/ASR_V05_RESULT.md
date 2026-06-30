# ASR v0.5 Result: GhanaNLP-Only Diagnostic

Artifact: `serendepify-gsl-asr-ak-gnlp-whisper-small-only-lowlr-freezeenc-fullft-v0.5`

Hub model: [teckedd/serendepify-gsl-asr-ak-gnlp-whisper-small-only-lowlr-freezeenc-fullft-v0.5](https://huggingface.co/teckedd/serendepify-gsl-asr-ak-gnlp-whisper-small-only-lowlr-freezeenc-fullft-v0.5)

Modal call: `fc-01KWBVNN2YE4WB6AVKRZ13BDH3`

Status: complete. Published for traceability only. This is not a promoted model.

## Configuration

- Base model: `teckedd/whisper-small-waxal-round2-specaug-v1`
- Train rows: 10,000 GhanaNLP rows
- Dev rows: 602 GhanaNLP validation rows
- Regression eval: 512 Waxal dev rows
- Method: full fine-tune with frozen encoder
- Learning rate: `3e-6`
- Effective batch size: 16
- Max steps: 800
- SpecAugment: enabled
- Best checkpoint: `checkpoint-800`
- Regression stop: not triggered

## Metrics

| Slice | Base WER | v0.5 WER | Decision |
| --- | ---: | ---: | --- |
| GhanaNLP validation | 113.97% | 87.55% | Large improvement, still poor |
| Waxal regression | 32.59% | 35.75% | Regression |

Training loss was `1.7474`. Final GhanaNLP eval loss was `1.3759`.

## What This Means

GhanaNLP is learnable: the run cut GhanaNLP WER by roughly 26 absolute points.
That disproves the idea that the dataset is entirely unusable. However, 87.55%
WER is still far from a useful ASR model, and the same checkpoint damages the
Waxal regression slice.

The most likely issue is not simply model capacity. The next step is data work:

1. inspect high-loss and high-WER GhanaNLP rows,
2. find transcript conventions that conflict with Waxal normalization,
3. check audio/text alignment failures,
4. group by duration and repeated transcript patterns,
5. create a cleaned GhanaNLP subset before another GPU run,
6. use Waxal replay only after GhanaNLP row quality is understood.

This result supports a correction-data strategy. GhanaNLP can contribute useful
signal, but it should not be mixed blindly into the main model.

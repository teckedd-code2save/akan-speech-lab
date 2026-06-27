---
language:
- tw
- ak
license: apache-2.0
tags:
- automatic-speech-recognition
- whisper
- akan
- twi
- ghanaian-speech-lab
- serendepify-gsl
pipeline_tag: automatic-speech-recognition
base_model: teckedd/whisper-small-waxal-round2-specaug-v1
library_name: transformers
---

# serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1

Status: **planned review artifact**. This model card is the publication contract
for the first Ghanaian Speech Lab ASR v0.1 pass. It must not be pushed as a
trained model until the required evidence below exists.

## Code Name

```text
serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1
```

Expected Hub repository:

```text
teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1
```

Meaning:

- `serendepify-gsl`: Ghanaian Speech Lab
- `asr`: automatic speech recognition
- `ak`: Akan-family target
- `waxal-gnlp`: Waxal plus GhanaNLP
- `whisper-small`: base family
- `replay-fullft`: replay-mixed full fine-tuning
- `v0.1`: first external review candidate

## Intended Use

Experimental Akan/Twi/Fante ASR research for Ghanaian speech applications. The
target use cases are health, ecommerce, customer support, local agents, and
speech-data tooling.

## Training Plan

- Base model: `openai/whisper-small`
- Starting checkpoint: `teckedd/whisper-small-waxal-round2-specaug-v1`
- Tuning method: `full_fine_tune`
- Dataset tokens: `waxal`, `gnlp`
- Replay strategy: Waxal remains the anchor/regression corpus; GhanaNLP is added
  only after corpus harmonization.

## Data Requirements Before Training

Training cannot start until each source has:

- source license and consent posture
- stable manifest row IDs
- audio hashes and text hashes
- raw transcript
- punctuated transcript when available
- WER-normalized transcript
- optional expressive tags
- duration, sample rate, silence, and clipping flags
- duplicate audio and duplicate text groups
- split-leakage report
- known orthography and spelling conventions

## Evaluation Requirements Before Publication

Publication requires:

- WER/CER by corpus
- WER/CER by duration bucket
- speaker breakdown where speaker IDs exist
- repetition-collapse count
- punctuation precision/recall/F1 when punctuation is enabled
- qualitative Ghanaian review notes
- failure taxonomy with concrete examples
- paired comparison against the previous artifacts on matched rows

## Current Metrics

No metrics yet. This is a planned artifact, not a trained checkpoint.

## Known Risks To Test

- GhanaNLP harmonization may help Twi but regress Waxal unless replay is mixed
  correctly.
- Whisper tokenizer fragmentation may still limit Akan orthography.
- Punctuation must be evaluated separately from WER.
- Expressive tags require manual labels before they can be trusted.

## Promotion Status

Not promoted. Not trained. Not published as a model yet.

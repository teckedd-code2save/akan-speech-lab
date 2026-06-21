---
language:
- ak
- tw
license: cc-by-sa-4.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
base_model: openai/whisper-small
datasets:
- google/WaxalNLP
tags:
- whisper
- akan
- twi
- asr
- experimental
model-index:
- name: Whisper Small Waxal Akan Round 2
  results:
  - task:
      type: automatic-speech-recognition
    dataset:
      name: WaxalNLP aka_asr immutable test_v1
      type: google/WaxalNLP
      config: aka_asr
      split: test
    metrics:
    - name: WER
      type: wer
      value: 32.8398
    - name: CER
      type: cer
      value: 11.7885
---

# Whisper Small Waxal Akan Round 2

Experimental Akan/Twi speech-recognition checkpoint selected at training step 1,200. This model is published for evaluation and community testing; it did not pass every production-promotion gate.

## Results

All models below were decoded in one fixed job on all 1,522 rows of `google/WaxalNLP`, configuration `aka_asr`, published test split. Decoding used beam size 1, task `transcribe`, maximum generation length 225, and no forced language token.

| Model | WER | CER |
|---|---:|---:|
| Original Waxal fine-tune | 34.32% | 12.23% |
| Published continuation v1 | 33.66% | 12.10% |
| **Round 2 checkpoint 1,200** | **32.84%** | **11.79%** |

Against `teckedd/whisper-small-waxal-akan-continuation-v1`, the paired 5,000-sample bootstrap candidate-minus-baseline WER interval was -1.39 to -0.06 percentage points, with 98.36% probability of improvement. Round 2 produced 677 better rows, 316 ties, and 529 worse rows.

## Training Data and Split

Only `google/WaxalNLP/aka_asr` was used. The dataset revisions, row IDs, speakers, normalized transcripts, and decoded-audio hashes were frozen before training.

- train: 9,133 rows / 83 speakers
- development: 2,091 rows / 18 speakers
- immutable test: 1,522 rows / 33 speakers
- zero speaker, normalized-transcript, sample-ID, or decoded-audio overlap between partitions
- one train clip outside the 0.4–30 second window removed
- three within-train duplicate clips and one within-development duplicate removed
- one train clip duplicated in development quarantined from train

Audio was decoded as mono float32 at 16 kHz. Text was normalized with Unicode NFC, lowercasing, whitespace collapse, and punctuation removal while preserving Akan letters such as `ɛ` and `ɔ`. Training used Whisper log-mel features and training-only SpecAugment.

## Training Configuration

- base: `openai/whisper-small`
- full-model fine-tuning, FP16
- effective batch size: 32
- learning rate: `1e-5`
- warmup: 200 steps
- maximum: 2,000 steps
- evaluation/save interval: 200 steps
- selected by unseen-speaker development WER
- selected checkpoint: step 1,200, development WER 32.14%
- training tokenizer prefix: Yoruba proxy, task `transcribe`
- evaluation: no forced language token

## Usage

```python
import torch
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="teckedd/whisper-small-waxal-round2-specaug-v1",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
asr.model.config.forced_decoder_ids = None
asr.model.generation_config.forced_decoder_ids = None
asr.model.generation_config.language = None
asr.model.generation_config.task = "transcribe"

print(asr("audio.wav", generate_kwargs={"task": "transcribe"})["text"])
```

## Limitations

- One immutable-test output entered a severe repetition loop, repeating `ayɛ` 62 times.
- Test speaker `4430` regressed from 43.24% to 49.69% WER relative to the continuation checkpoint.
- The model missed the preregistered target of WER below 30.86%.
- Akan spelling, dialect variation, code-switching, names, and noisy recordings remain difficult.
- Do not use this model as the sole basis for medical, emergency, legal, or financial decisions.

The full experiment specification, preprocessing audit, paired evaluation, and failed promotion decision are documented in the [Akan Speech Lab](https://github.com/teckedd-code2save/akan-speech-lab).

## License

Released under CC BY-SA 4.0 as a conservative choice reflecting the licenses listed for WaxalNLP. Users must review and satisfy the source dataset's attribution and share-alike requirements.

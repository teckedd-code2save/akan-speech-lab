from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AsrArtifactSpec:
    code_name: str
    hf_repo: str
    task: str
    language: str
    datasets: tuple[str, ...]
    base_model: str
    starting_checkpoint: str
    method: str
    tuning_type: str
    status: str = "planned"

    @property
    def dataset_slug(self) -> str:
        return "-".join(self.datasets)

    def to_json(self) -> dict:
        return {
            "code_name": self.code_name,
            "hf_repo": self.hf_repo,
            "task": self.task,
            "language": self.language,
            "datasets": list(self.datasets),
            "base_model": self.base_model,
            "starting_checkpoint": self.starting_checkpoint,
            "method": self.method,
            "tuning_type": self.tuning_type,
            "status": self.status,
        }


FIRST_ASR_REVIEW = AsrArtifactSpec(
    code_name="serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1",
    hf_repo="teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1",
    task="asr",
    language="ak",
    datasets=("waxal", "gnlp"),
    base_model="openai/whisper-small",
    starting_checkpoint="teckedd/whisper-small-waxal-round2-specaug-v1",
    method="replay-fullft",
    tuning_type="full_fine_tune",
)


def render_review_model_card(spec: AsrArtifactSpec = FIRST_ASR_REVIEW) -> str:
    dataset_list = ", ".join(f"`{dataset}`" for dataset in spec.datasets)
    return f"""---
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
base_model: {spec.starting_checkpoint}
library_name: transformers
---

# {spec.code_name}

Status: **planned review artifact**. This model card is the publication contract
for the first Ghanaian Speech Lab ASR v0.1 pass. It must not be pushed as a
trained model until the required evidence below exists.

## Code Name

```text
{spec.code_name}
```

Expected Hub repository:

```text
{spec.hf_repo}
```

Meaning:

- `serendepify-gsl`: Ghanaian Speech Lab
- `{spec.task}`: automatic speech recognition
- `{spec.language}`: Akan-family target
- `{spec.dataset_slug}`: Waxal plus GhanaNLP
- `whisper-small`: base family
- `{spec.method}`: replay-mixed full fine-tuning
- `v0.1`: first external review candidate

## Intended Use

Experimental Akan/Twi/Fante ASR research for Ghanaian speech applications. The
target use cases are health, ecommerce, customer support, local agents, and
speech-data tooling.

## Training Plan

- Base model: `{spec.base_model}`
- Starting checkpoint: `{spec.starting_checkpoint}`
- Tuning method: `{spec.tuning_type}`
- Dataset tokens: {dataset_list}
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
"""


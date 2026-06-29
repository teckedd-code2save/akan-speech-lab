# Akan Speech Lab

Reusable fine-tuning and evaluation infrastructure for Akan-language speech systems. The active ASR target is an expressive Akan recognizer: stronger word transcription plus punctuation, pause, emphasis, and stretch capture where the data supports it.

The goal is commercial-grade speech infrastructure for domains such as health, ecommerce, customer support, family care, and local voice agents. This repo is intentionally separate from the hackathon app so experiments, datasets, checkpoints, and Modal jobs do not bloat product code.

## Milestones

1. Audit Waxal Akan ASR and freeze a reproducible, speaker-balanced benchmark.
2. Normalize Akan text consistently and inspect data quality before training.
3. Evaluate existing baselines:
   - `teckedd/whisper_small-waxal_akan-asr-v1`
   - MMS ASR variants
   - any strong public Whisper/Akan checkpoint we validate
4. Publish the experimental Round 2 model with its failed-promotion limitations.
5. Build ASR Milestone 1: correction capture, punctuation restoration, and expressive prosody tags.
6. Build commercially usable Asante Twi TTS after ASR Milestone 1 has a clean feedback loop.
7. Run Modal only for durable gated jobs; one container maximum and scale-to-zero.

## Repository Layout

```text
configs/        Training and dataset configs.
data/           Local raw/processed data and manifests; gitignored.
docs/           Pipeline notes, dataset cards, Modal usage, eval protocol.
modal_jobs/     Modal GPU entrypoints.
src/            Reusable Python package.
scripts/        CLI scripts for prep, smoke tests, and publishing.
evals/          Small local samples and generated reports.
tests/          Unit tests for normalization and manifest logic.
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/smoke_test.py
pytest
```

## Local UI

Run the Gradio control console:

```bash
python app.py
```

Open `http://127.0.0.1:7862`.

The UI can:

- build a fixed Waxal Akan sample pack,
- download the sample audio locally,
- dry-run model evaluation,
- cache a Hugging Face ASR model,
- run a small ASR eval,
- transcribe microphone/upload audio with three published checkpoints,
- render WER/CER reports,
- operate the ordered TTS corpus, smoke, overfit, pilot, and full-run gates.

## Cost Rule

Default workflow is local and CPU-safe. Modal GPU jobs should only run after:

- the dataset manifest validates,
- a small eval sample is ready,
- the target baseline is recorded,
- the expected output path and Hugging Face repo are known.

## Current ASR Result

The immediate baseline is `teckedd/whisper_small-waxal_akan-asr-v1`. On this repo's frozen 99-row, 33-speaker Waxal benchmark it scores **33.62% WER / 12.37% CER** with a 30.91%-36.79% bootstrap WER interval. No forced language and the checkpoint's stored Yoruba prompt produce identical predictions; English prompting is invalid for this model. The first candidate therefore uses no forced language token.

The first evidence-driven continuation run improved the complete 1,123-row validation split from **32.69% to 31.45% WER** at step 300. Step 400 regressed to 31.83%. On all 1,522 held-out Waxal test rows, the selected checkpoint improves **33.84% to 32.77% WER** and **12.74% to 12.47% CER**. A 5,000-sample paired bootstrap gives 99.86% probability of improvement and a -1.90 to -0.33 point 95% WER-difference interval. The [experimental candidate](https://huggingface.co/teckedd/whisper-small-waxal-akan-continuation-v1) remains gated because two outputs entered severe repetition loops and Ghanaian listening review is pending.

Round 2 reached **32.84% WER / 11.79% CER** on the immutable 1,522-row Waxal test, compared with 34.32% for the original and 33.66% for the continuation under the same harness. It is published as [teckedd/whisper-small-waxal-round2-specaug-v1](https://huggingface.co/teckedd/whisper-small-waxal-round2-specaug-v1), but is explicitly experimental because one repetition collapse and the speaker-4430 regression failed the promotion gate. The next ASR milestone is not another blind run: it adds correction capture, punctuation restoration, and expressive tags while keeping WER evaluation separate.

The first broader Waxal+GhanaNLP training attempt, v0.2, is published for traceability at [teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-balanced-fullft-v0.2](https://huggingface.co/teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-balanced-fullft-v0.2), but it is **not promoted**. It trained successfully, but worsened dev WER from **44.32% to 52.46%**, likely due to small-set overfitting across 4.8 effective epochs. See [ASR v0.2 Result](docs/ASR_V02_RESULT.md).

The v0.3 broader low-learning-rate frozen-encoder retry is published for traceability at [teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-broad-lowlr-freezeenc-fullft-v0.3](https://huggingface.co/teckedd/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-broad-lowlr-freezeenc-fullft-v0.3), but it is **not promoted**. It barely moved mixed dev WER from **45.44% to 45.18%**, regressed Waxal from **34.54% to 35.84%**, and left GhanaNLP unusable at **96.92% WER**. See [ASR v0.3 Result](docs/ASR_V03_RESULT.md).

## Key Docs

- [Roadmap and handoff](docs/ROADMAP.md)
- [Artifact versioning](docs/ARTIFACT_VERSIONING.md)
- [ASR pipeline loop](docs/ASR_PIPELINE_LOOP.md)
- [ASR Milestone 1: Expressive Akan Recognition](docs/ASR_MILESTONE1_EXPRESSIVE.md)
- [ASR research spine](docs/ASR_RESEARCH_SPINE.md)
- [ASR v0.2 result](docs/ASR_V02_RESULT.md)
- [ASR v0.3 result](docs/ASR_V03_RESULT.md)
- [Asante Twi TTS research and execution record](docs/TTS_RESEARCH_AND_EXECUTION.md)
- [Contamination-safe ASR Round 2 specification](docs/ASR_ROUND2_SPEC.md)
- [Dataset plan](docs/datasets.md)
- [Preprocessing strategy](docs/preprocessing.md)
- [Experiment plan](docs/experiments.md)
- [Evaluation protocol](docs/evaluation.md)
- [Frozen benchmark](docs/benchmark.md)
- [Modal usage](docs/modal.md)
- [Old Waxal notebook findings](docs/old_waxal_notebook_findings.md)

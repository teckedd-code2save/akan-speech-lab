# Akan Speech Lab

Reusable fine-tuning and evaluation infrastructure for Akan-language speech systems, starting with Twi/Fante ASR and later TTS.

The goal is commercial-grade speech infrastructure for domains such as health, ecommerce, customer support, family care, and local voice agents. This repo is intentionally separate from the hackathon app so experiments, datasets, checkpoints, and Modal jobs do not bloat product code.

## Initial Roadmap

1. Audit Waxal Akan ASR and freeze a reproducible, speaker-balanced benchmark.
2. Normalize Akan text consistently and inspect data quality before training.
3. Evaluate existing baselines:
   - `teckedd/whisper_small-waxal_akan-asr-v1`
   - MMS ASR variants
   - any strong public Whisper/Akan checkpoint we validate
4. Fine-tune ASR with strict eval gates before pushing to Hugging Face.
5. Add TTS data prep and evaluation after ASR is stable.
6. Run Modal only for training/eval jobs that need GPU, then shut jobs down.

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
- render WER/CER reports.

## Cost Rule

Default workflow is local and CPU-safe. Modal GPU jobs should only run after:

- the dataset manifest validates,
- a small eval sample is ready,
- the target baseline is recorded,
- the expected output path and Hugging Face repo are known.

## Current ASR Baseline

The immediate baseline is `teckedd/whisper_small-waxal_akan-asr-v1`. On this repo's frozen 99-row, 33-speaker Waxal benchmark it scores **33.62% WER / 12.37% CER** with a 30.91%-36.79% bootstrap WER interval. No forced language and the checkpoint's stored Yoruba prompt produce identical predictions; English prompting is invalid for this model. The first candidate therefore uses no forced language token.

The first evidence-driven continuation run improved the complete 1,123-row validation split from **32.69% to 31.45% WER** at step 300. Step 400 regressed to 31.83%. On all 1,522 held-out Waxal test rows, the selected checkpoint improves **33.84% to 32.77% WER** and **12.74% to 12.47% CER**. A 5,000-sample paired bootstrap gives 99.86% probability of improvement and a -1.90 to -0.33 point 95% WER-difference interval. The [experimental candidate](https://huggingface.co/teckedd/whisper-small-waxal-akan-continuation-v1) remains gated because two outputs entered severe repetition loops and Ghanaian listening review is pending.

The first GhanaNLP-only adaptation reduced held-out GhanaNLP WER from **160.65% to 99.35%**, but regressed the complete Waxal test from **32.77% to 37.80%**. It is retained as a failed-promotion experiment, not published as a model. The next ASR round must use Waxal replay or an adapter to control forgetting.

## Key Docs

- [Roadmap and handoff](docs/ROADMAP.md)
- [Dataset plan](docs/datasets.md)
- [Preprocessing strategy](docs/preprocessing.md)
- [Experiment plan](docs/experiments.md)
- [Evaluation protocol](docs/evaluation.md)
- [Frozen benchmark](docs/benchmark.md)
- [Modal usage](docs/modal.md)
- [Old Waxal notebook findings](docs/old_waxal_notebook_findings.md)

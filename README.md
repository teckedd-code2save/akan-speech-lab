# Akan Speech Lab

Reusable fine-tuning and evaluation infrastructure for Akan-language speech systems, starting with Twi/Fante ASR and later TTS.

The goal is commercial-grade speech infrastructure for domains such as health, ecommerce, customer support, family care, and local voice agents. This repo is intentionally separate from the hackathon app so experiments, datasets, checkpoints, and Modal jobs do not bloat product code.

## Initial Roadmap

1. Build clean dataset manifests for Waxal Akan ASR and GhanaNLP Twi speech-text.
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

## Cost Rule

Default workflow is local and CPU-safe. Modal GPU jobs should only run after:

- the dataset manifest validates,
- a small eval sample is ready,
- the target baseline is recorded,
- the expected output path and Hugging Face repo are known.

## Current ASR Baseline

The immediate baseline is `teckedd/whisper_small-waxal_akan-asr-v1`, trained on Waxal Akan ASR with reported WER `34.2849`. The target is materially better than this through better preprocessing, evaluation, and training discipline.

## Key Docs

- [Dataset plan](docs/datasets.md)
- [Preprocessing strategy](docs/preprocessing.md)
- [Experiment plan](docs/experiments.md)
- [Evaluation protocol](docs/evaluation.md)
- [Modal usage](docs/modal.md)
- [Old Waxal notebook findings](docs/old_waxal_notebook_findings.md)

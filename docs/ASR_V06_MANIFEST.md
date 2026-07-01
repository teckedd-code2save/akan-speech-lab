# ASR v0.6 Manifest Plan

Artifact code:

```text
serendepify-gsl-asr-ak-waxal-gnlpclean-whisper-small-replay-fullft-v0.6
```

Purpose: build the next trainable ASR input from evidence instead of another raw
training attempt.

## Inputs

- Parent manifest:
  `data/manifests/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1.jsonl`
- GhanaNLP clean candidates:
  `outputs/audits/gnlp_manifest_v0.5/clean_candidates.jsonl`

## Command

```bash
.venv/bin/python scripts/build_asr_v06_manifest.py
```

Generated outputs:

```text
data/manifests/serendepify-gsl-asr-ak-waxal-gnlpclean-whisper-small-replay-fullft-v0.6.jsonl
evals/reports/serendepify-gsl-asr-ak-waxal-gnlpclean-whisper-small-replay-fullft-v0.6-manifest.json
```

These are generated artifacts and are gitignored.

## Policy

- Keep Waxal as the replay anchor and regression evidence.
- Use only GhanaNLP rows that passed the v0.5 audit clean-candidate gate.
- Exclude GhanaNLP test rows from the v0.6 manifest.
- Mark every row with `row_role`:
  - `train`
  - `validation`
  - `regression_test`
- Mark every row with `training_bucket` so the training code cannot silently
  mix validation or regression-test rows into training.

## Expected Next Training Recipe

Generated manifest result:

| Bucket | Rows |
|---|---:|
| Waxal replay train | 9,138 |
| Clean GhanaNLP adaptation train | 6,033 |
| Waxal replay validation | 2,092 |
| Clean GhanaNLP adaptation validation | 326 |
| Waxal regression test | 1,522 |
| Excluded GhanaNLP non-clean candidates | 5,126 |
| Reserved GhanaNLP test rows | 345 |

Total trainable rows: 15,171. Total validation rows: 2,418.

Use train rows from:

- `waxal_replay_train`
- `gnlp_clean_adaptation_train`

Evaluate during training on:

- `waxal_replay_validation`
- `gnlp_clean_adaptation_validation`

Never train on:

- `waxal_regression_test`
- any GhanaNLP test row

The next Modal job should stop on Waxal regression or repetition collapse and
should report per-corpus WER separately.

## Modal Executor

Executor:

```text
modal_jobs/asr_v06_clean_replay.py
```

Durable controller:

```text
scripts/modal_asr_v06_clean_replay_jobs.py
```

Cost-safe order:

```bash
.venv/bin/python scripts/modal_asr_v06_clean_replay_jobs.py status
.venv/bin/python scripts/modal_asr_v06_clean_replay_jobs.py deploy
```

Only submit training after confirming the manifest report and Modal deployment:

```bash
.venv/bin/python scripts/modal_asr_v06_clean_replay_jobs.py train
.venv/bin/python scripts/modal_asr_v06_clean_replay_jobs.py status
```

Cancel if needed:

```bash
.venv/bin/python scripts/modal_asr_v06_clean_replay_jobs.py cancel
```

Default training controls:

- Base: `teckedd/whisper-small-waxal-round2-specaug-v1`
- Max steps: 1,200
- Learning rate: `2e-6`
- Effective batch size: 16
- Encoder frozen: yes
- SpecAugment: yes
- Eval every 200 steps
- Early stop on first Waxal validation regression
- Publish target:
  `teckedd/serendepify-gsl-asr-ak-waxal-gnlpclean-whisper-small-replay-fullft-v0.6`

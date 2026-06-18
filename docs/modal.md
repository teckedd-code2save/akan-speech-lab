# Modal Usage

Modal is for GPU-bound training and evaluation only.

## Rules

- Validate manifests locally before running GPU jobs.
- Use `scaledown_window=60` or similarly short idle windows.
- Prefer one-shot jobs over always-on services.
- Log model ID, dataset manifest, sample count, runtime, and cost notes.
- Stop or let jobs scale down after each run.

## Volumes

- `akan-speech-hf-cache`: reusable model and dataset cache
- `akan-speech-checkpoints`: smoke summaries, checkpoints, and final model
- `akan-speech-eval-results`: durable benchmark reports

No ASR training web service is deployed. Each job is one-shot and the L4 scales to zero when the command exits.

## Commands

```bash
modal run modal_jobs/asr_train.py --smoke
```

The smoke run uses 32 train rows, 16 validation rows, and two steps. It must write `summary.json` before a full run is allowed.

```bash
modal run modal_jobs/asr_train.py --max-steps 1200
```

At the June 2026 L4 list price of about $0.80/hour, a conservative three-hour full-run estimate is $2.40. Dataset preparation must run on CPU first so download time does not consume L4 minutes.

Retrieve a result:

```bash
modal volume get akan-speech-checkpoints \
  whisper-small-waxal-aka-no-language-v2/summary.json \
  outputs/modal/summary.json
```

Evaluate a persisted candidate directly, without publishing it to the Hub:

```bash
modal run modal_jobs/asr_eval.py \
  --model-id /checkpoints/<run-name>/checkpoint-<step> \
  --strategies yoruba,no_forced_language \
  --output evals/reports/<run-name>-checkpoint-<step>.json
```

After a run, verify that `modal app list` shows no active `akan-speech-asr-train` task. Volumes persist; GPU containers do not.

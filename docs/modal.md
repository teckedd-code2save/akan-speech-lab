# Modal Usage

Modal is used for CPU audio preparation and GPU-bound training/evaluation.

## Round 2 Durable Jobs

Round 2 does not use an attached `modal run` process. The pipeline follows Modal's deployed job-processing pattern:

1. Deploy `akan-speech-asr-round2` once.
2. Resolve deployed functions with `modal.Function.from_name`.
3. Submit with `.spawn()` and persist the returned `fc-...` call ID.
4. Poll later with `modal.FunctionCall.from_id(call_id).get(timeout=0)`.
5. Reject duplicate submissions while an existing call is submitted, running, or complete.

Deployed functions scale to zero by default, so deployment does not keep paid containers warm. Job state is stored in `outputs/modal_jobs/asr_round2.json`; datasets and checkpoints remain in Modal Volumes.

```bash
python scripts/modal_round2_jobs.py deploy
python scripts/modal_round2_jobs.py prepare
python scripts/modal_round2_jobs.py status
python scripts/modal_round2_jobs.py train --mode smoke
python scripts/modal_round2_jobs.py train --mode pilot
python scripts/modal_round2_jobs.py cancel train_pilot
```

Preparation and training have bounded retries. Training automatically resumes from the newest persisted checkpoint and returns an existing completed summary instead of repeating work.

## Rules

- Validate manifests locally before running GPU jobs.
- Use `scaledown_window=30` for GPU jobs.
- Prefer one-shot jobs over always-on services.
- Log model ID, dataset manifest, sample count, runtime, and cost notes.
- Stop or let jobs scale down after each run.

## Volumes

- `akan-speech-hf-cache`: reusable model and dataset cache
- `akan-speech-checkpoints`: smoke summaries, checkpoints, and final model
- `akan-speech-eval-results`: durable benchmark reports

The Round 2 app is deployed as a persistent function namespace, not an always-on web server. CPU and GPU containers still scale to zero when no call is active.

## Legacy Commands (Do Not Use For Round 2)

The commands below document older experiments only. They create attached or ephemeral app runs and are intentionally excluded from the Round 2 execution path.

```bash
modal run modal_jobs/asr_train.py --smoke
```

The smoke run uses 32 train rows, 16 validation rows, and two steps. It must write `summary.json` before a full run is allowed.

Run the isolated GhanaNLP wiring smoke and full continuation:

```bash
modal run modal_jobs/asr_train.py --smoke --arm ghana_nlp_only
modal run modal_jobs/asr_train.py --arm ghana_nlp_only
```

The full arm first prepares and persists the dataset on CPU, then allocates an L4. Its held-out test split is saved but never passed to the trainer.

```bash
modal run modal_jobs/asr_train.py --max-steps 1200
```

Resume a disconnected run from a durable checkpoint. Use detached mode so a local network
drop cannot terminate the remote job:

```bash
modal run --detach modal_jobs/asr_train.py \
  --arm continue_yoruba \
  --resume-step 100
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

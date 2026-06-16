# Modal Usage

Modal is for GPU-bound training and evaluation only.

## Rules

- Validate manifests locally before running GPU jobs.
- Use `scaledown_window=60` or similarly short idle windows.
- Prefer one-shot jobs over always-on services.
- Log model ID, dataset manifest, sample count, runtime, and cost notes.
- Stop or let jobs scale down after each run.

## First Commands

```bash
modal run modal_jobs/asr_eval.py::eval_asr --model-id teckedd/whisper_small-waxal_akan-asr-v1 --manifest-path data/manifests/ghana_nlp_twi_eval.jsonl
```

Training command will be enabled after the local manifest and eval baseline are real.


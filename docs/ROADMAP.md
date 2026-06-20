# Akan Speech Lab Roadmap and Handoff

Last updated: 2026-06-20

## Objective

Build commercially usable Akan speech infrastructure for health, ecommerce, support, and voice-agent products. The lab must improve ASR and TTS through reproducible data preparation, frozen evaluation sets, controlled training experiments, Ghanaian listening review, and explicit licensing.

The current work is **ASR round 2**. TTS training has not started.

## Current Position

### Completed

1. Built a local Gradio operator UI at `http://127.0.0.1:7862`.
2. Reproduced and evaluated `teckedd/whisper_small-waxal_akan-asr-v1`.
3. Trained and published `teckedd/whisper-small-waxal-akan-continuation-v1`.
4. Evaluated both Waxal models on all 1,522 held-out test rows.
5. Added per-row WER/CER, audio inspection, paired comparison, and repetition-collapse detection.
6. Audited all GhanaNLP Dataset Viewer metadata without downloading audio locally.
7. Created deterministic normalized-transcript groups for a 90/5/5 GhanaNLP split.
8. Passed a two-step GhanaNLP Modal smoke test.
9. Completed a 400-step GhanaNLP-only continuation from the published Waxal candidate.

### Evidence

| Experiment | Baseline | Candidate | Interpretation |
|---|---:|---:|---|
| Waxal full test, 1,522 rows | 33.84% WER | 32.77% WER | Small statistically supported gain |
| GhanaNLP validation, 602 rows | 165.53% WER | 84.58% WER | Large domain-adaptation gain; absolute quality still poor |
| GhanaNLP test, 571 rows | 160.65% WER | 99.35% WER | Statistically clear gain; still unusable absolutely |
| Waxal regression test, 1,522 rows | 32.77% WER | 37.80% WER | Failed promotion: catastrophic forgetting |

GhanaNLP checkpoint selection:

| Step | Validation WER |
|---:|---:|
| 100 | 103.88% |
| 200 | 88.11% |
| 300 | 88.99% |
| 400 | **84.58%** |

Selected checkpoint: `checkpoint-400`.

### Round 1 Status

**Complete. GhanaNLP checkpoint not promoted.** All Akan Speech Lab Modal jobs are stopped. The checkpoint remains in the Modal volume as research evidence and is not published to the Hub.

Completed GhanaNLP test analysis: 341 rows improved, 163 tied, and 67 worsened. The paired 95% candidate-minus-baseline WER interval is -92.34 to -26.53 points, with 99.9% bootstrap probability of improvement. Repetition-collapse detections fell from 17 to 1, but 11 rows became new >=100% WER regressions.

Completed Waxal regression analysis: 407 rows improved, 226 tied, and 889 worsened. Waxal WER rose from 32.77% to 37.80%; the paired 95% WER-change interval is +3.67 to +7.10 points. Two repetition collapses remain. This is measurable catastrophic forgetting.

## Dataset Findings

### Waxal `aka_asr`

- Provides train, validation, and test splits with speaker IDs.
- Frozen full test: 1,522 rows.
- Current best published lab model: `teckedd/whisper-small-waxal-akan-continuation-v1`.

### GhanaNLP Twi

- Dataset: `ghananlpcommunity/twi-speech-text-multispeaker-16k`.
- Viewer rows: 15,560; card prose says 21,138.
- Usable transcript rows: 15,372 / 9.51 hours.
- Existing 0.4-30 second filter retains 11,830 rows and removes 3,542 very short clips.
- Published schema has no speaker ID, so speaker-safe evaluation cannot be claimed.
- 1,057 duplicate transcript groups are kept within one partition.
- Prepared partitions after duration filtering: 10,657 train / 602 validation / 571 test.

## Decision Gate for ASR Round 1

Do not publish or promote the GhanaNLP candidate until all are true:

- GhanaNLP held-out test improves materially over the Waxal continuation.
- Waxal full-test regression is measured and acceptable.
- Repetition-collapse count is zero or a guarded retry/fallback is validated.
- Ghanaian listening review confirms useful spelling and pronunciation behavior.

If GhanaNLP improves its own test but damages Waxal heavily, use mixed replay or adapter-based training next. Do not continue blind single-corpus training.

## Next ASR Round

Canonical specification: [ASR Round 2](ASR_ROUND2_SPEC.md).

Implementation status:

- pinned Waxal metadata revision and speaker-component split: complete
- metadata contamination assertions: passed
- deployed scale-to-zero Modal runner: complete
- decoded-audio quality and contamination audit: passed
- durable prepared dataset: 9,133 train / 2,091 dev / 1,522 test
- two-step L4 smoke with SpecAugment, eval, checkpoint, and resume artifacts: passed
- 200-step pilot: ready, not submitted

Round 2 audit details:

- speaker overlap: zero across train, dev, and test
- normalized-transcript overlap: zero across train, dev, and test
- decoded-audio overlap: zero after quarantining one train copy duplicated in dev
- removed before training: one out-of-range train clip, three duplicate train clips, and one duplicate dev clip
- preparation call: `fc-01KVJW2D1MC677XZ1GTV957PDY`
- smoke call: `fc-01KVJW5QZMZE0ZYFZPMJ1PBYH7`

1. Run the 200-step Waxal-only pilot and inspect unseen-speaker dev WER plus predictions.
2. Continue to the full 2,000-step run only if the pilot is stable and improves the frozen dev baseline.
3. Evaluate the selected checkpoint once on the immutable 1,522-row Waxal test.
4. Build error buckets: duration, dialect markers, English/code-switching, names, and repetition.
5. Promote only if test WER beats 30.86% and Ghanaian listening review confirms the gain.

## TTS Roadmap

TTS starts after ASR round 1 is closed:

1. Audit consent, license, speaker structure, transcript quality, sample rate, clipping, and silence.
2. Build a fixed Akan prompt set covering health, ecommerce, names, numbers, questions, and code-switching.
3. Establish objective audio checks and a Ghanaian MOS/intelligibility review form.
4. Train a single-speaker or explicitly conditioned multi-speaker baseline only when speaker identity is reliable.
5. Keep `facebook/mms-tts-aka` as a non-commercial diagnostic baseline because of its CC-BY-NC license.
6. Publish a commercial candidate only from data and model licenses that permit commercial use.

## Cost and Operations

- Modal dataset preparation runs on CPU before L4 allocation.
- Smoke tests use 2 steps and isolated output directories.
- Pilot/full checkpoints persist to `akan-speech-checkpoints`; the newest checkpoint resumes automatically.
- Round 2 is a named deployment submitted with `.spawn()` and persisted call IDs, not an attached `modal run` process.
- GPU functions use `scaledown_window=30`, `max_containers=1`, and bounded retries.
- Before ending a session, run `modal app list` and confirm no Akan Speech Lab task is active.

## Artifact Map

- Gradio UI: `app.py`
- Round 2 training: `modal_jobs/asr_round2.py`
- Durable job controller: `scripts/modal_round2_jobs.py`
- Frozen split builder: `scripts/build_waxal_round2_manifest.py`
- Evaluation: `modal_jobs/asr_eval.py`
- GhanaNLP metadata prep: `scripts/prepare_ghana_nlp.py`
- Frozen/evaluation reports: `evals/reports/` (large JSON files are gitignored)
- Downloaded Modal summaries: `outputs/modal/` (gitignored)
- Data policy: `docs/datasets.md`
- Evaluation policy: `docs/evaluation.md`
- Training record: `docs/training.md`

## Resume Commands

Inspect the durable Round 2 state:

```bash
.venv/bin/python scripts/modal_round2_jobs.py status
```

Submit the gated pilot only after reviewing the passed smoke:

```bash
.venv/bin/python scripts/modal_round2_jobs.py train --mode pilot
```

Cancel a submitted pilot without depending on a local terminal heartbeat:

```bash
.venv/bin/python scripts/modal_round2_jobs.py cancel train_pilot
```

Run the local lab:

```bash
.venv/bin/python app.py
```

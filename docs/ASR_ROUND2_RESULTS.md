# ASR Round 2 Results

Date: 2026-06-21

## Decision

**Do not promote or publish the Round 2 checkpoint.** It is a statistically supported improvement over both fixed baselines under the same evaluation harness, but it misses the absolute WER target, retains one repetition collapse, and materially regresses one test speaker.

Selected checkpoint: `whisper-small-waxal-round2-specaug-v1/checkpoint-1200`

Immutable dataset: `google/WaxalNLP/aka_asr/test_v1`, 1,522 rows, 33 test-only speakers.

## Corpus Results

| Model | WER | CER | Repetition collapses |
|---|---:|---:|---:|
| Original Waxal fine-tune | 34.32% | 12.23% | 1 |
| Published continuation v1 | 33.66% | 12.10% | 1 |
| Round 2 checkpoint 1200 | **32.84%** | **11.79%** | 1 |

All three models were decoded once in the same job with no forced language token, beam size 1, maximum generation length 225, and the same normalized references.

## Paired Evidence

Against the original Waxal fine-tune:

- paired candidate-minus-baseline WER interval: -2.09 to -0.71 percentage points
- bootstrap probability that Round 2 improves: 99.96%
- 713 rows better, 334 tied, 475 worse

Against the published continuation:

- paired candidate-minus-baseline WER interval: -1.39 to -0.06 percentage points
- bootstrap probability that Round 2 improves: 98.36%
- 677 rows better, 316 tied, 529 worse

## Breakdown

| Duration | Continuation WER | Round 2 WER |
|---|---:|---:|
| 10-20 seconds | 33.96% | 33.28% |
| Over 20 seconds | 33.26% | 32.26% |

Largest speaker regression: speaker `4430`, 43.24% to 49.69% WER across 45 rows. Four other speaker regressions were between 0.15 and 1.32 points.

Largest speaker gains were 3.30 points for speaker `4708`, 3.24 for `5247`, and 2.99 for `5137`.

The repetition failure is dataset row 347, sample `ak_gh_image_0828_u776_1_1682163241022_05459`, where the decoder repeats `ayɛ` 62 consecutive times.

## Promotion Gates

| Gate | Result |
|---|---|
| Test WER below 30.86% | Failed: 32.84% |
| Paired interval below zero vs continuation | Passed |
| No material unexplained speaker regression | Failed: speaker 4430 +6.45 points |
| Zero repetition collapses or validated fallback | Failed: one collapse |
| Ghanaian listening review | Pending |

## Artifacts

- Training call: `fc-01KVK05E72XEYKY86K474AEBDM`
- Immutable evaluation call: `fc-01KVKND8SSVYMKJ1A7CXHE4D0Q`
- Full report: Modal volume `akan-speech-eval-results`, file `waxal-round2-immutable-test-v1.json`
- Compact summary: `waxal-round2-immutable-test-v1.summary.json`

The next ASR experiment must target the identified speaker and repetition failures without decoding `test_v1` again for checkpoint selection. New model selection remains development-only.

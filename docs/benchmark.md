# Waxal Akan ASR Benchmark v1

## Purpose

This benchmark is the fixed promotion set for Akan ASR experiments in this repository. It avoids convenient first-row samples and keeps model comparisons paired.

## Selection

- Dataset: `google/WaxalNLP`
- Config: `aka_asr`
- Source split: `test`
- Seed: `20260618`
- Rows: `99`
- Speakers: `33`
- Sampling: exactly three utterances per test speaker
- Audio: `0.5457` hours, 15.48–28.70 seconds per utterance
- Duplicate audio hashes: `0`
- Missing transcripts: `0`
- Test speakers overlapping train or validation: `0`

The exact public sample IDs and Dataset Viewer row IDs are tracked in `evals/waxal_aka_benchmark_v1.json`. Audio and transcripts are materialized locally and remain gitignored.

## Published Baseline

Model: `teckedd/whisper_small-waxal_akan-asr-v1`

| Decoder strategy | WER | CER | 95% bootstrap WER interval |
|---|---:|---:|---:|
| No forced language | 33.62% | 12.37% | 30.91%–36.79% |
| Yoruba proxy | 33.62% | 12.37% | 30.91%–36.79% |
| English proxy | 205.23% | 153.75% | 137.41%–281.57% |

The checkpoint stores `generation_config.language=yoruba`. Clearing it and allowing language auto-detection still produces the same outputs on benchmark v1, so no-forced-language is retained for future training without claiming that Whisper has native Akan language support.

## Tokenizer Finding

The current Whisper tokenizer averages `2.6015` tokens per reference word and `1.4345` non-space characters per token. This fragmentation is evidence for a later controlled tokenizer-extension experiment, not justification for replacing Whisper's vocabulary before a clean training baseline exists.

## Evidence Rule

A candidate must:

1. Report paired WER/CER against the frozen reference.
2. Report CER, per-speaker WER, and duration-bucket WER.
3. Pass manual Ghanaian review for orthography, code-switching, and meaning preservation.
4. Avoid using benchmark v1 for gradient updates or checkpoint selection.
5. Explain its training data mixture and why each corpus was included.

# ASR Experiment Plan

## Objective

Build an Akan ASR model from a defensible data recipe. The first priority is to
understand Waxal, GhanaNLP, correction data, and supplemental corpora deeply
enough that preprocessing and mixing decisions are justified before GPU
training.

The existing Waxal Whisper model remains reference evidence, not the project
goal:

- reference model: `teckedd/whisper_small-waxal_akan-asr-v1`
- reported WER: `34.2849`

## Phase 1: Reproducible Data

1. Prepare Waxal Akan ASR with released splits.
2. Prepare Waxal Akan ASR with speaker-safe splits.
3. Inspect both manifests for:
   - split sizes,
   - speaker overlap,
   - duration distribution,
   - empty transcripts,
   - top speaker dominance.

Decision: use speaker-safe split for honest Waxal eval unless it is too small or broken.

For every added corpus, inspect:

- license and consent
- speaker identity availability
- duration/noise/clipping/silence
- punctuation/casing/orthography conventions
- duplicate text/audio groups
- dialect/domain mix
- code-switching and numerals

## Phase 2: Baseline Evaluation

Evaluate:

- `teckedd/whisper_small-waxal_akan-asr-v1`
- `openai/whisper-small`
- `facebook/mms-1b-all`

Use the same manifest and decoding settings for all models.

## Phase 3: Controlled Fine-Tune Arms

Run small budget arms before a long job:

| Arm | Data | Tokenizer/language strategy | Purpose |
| --- | --- | --- | --- |
| A | Waxal speaker-safe | old Yoruba proxy | reproduce useful old behavior |
| B | Waxal speaker-safe | no forced language | test if Whisper can adapt without proxy bias |
| C | Waxal speaker-safe | English proxy | measure accidental English/code-switch impact |
| D | Waxal clean subset | best from A-C | test whether filtering beats more data |

Do not run another GhanaNLP-only continuation as a promotion path. GhanaNLP
should enter the serious recipe through harmonization, replay mixing, and
cross-corpus regression gates.

## Phase 4: Error Analysis

For each candidate, save:

- 50 best improvements,
- 50 worst regressions,
- repeated substitutions,
- short-clip errors,
- long-clip errors,
- diacritic/spelling errors,
- English insertion errors.

## Promotion Gate

A model can be published as a serious v2 only if:

- WER improves materially against the old baseline on the same held-out set,
- CER improves or stays acceptable,
- qualitative examples show better Akan, not just better normalization,
- model card documents data, split, tokenizer strategy, and limitations.

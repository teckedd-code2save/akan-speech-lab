# ASR Milestone 1: Expressive Akan Recognition

Status: planned

## Goal

Build the first useful Akan ASR release for real product use, not another
infrastructure-only experiment. The central goal is to understand and harmonize
the Akan speech datasets well enough that training quality follows from the data
recipe. The recognizer should handle faithful words plus expressive information
that matters in spoken Akan: punctuation, question/exclamation force, pauses,
emphatic stretching, and auditable tone-related annotations where the data
supports them.

Current reference checkpoint:

- model: `teckedd/whisper-small-waxal-round2-specaug-v1`
- immutable Waxal test: 32.84% WER / 11.79% CER
- known failures: one repetition collapse, speaker-4430 regression, punctuation
  absent because labels were normalized without punctuation

Target release:

- a documented corpus audit for every training source before training begins
- a contamination-safe multi-corpus recipe, not a single-corpus hope run
- measured WER/CER improvements or non-regressions on each relevant held-out
  set, with domain-specific interpretation instead of one global magic number
- no severe repetition collapses under the default decoder
- punctuation restoration that is evaluated separately from WER
- an annotation path for long vowels, hesitations, pauses, and emphasis
- a correction feedback loop that improves future batches without contaminating
  frozen test sets

## Scope Boundary

This milestone is ASR only. It does not train TTS, a dialogue system, or a
voice assistant runtime.

The output is not just a checkpoint. It must include:

1. a published model or adapter on Hugging Face,
2. a fixed eval report,
3. a live recording/upload test panel,
4. a correction capture/export path,
5. documentation of what the model can and cannot recognize.

## Expressive Recognition Targets

### 1. Words

The base ASR target remains faithful Akan/Twi/Fante words with Akan characters
preserved. This is still measured with normalized WER/CER, where punctuation is
removed so word accuracy is comparable to prior rounds.

### 2. Punctuation

Punctuation is not learned from the Round 2 acoustic labels because Round 2
explicitly removed punctuation before training. For Milestone 1, punctuation is
a separate supervised post-processing layer unless we have enough reliable
punctuated speech labels to train an end-to-end punctuated ASR variant.

Allowed punctuation target:

- `.`
- `,`
- `?`
- `!`

Metrics:

- punctuation precision, recall, and F1 by mark
- sentence-boundary F1
- question-mark F1
- WER before punctuation and after punctuation separately

Promotion rule: punctuation must not hide ASR word errors. A model can improve
punctuation F1 and still fail if word WER regresses.

### 3. Tonation and Tone-Related Evidence

Akan is tonal, but ordinary ASR transcripts do not contain full tone labels.
The model cannot be expected to output lexical tone unless the training data
contains consistent tone annotation.

Milestone 1 therefore separates two levels:

- **prosody evidence**: pauses, rising intonation, falling intonation,
  emphatic delivery, hesitations, and stretched syllables inferred from audio
- **lexical tone labels**: optional future annotation only when Ghanaian
  annotators provide a consistent scheme

Initial prosody tags:

- `<pause-short>`
- `<pause-long>`
- `<question-rise>`
- `<emphasis>`
- `<stretch>`
- `<hesitation>`

These tags are diagnostic at first. They are not included in ordinary WER.

Metrics:

- tag precision, recall, and F1 on a manually labeled dev set
- false-positive rate for question-rise and emphasis
- listener review: whether tags help a Ghanaian reader understand intent

### 4. Stretches and Long Speech

Stretched words such as drawn-out agreement, hesitation, distress, calling
someone's name, or emphatic repetition need explicit transcription rules.

Milestone 1 convention:

- preserve normal Akan spelling in the main transcript
- add `<stretch>` after the stretched word in the expressive transcript
- avoid spelling distortions like arbitrary repeated letters unless the source
  corpus already uses them consistently

Example:

```text
words: ɛyɛ
expressive: ɛyɛ <stretch>
```

## Data Plan

### Frozen Evaluation Data

Do not train on any existing frozen Waxal test rows. The Round 2 immutable test
remains a reporting-only benchmark.

### Training Data

Use only contamination-safe combinations:

1. Waxal train/dev partitions from the Round 2 speaker-safe split.
2. New user-recorded correction data with explicit consent.
3. GhanaNLP after corpus characterization: transcript conventions, duration
   distribution, audio quality, spelling variants, duplicate text/audio groups,
   and dialect/domain coverage.
4. Supplemental corpora only when license, split isolation, and transcript
   conventions are understood.

GhanaNLP should not be treated as bad data. The earlier GhanaNLP-only
continuation was a blunt adaptation recipe: it improved GhanaNLP while
regressing Waxal. The lesson is that GhanaNLP needs harmonization and replay or
balanced mixing, not dismissal.

### Corpus Understanding Checklist

For each dataset, record:

- source license and consent posture
- language/dialect labels and likely dialect mix
- sample rate, duration distribution, silence, clipping, and noise profile
- speaker identity availability and split strategy
- duplicate audio and duplicate transcript groups
- punctuation/casing conventions
- Akan character coverage, especially `ɛ` and `ɔ`
- spelling variants and code-switching rate
- words per second and transcript/audio mismatch flags
- domain mix: scripture, image captions, conversation, health, ecommerce,
  support, names, numbers, and locations

### Correction Data

Each correction row must store:

- audio hash and duration
- model ID that produced the hypothesis
- raw hypothesis
- corrected transcript with punctuation
- normalized transcript for WER
- optional expressive tags
- speaker/session ID if available
- consent and allowed usage
- timestamp and app version

Corrections are batched. They do not trigger immediate online training.

## Modeling Plan

### Stage A: ASR Word Model

Continue from the strongest Whisper-small path unless corpus analysis shows a
better base. The first serious run should be a replay-mixed supervised
continuation, not a new architecture search and not a single-corpus continuation.

Training mix:

- Waxal train replay
- selected high-error dev-like rows, excluding frozen test
- user corrections after deduplication and consent checks
- GhanaNLP rows after harmonization, mixed with replay rather than isolated
  continuation
- optional supplemental corpora only after license and contamination checks

Decoder:

- default greedy or small-beam decoder for comparable WER
- repetition fallback evaluated separately, not silently enabled

### Stage B: Punctuation Restorer

Train a small text/audio-aware restoration layer after ASR. Start with text
sequence tagging from corrected punctuated transcripts. Add acoustic features
only if text-only punctuation is weak on question and emphasis cases.

Inputs:

- ASR words
- optional pause durations from VAD/alignment
- optional confidence/timestamp features

Outputs:

- punctuated transcript
- mark-level confidence

### Stage C: Expressive Tagger

Build a small manually labeled set first. Do not train expressive tags from
unlabeled data.

Minimum pilot set:

- 200 short clips
- at least 3 Ghanaian reviewers for ambiguous tags
- labels for pause, question-rise, emphasis, stretch, and hesitation

Start with timestamp/VAD and pitch-energy features before adding a neural
tagger. This keeps the first result inspectable.

## Evaluation

Report all metrics separately:

| Layer | Primary Metric | Secondary Checks |
|---|---|---|
| Words | normalized WER/CER | speaker WER, duration buckets, repetition count |
| Punctuation | mark F1 | sentence-boundary F1, question F1 |
| Expressive tags | tag F1 | false positives, listener usefulness |
| Corrections | batch WER delta | no regression on frozen Waxal test |

Do not use punctuation or expressive tags to claim a lower WER.

## Acceptance Gate

Milestone 1 passes only if:

- the dataset audit explains what each corpus contributes and what risks it
  carries
- word WER/CER improves or stays stable on the relevant held-out sets for the
  intended release scope
- no severe repetition-collapse regression is introduced
- punctuation restoration beats a no-punctuation baseline with useful question
  and sentence-boundary F1
- expressive tags are available at least as a diagnostic preview with manual
  review evidence
- the UI lets a user record audio, inspect hypothesis, correct transcript,
  add expressive tags, and export a clean training batch

## Immediate Implementation Order

1. Add correction capture to the UI and local JSONL store.
2. Add export script for clean correction manifests.
3. Add punctuation-preserving and WER-normalized transcript fields.
4. Add a small punctuation restoration baseline.
5. Add expressive tag schema and manual labeling UI fields.
6. Build a multi-corpus training manifest only after the corpus audit is complete.
7. Train one replay-mixed ASR continuation only after the manifest explains why
   each corpus is included and how regressions are measured.
8. Publish only if the frozen gates pass.

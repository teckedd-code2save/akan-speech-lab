# Asante Twi TTS: Research and Execution Record

## Decision

The primary experiment is `microsoft/speecht5_tts` with
`microsoft/speecht5_hifigan`. Both model repositories are MIT licensed. Speaker
conditioning uses SpeechBrain x-vectors under Apache 2.0. The only permitted
fallback is raw-grapheme VITS, and it is gated behind a failed SpeechT5 pilot.

This is a data-first decision rather than a claim that SpeechT5 is the newest
architecture. The [Hugging Face SpeechT5 fine-tuning course](https://huggingface.co/learn/audio-course/chapter6/fine-tuning)
provides a reproducible low-resource path: 16 kHz audio, character audit,
written-out numbers, duration filtering, speaker embeddings, gradient
checkpointing, and fixed evaluation data. Its
[evaluation chapter](https://huggingface.co/learn/audio-course/chapter6/evaluation)
also makes human MOS central; no objective score is accepted as a substitute
for Ghanaian listeners.

## Architecture exclusions

- [YourTTS](https://huggingface.co/papers/2112.02418) informs VAD trimming,
  speaker conditioning, loudness consistency, and raw graphemes where a trusted
  G2P system is unavailable.
- [VITS](https://huggingface.co/papers/2106.06103) is the one fallback because it
  learns monotonic alignment and waveform generation jointly.
- [StyleTTS2](https://huggingface.co/papers/2306.07691) is deferred. Its published
  recipe adds phonemes, diffusion, WavLM discrimination, and alignment
  dependencies validated on much larger English corpora.
- `facebook/mms-tts-aka` remains a CC-BY-NC-4.0 diagnostic reference only.

## Orthography and tokenizer

SpeechT5's stock tokenizer maps Akan `ɛ` and `ɔ` to unknown tokens. The pipeline
therefore audits the full accepted character inventory, adds missing graphemes,
resizes model embeddings, and requires zero unknown-token rows before training.
It never transliterates those letters. Numeric/currency tokens are quarantined
until a Ghanaian-reviewed spoken-Twi expansion exists; the system does not guess.

## Data stages

### Diagnostic corpus

`Farmerline-DCS-HCI25/akan_tts_dataset` is pinned to revision
`050b308e66155bb5e3d843b01ebca51f9d7056e2` and CC-BY-4.0. The Akosua material
is roughly 2.19 hours. It validates the pipeline only because its public
documentation does not establish explicit speaker consent for commercial voice
synthesis.

Preparation decodes and resamples to 16 kHz, trims edge silence, normalizes
loudness, hashes processed audio and normalized text, rejects corrupted/empty,
clipped, low-level, excessive-silence, duplicate, and unresolved-number rows,
then freezes 80/10/10 splits by normalized-text hash. Equal text can never cross
a split.

The first frozen CPU audit completed under durable call
`fc-01KVNT6W6G9BDWKRJCAZ57SZQ8`: 1,467 Akosua rows inspected, 861 accepted,
and 715/76/70 assigned to train/validation/test with zero normalized-text
overlap. Rejections included 353 duration flags, 209 excessive-silence flags,
162 duplicate-audio flags, 254 duplicate-text flags, and one empty transcript.
Counts overlap because one row may carry more than one flag.

### Production corpus

Commercial promotion requires one explicitly consented Asante Twi speaker:
12 hours, 24 kHz mono source recordings, 4-12 second utterances, compensation
and withdrawal terms, and 55% conversational / 15% health / 15% commerce and
support / 10% Ghanaian entities and numbers / 5% English code-switching. Dev and
test each receive 10% grouped by normalized text.

## Training gates

1. CPU corpus and tokenizer audits must pass.
2. A 20-step smoke run must finish with finite loss and synthesis.
3. A 32-example overfit must show materially falling loss and usable alignment.
4. A 1,000-step diagnostic pilot generates the fixed prompt suite.
5. A human gate records intelligibility, MOS, repetitions, skips, and preference.
6. The 8,000-step ceiling is available only after that pilot gate passes.
7. VITS is available only after a documented SpeechT5 pilot failure.

Every Modal call is submitted to the stable `akan-speech-tts` deployment,
persisted by call ID, limited to one container, bounded to one retry, and scaled
to zero after 30 seconds.

## First Gate Results

- CPU preparation: call `fc-01KVNT6W6G9BDWKRJCAZ57SZQ8`, passed.
- First smoke attempt: failed at step zero because reentrant PyTorch gradient
  checkpointing attempted a second backward pass. No learned checkpoint was
  produced. The runner now fixes `use_reentrant=False`.
- Corrected 20-step smoke: call `fc-01KVNTV3NWY621Z166A1W3186R`, passed in
  33.36 seconds; validation loss 1.3720. The tokenizer added only `ɔ`, `ɛ`, and
  `ʼ`, and zero accepted rows contain an unknown token.
- 32-example overfit: call `fc-01KVNTXQDV97AJ4FKPECSQ6MYJ`, completed in
  123.87 seconds. Logged loss fell from 1.8701 to 0.6633 (64.5%) and validation
  loss reached 0.5703. The pilot remains locked pending Ghanaian listening
  review of the saved synthesis.

## Evaluation

The frozen 120-prompt suite must cover conversational Twi, health, commerce,
questions, Ghanaian names/locations/numbers, and code-switching. Corpus-derived
prompts may fill general categories, but domain prompts remain `needs_review`
until a Ghanaian speaker approves their wording.

Reported outputs are blinded MOS and preference, listener transcription
accuracy, ASR round-trip WER/CER as a diagnostic, speaker cosine similarity,
duration/silence/clipping/repetition failures, real-time factor, and p95 latency.
The diagnostic gate is 80% listener transcription accuracy with no catastrophic
failures. Production requires 95%, MOS >= 4.0 from five Ghanaian listeners, zero
repeated/skipped/hallucinated phrases, RTF < 0.5, and p95 < 1.5 seconds for a
ten-word response.

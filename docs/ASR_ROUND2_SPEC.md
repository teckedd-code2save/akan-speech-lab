# ASR Round 2: Contamination-Safe Waxal Training Specification

Status: complete; checkpoint not promoted. See [ASR Round 2 Results](ASR_ROUND2_RESULTS.md).

## 1. Goal

Train a stronger Twi/Akan ASR model with defensible evaluation and no known train/test contamination. The primary promotion target is below **30.86% WER** on the untouched 1,522-row Waxal test split, representing at least a 10% relative gain over the original 34.28% baseline.

This round changes one central assumption: the existing fine-tuned checkpoint remains a benchmark, but is not the training base. It has already seen all published Waxal training speakers, so it cannot support a new genuinely speaker-unseen development split from those speakers.

## 2. Dataset Choice

### Included

`google/WaxalNLP`, configuration `aka_asr`, and no other corpus.

Reasons:

- It produced the best measured model so far: 32.77% WER on the complete test split.
- It exposes speaker IDs.
- Its published test speakers have zero overlap with train and validation speakers in the stored audit.
- Audio, transcripts, speaker IDs, and split membership can be fingerprinted and frozen.
- Using one source avoids unknown cross-corpus recording, speaker, or transcript overlap.

### Excluded

- `ghananlpcommunity/twi-speech-text-multispeaker-16k`
- `AfriSpeech/youversion-african-speech`
- Any YouVersion-derived speech corpus
- Synthetic or external noise corpora
- Pseudo-labelled web audio

These sources are excluded because we cannot currently prove that their speakers, source recordings, or transcript material do not overlap Waxal. GhanaNLP also exposes no speaker IDs. They may become eligible only after provenance and overlap checks provide positive evidence, not assumptions.

## 3. Contamination Boundary

The stored Waxal audit contains:

- 10,107 published training rows
- 1,123 published validation rows
- 1,522 published test rows
- 134 known speakers
- zero test/train speaker overlap
- zero test/validation speaker overlap
- 99 train/validation overlapping speakers
- zero cross-split normalized-transcript duplicate groups

The published validation split is therefore unsuitable as a speaker-unseen development set.

### New split construction

1. Concatenate only the published `train` and `validation` rows.
2. Group every row by `speaker_id`.
3. Assign speaker groups deterministically with SHA-256 and seed 42:
   - 85% of speakers to `train_v2`
   - 15% of speakers to `dev_v2`
4. Keep every utterance from one speaker in exactly one partition.
5. Keep the published `test` split unchanged as `test_v1`.
6. Freeze row IDs, speaker IDs, transcript fingerprints, audio fingerprints, source revision, split algorithm, and seed in a versioned manifest.

### Mandatory assertions

Training must refuse to start unless all assertions pass:

- zero speaker overlap between train, dev, and test
- zero decoded-audio SHA-256 overlap between partitions
- zero sample-ID overlap between partitions
- zero normalized-transcript fingerprints crossing partitions
- no test row in preprocessing statistics fitted from training data
- no test decoding during checkpoint selection
- exact dataset revision recorded

The test split is evaluated only after a checkpoint is selected from development WER.

## 4. Audio Preprocessing

The Hugging Face Audio Course recommends matching model sampling rate, filtering unsuitable durations, and converting waveforms with the model feature extractor. Whisper expects 16 kHz audio and converts it to log-mel features.

References:

- [Audio Course: preprocessing an audio dataset](https://huggingface.co/learn/audio-course/en/chapter1/preprocessing)
- [Audio Course: fine-tuning ASR](https://huggingface.co/learn/audio-course/chapter5/fine-tuning)
- [Transformers Whisper documentation](https://huggingface.co/docs/transformers/model_doc/whisper)

### Ordered pipeline

1. **Pin source revision**
   - Resolve and store the exact Waxal dataset revision and parquet file hashes.
   - Reason: future Hub updates must not silently alter an experiment.

2. **Decode consistently**
   - Decode with one library version and convert to mono float32.
   - Reason: fingerprints and signal checks must be reproducible.

3. **Resample to 16 kHz**
   - Use `datasets.Audio(sampling_rate=16000)` or a well-tested resampler.
   - Never downsample by dropping samples.
   - Reason: Whisper was trained for 16 kHz input; correct anti-alias filtering is required.

4. **Fingerprint before augmentation**
   - Hash canonical decoded PCM plus sample rate.
   - Hash normalized transcript separately.
   - Reason: detect duplicate audio even when filenames differ.

5. **Remove invalid rows**
   - Empty audio or transcript
   - Decode failures
   - NaN or infinite samples
   - Exact duplicate audio
   - Exact duplicate sample IDs

6. **Duration filtering**
   - Keep 0.4 to 30.0 seconds.
   - Reason: Whisper operates on 30-second windows; very short clips disproportionately encourage fragment insertion and repetition.
   - Report removals by speaker and split before applying them.

7. **Signal-quality flags**
   - Peak clipping ratio
   - RMS energy and near-silence ratio
   - Excessive leading/trailing silence
   - Transcript words per second
   - Do not silently discard flagged rows. Produce a review report and use fixed thresholds approved before training.

8. **Text normalization**
   - Unicode NFC
   - Lowercase
   - Collapse whitespace
   - Remove punctuation for this acoustic-model target
   - Preserve `ɛ`, `ɔ`, diacritics, apostrophe-bearing forms when linguistically meaningful, Twi/Fante spelling, names, and code-switching
   - Do not transliterate into English or Yoruba spelling

9. **Feature extraction**
   - Use the Whisper processor from the base checkpoint.
   - Produce 80-bin log-mel input features through `WhisperFeatureExtractor`.
   - Keep attention masks because SpecAugment must not mask padding.

10. **Training-only augmentation**
    - Enable Whisper SpecAugment only on `train_v2`.
    - `apply_spec_augment=True`
    - `mask_time_prob=0.05`
    - `mask_time_length=10`
    - `mask_feature_prob=0.05`
    - `mask_feature_length=10`
    - Reason: regularize acoustic features without importing external audio or creating cross-corpus contamination.
    - No augmentation on development or test.

## 5. Model Choice

### Base model

`openai/whisper-small`

Reasons:

- It is the base of the strongest measured Akan checkpoint in this lab.
- The existing Waxal fine-tune reached 34.28% WER and the continuation reached 32.77%.
- It fits an L4 comfortably with effective batch size 32.
- The current evidence does not justify paying for Whisper Medium before the clean Small recipe is established.
- Starting from the original base allows a genuinely unseen speaker-development partition.

### Tokenizer and decoder policy

- Keep Whisper's multilingual tokenizer.
- Training tokenizer prefix: `language="yoruba", task="transcribe"` as an explicit proxy experiment.
- Inference/evaluation: no forced language token, task `transcribe`.
- Reason: the previous from-base no-language arm was substantially worse, while the Yoruba-prefixed training recipe produced the useful baseline. Inference tests found no-forced-language and stored Yoruba prompting identical for the best checkpoint.
- Do not build a new tokenizer in this round. Replacing Whisper's tokenizer would require relearning decoder embeddings from limited Akan data and would confound the data-pipeline experiment.

The Yoruba prefix is metadata for Whisper's decoder, not a request to convert Akan spelling into Yoruba. Akan transcripts remain unchanged apart from the conservative normalization above.

## 6. Training Configuration

| Parameter | Value | Reason |
|---|---:|---|
| Base | `openai/whisper-small` | Empirically strongest cost-controlled family |
| Precision | FP16 | L4 throughput and memory |
| Train batch/device | 8 | Proven to fit L4 |
| Gradient accumulation | 4 | Effective batch 32 |
| Eval batch/device | 8 | Proven evaluation fit |
| Optimizer | AdamW | Established Whisper fine-tuning default |
| Learning rate | `1e-5` | Reproduce successful from-base Waxal scale |
| Weight decay | `0.01` | Regularization |
| Warmup | 200 steps | Stabilize from-base adaptation |
| Scheduler | Linear decay | Matches successful baseline recipe |
| Maximum steps | 2,000 | Covers the prior best region without open-ended spend |
| Evaluate | Every 200 steps | Directly comparable checkpoints |
| Save | Every 200 steps | Recovery and bounded storage |
| Save limit | 3 | Best plus recent recovery points |
| Generation beams | 1 | Stable and cost-controlled WER comparison |
| Generation max length | 225 | Existing tested setting |
| Gradient checkpointing | Enabled | Memory control |
| SpecAugment | Time 0.05 / feature 0.05 | Training-only acoustic regularization |
| Seed | 42 | Deterministic first run |
| Best-model metric | Development WER | Correct checkpoint objective |
| Early stopping | Patience 3 evaluations | Stop after 600 steps without improvement |

Do not add label smoothing, speed perturbation, external noise, LoRA, or a larger model in the same run. This round isolates the effects of clean speaker splitting, filtering, normalization, and SpecAugment.

## 7. Evaluation Protocol

### During training

- Compute normalized WER and CER on `dev_v2` every 200 steps.
- Store substitutions, deletions, insertions, repetition-collapse count, duration buckets, and speaker macro-WER.
- Use macro-WER by speaker as a diagnostic, but select by corpus WER for continuity with published results.
- Stop early after three non-improving evaluations.

### After checkpoint selection

Run once on frozen `test_v1`:

- WER and CER
- 5,000-sample paired bootstrap against:
  - `teckedd/whisper_small-waxal_akan-asr-v1`
  - `teckedd/whisper-small-waxal-akan-continuation-v1`
- Better/tied/worse row counts
- Speaker and duration buckets
- Repetition-collapse count
- Largest regressions and gains
- Ghanaian listening review of a fixed sample set

### Promotion gate

All conditions are required:

- test WER below 30.86%
- paired 95% WER-difference interval below zero against the 32.77% candidate
- no material speaker bucket regresses without explanation
- zero repetition collapses, or a validated retry/fallback with measured final WER
- Ghanaian review accepts representative transcripts

## 8. Cost-Controlled Execution

1. Build manifests and audits locally or on Modal CPU.
2. Run a two-step L4 smoke on 32 train and 16 dev rows.
3. Run a 200-step pilot.
4. Continue only if development WER is on a plausible path toward the current 32.77% test benchmark.
5. Persist every checkpoint before evaluation.
6. Use resumable attached runs; avoid repeating completed steps.
7. Stop Modal immediately after training and final evaluation.

No Modal job starts until the frozen manifests, assertions, and audit report are visible in the UI.

## 9. Deliverables

- Versioned contamination report
- Frozen train/dev/test manifests
- Dataset card with exclusions and limitations
- Training config and source revision
- Checkpoint metrics at every evaluation boundary
- Full paired test report
- Listening-review pack
- Published model only if every promotion gate passes

After ASR promotion, the next independent milestone is a commercially licensed Akan TTS dataset and baseline, followed by the realtime microphone -> ASR -> response -> TTS loop.

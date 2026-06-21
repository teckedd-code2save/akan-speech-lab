# ASR Round 3: Whisper Medium LoRA Specification

Status: complete; pilot failed the development gate. See [Round 3 results](ASR_ROUND3_LORA_RESULTS.md).

## Goal

Test whether additional model capacity can improve Akan ASR without paying for a full Whisper Medium fine-tune. Round 3 is allowed to continue only if its 200-step pilot beats the Round 2 selected checkpoint's **32.14% unseen-speaker development WER**.

## Fixed Data Boundary

- Reuse the audited Waxal Round 2 dataset: 9,133 train / 2,091 dev / 1,522 immutable test rows.
- Preserve all speaker, transcript, sample-ID, and decoded-audio isolation assertions.
- Use only train for optimization and dev for model selection.
- Do not decode `test_v1` unless the final selected LoRA adapter beats Round 2 on dev.
- Do not combine GhanaNLP, AfriSpeech, YouVersion, synthetic speech, or pseudo-labels.

## Model and Adapter

| Setting | Value |
|---|---|
| Base model | `openai/whisper-medium` |
| Method | LoRA through PEFT |
| Target modules | attention `q_proj`, `v_proj` |
| Rank | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Bias | none |
| Precision | FP16 |
| Tokenizer prefix | Yoruba proxy for training only |
| Evaluation decoding | transcribe task, no forced language token |

LoRA is used because Whisper Medium fits inference on an L4, while adapter-only optimization materially reduces trainable parameters and optimizer memory. A new tokenizer is excluded because it would reinitialize decoder embeddings and confound the capacity experiment.

## Training

| Setting | Pilot | Full |
|---|---:|---:|
| Steps | 200 | 1,200 maximum |
| Learning rate | 5e-4 | 5e-4 |
| Warmup | 40 | 100 |
| Batch/device | 4 | 4 |
| Gradient accumulation | 8 | 8 |
| Effective batch | 32 | 32 |
| Eval batch/device | 4 | 4 |
| Eval/save interval | 200 | 200 |
| Early stopping | n/a | patience 2 |
| SpecAugment | time 0.05 / feature 0.05 | same |

The pilot gate compares final dev WER to 32.14%. The full run is blocked if the pilot does not improve it. The full selected adapter is evaluated once on `test_v1`; ASR work then stops regardless of outcome so the lab can move to TTS.

## Operations

- Stable deployed Modal app: `akan-speech-asr-round3-lora`.
- Submit detached calls with persisted `fc-...` IDs.
- `max_containers=1`, bounded retries, 30-second scale-down.
- Adapter checkpoints persist to `akan-speech-checkpoints`.
- No always-on GPU service and no attached ephemeral training app.

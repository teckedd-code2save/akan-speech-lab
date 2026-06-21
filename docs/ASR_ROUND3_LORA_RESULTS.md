# ASR Round 3 Medium-LoRA Results

Date: 2026-06-21

## Decision

**Stop the experiment at the pilot gate and close ASR development.** Whisper Medium LoRA reached **43.73% unseen-speaker dev WER**, versus **32.14%** for the selected Round 2 Whisper Small checkpoint. Full Medium-LoRA training and immutable-test decoding were therefore blocked.

## Configuration

- Base: `openai/whisper-medium` (768.58M parameters)
- Trainable LoRA parameters: 4.72M (0.614%)
- Targets: attention `q_proj` and `v_proj`
- Rank / alpha / dropout: 16 / 32 / 0.05
- Training: 200 steps, effective batch 32, `5e-4` learning rate, 40 warmup steps
- Data: the unchanged contamination-safe Waxal Round 2 train/dev partitions
- Adapter checkpoint: `pilot-whisper-medium-waxal-round3-lora-v1/checkpoint-200`
- Modal call: `fc-01KVKS2EBH3NXBKD3VA92TYAY9`

## Result

| Model | Development WER |
|---|---:|
| Round 2 Whisper Small checkpoint 1200 | **32.14%** |
| Round 3 Whisper Medium LoRA checkpoint 200 | 43.73% |

The LoRA adapter trained correctly, used only 0.614% trainable parameters, and produced a valid checkpoint. The result shows that increased frozen model capacity plus attention-only LoRA does not reproduce the quality of full-model acoustic adaptation on this corpus within 200 steps.

The Modal wrapper was cancelled only after checkpoint 200 and its complete 2,091-row dev evaluation were durably stored. Cancellation stopped a redundant second evaluation and did not invalidate the pilot metric. Modal then reported zero active tasks.

## Learnings

1. LoRA wiring and cost control worked; model quality did not.
2. Attention `q_proj`/`v_proj` adapters alone are too constrained for this acoustic/domain shift at the tested rank and schedule.
3. A larger frozen base is not automatically better than fully adapting Whisper Small.
4. Full training is not justified because the preregistered pilot missed the gate by 11.59 absolute WER points.
5. `test_v1` was not decoded for Round 3, preventing another adaptive test round.

ASR is now frozen at the Round 2 research checkpoint and the published continuation remains the production-facing baseline until a future project brings cleaner or broader licensed Akan speech data. The lab now moves to commercially usable Akan TTS.

# ASR Pipeline Loop

Status: active operating model

The lab is a research pipeline, not a UI demo. Each pass should produce a
reviewable artifact, preferably a Hugging Face model or dataset card, plus the
evidence needed to decide what to do next.

## One Pass

1. **Pick**
   - Choose the exact corpora and model base.
   - State why each dataset is included.
   - State what would make the pass fail.

2. **Prepare**
   - Build immutable manifests with stable row IDs, audio hashes, text hashes,
     source license, split, speaker if available, duration, sample rate, and
     quality flags.
   - Keep raw text, punctuated text, WER-normalized text, and optional expressive
     tags as separate fields.

3. **Sanitize**
   - Remove or quarantine empty audio/text, duplicates across splits, clipping,
     excessive silence, transcript/audio mismatch, and impossible duration/text
     ratios.
   - Harmonize Akan characters, numerals, punctuation policy, code-switching,
     and spelling variants without erasing dialectal evidence.

4. **Train**
   - Train only from an approved manifest.
   - Prefer replay-mixed adaptation over single-corpus continuation.
   - Use bounded Modal jobs with persisted call IDs, `max_containers=1`, and
     scale-to-zero.

5. **Test**
   - Decode fixed held-out sets.
   - Report WER/CER, per-corpus results, duration buckets, speaker buckets when
     available, repetition loops, diacritic errors, punctuation F1, and
     qualitative Akan examples.

6. **Save Results**
   - Persist metrics, predictions, configs, manifests, training state, and model
     card draft.
   - The result must be reproducible without rerunning the training job.

7. **Publish Review Artifact**
   - Publish a Hugging Face model, adapter, dataset, or evaluation artifact when
     the pass is worth external review.
   - Experimental artifacts are allowed, but the card must say exactly what
     passed and failed.

8. **Compare**
   - Compare against previous artifacts with paired rows.
   - Do not compare unmatched sample packs as proof.

9. **Review Corrections**
   - Listen to failures.
   - Capture corrected transcripts, punctuation, expressive tags, speaker/session
     metadata, consent, and model version.

10. **Restrategize**
    - **Learn:** add new corrected or harmonized data that fixes real gaps.
    - **Relearn:** rerun with better preprocessing, mixing, tokenizer, decoder,
      or hyperparameters.
    - **Unlearn:** remove harmful rows, noisy conventions, overrepresented
      domains, leakage, or augmentation that damages held-out speech.

## Artifact Contract

Every serious pass should leave:

- dataset manifest hash
- training config hash
- model base and checkpoint ID
- Hugging Face artifact URL when published
- WER/CER by corpus
- qualitative examples
- failure taxonomy
- next decision: learn, relearn, unlearn, or stop

## Research Anchors

The research spine explains why this loop is structured this way:

- Whisper and HF Audio Course anchor the fine-tuning workflow.
- SpecAugment informs train-only feature masking.
- Continual-learning and replay papers justify mixed replay over single-corpus
  continuation.
- Punctuation restoration papers keep punctuation separate from WER.
- wav2vec 2.0 is the plateau-escape branch if Whisper stops improving after
  data quality and mixing are fixed.


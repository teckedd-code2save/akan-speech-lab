# ASR Research Spine

Status: active reference set

This is the working research spine for Akan ASR v2. These papers and articles
are selected because each one changes a concrete pipeline decision. The goal is
not to cite broadly; it is to use research to drive data understanding,
preprocessing, training, evaluation, and eventual low-WER releases.

## 1. Foundation Model Choice

### Whisper: robust multilingual transfer

- Source: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- Use in this lab: keep Whisper as the main supervised fine-tuning path while
  it remains the strongest empirical Akan base.
- Pipeline consequence: do not replace Whisper just because Akan is not a
  first-class language token; first fix data, labels, decoding, and corpus mix.

### Hugging Face Audio Course: Whisper fine-tuning procedure

- Source: [HF Audio Course, Chapter 5: Fine-tuning ASR](https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning)
- Use in this lab: anchor the standard training loop: dataset cleaning,
  processor/tokenizer handling, data collator, generation metrics, WER, and
  Hub publishing.
- Pipeline consequence: every experiment must have a manifest, reproducible
  preprocessing, fixed eval split, and model card.

## 2. Low-Resource Learning and Plateau Escape

### wav2vec 2.0: self-supervised speech representation learning

- Source: [wav2vec 2.0](https://arxiv.org/abs/2006.11477)
- Use in this lab: if Whisper fine-tuning plateaus around the low-30s WER,
  test whether self-supervised acoustic representations trained or adapted on
  Akan audio give a better encoder for local speech.
- Pipeline consequence: collect and preserve unlabeled Akan audio, because it
  may become useful before it has transcripts.

### Multilingual and cross-lingual low-resource ASR

- Source: [Improving Low-Resource Speech Recognition through Multilingual Fine-tuning](https://aclanthology.org/2023.rocling-1.8.pdf)
- Use in this lab: treat related or useful languages as a controlled training
  signal, not as accidental noise.
- Pipeline consequence: if we add non-Akan corpora, each row needs language ID,
  script/orthography normalization, and held-out Akan regression checks.

## 3. Data Augmentation

### SpecAugment

- Source: [SpecAugment](https://arxiv.org/abs/1904.08779)
- Use in this lab: apply feature masking to reduce overfitting without changing
  transcripts or creating synthetic label noise.
- Pipeline consequence: augmentation is training-only; dev/test stay untouched.

### Cross-lingual mappings and augmentation for low-resource ASR

- Source: [Learning Cross-lingual Mappings for Data Augmentation to Improve Low-Resource Speech Recognition](https://www.isca-archive.org/interspeech_2023/farooq23_interspeech.html)
- Use in this lab: evaluate augmentation as a data-scarcity tool only after
  real corpus cleaning is done.
- Pipeline consequence: synthetic or perturbed data must be tagged in the
  manifest and evaluated separately from original speech.

## 4. Continual Learning and Corpus Mixing

### Experience replay against catastrophic forgetting

- Source: [Continual Layer-Specific Fine-Tuning for German Speech Recognition](https://link.springer.com/chapter/10.1007/978-3-031-44195-0_40)
- Use in this lab: GhanaNLP should not be dismissed because a GhanaNLP-only
  continuation regressed Waxal. The right next path is replay-mixed adaptation.
- Pipeline consequence: every new corpus mixture must include regression
  slices, especially Waxal replay, correction data, and domain-specific evals.

### Online continual ASR

- Source: [Online Continual Learning of End-to-End Speech Recognition Models](https://www.isca-archive.org/interspeech_2022/yang22w_interspeech.html)
- Use in this lab: correction feedback should be batched and gated, not used
  for uncontrolled online updates.
- Pipeline consequence: user corrections are saved with audio hash, model ID,
  corrected transcript, consent, and eval split status before any retraining.

## 5. Punctuation and Expressive ASR

### Punctuation restoration for ASR transcripts

- Source: [LSTM for Punctuation Restoration in Speech Transcripts](https://www.isca-archive.org/interspeech_2015/tilk15_interspeech.html)
- Use in this lab: punctuation should start as a separate restoration layer so
  it does not obscure word-recognition errors.
- Pipeline consequence: report punctuation precision, recall, and F1 separately
  from WER/CER.

### Acoustic punctuation prediction

- Source: [Adapting ASR Models for Speech-to-Punctuated-Text Recognition](https://aclanthology.org/2025.icnlsp-1.8.pdf)
- Use in this lab: question marks, pauses, and emphatic punctuation may need
  acoustic evidence, not just text post-processing.
- Pipeline consequence: store timestamps, VAD pauses, and optional prosody tags
  so punctuation can become audio-aware later.

### Spontaneous speech punctuation data

- Source: [SponSpeech](https://arxiv.org/abs/2409.11241)
- Use in this lab: scripted corpora underrepresent stutters, interruptions,
  long pauses, and informal speech.
- Pipeline consequence: our correction data should include real spontaneous
  Ghanaian speech, not only read sentences.

## 6. Practical Progression Toward Very Low WER

The research-backed path is:

1. Audit and harmonize Waxal, GhanaNLP, correction data, and any supplemental
   corpus before training.
2. Keep Whisper as the near-term base while it remains strongest.
3. Train replay-mixed supervised continuations, not single-corpus hope runs.
4. Use SpecAugment and conservative audio perturbation only after labels are
   clean.
5. Keep punctuation and expressive tags as separately evaluated layers.
6. If Whisper plateaus after clean data and replay mixing, test wav2vec-style
   acoustic adaptation or a stronger encoder with the same frozen eval gates.


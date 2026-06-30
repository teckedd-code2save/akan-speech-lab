# GhanaNLP Manifest Audit v0.5

Date: 2026-06-30

Command:

```bash
.venv/bin/python scripts/audit_gnlp_manifest.py
```

Input manifest:

```text
data/manifests/serendepify-gsl-asr-ak-waxal-gnlp-whisper-small-replay-fullft-v0.1.jsonl
```

Outputs:

```text
outputs/audits/gnlp_manifest_v0.5/report.json
outputs/audits/gnlp_manifest_v0.5/audited_rows.jsonl
outputs/audits/gnlp_manifest_v0.5/clean_candidates.jsonl
outputs/audits/gnlp_manifest_v0.5/clean_train_candidates.jsonl
outputs/audits/gnlp_manifest_v0.5/clean_validation_candidates.jsonl
outputs/audits/gnlp_manifest_v0.5/flagged_rows.csv
```

These output files are generated artifacts and are intentionally not committed.

## Result

GhanaNLP is not hopeless, but it is not clean enough for blind training.

| Check | Count |
|---|---:|
| Total rows audited | 11,830 |
| Clean candidates | 6,704 |
| Clean train candidates | 6,033 |
| Clean validation candidates | 326 |
| Clean test candidates | 345 |
| Missing speaker IDs | 11,830 |
| Too-short audio | 3,088 |
| Suspicious word/audio speed | 2,162 |
| Duplicate transcripts | 2,038 |
| Too few words | 1,010 |

Duration median is 2.06 seconds. Median word count is 5. Median words per second is 2.86, but the maximum detected ratio is 78.26 words per second, which is a clear alignment/text mismatch signal.

## Interpretation

The v0.5 GhanaNLP-only run improved GhanaNLP validation WER from 113.97% to 87.55%, so the corpus contains learnable acoustic/text signal. The same checkpoint regressed Waxal from 32.59% to 35.75%, so direct GhanaNLP-only adaptation damages the current best Waxal path.

The manifest audit explains why:

- GhanaNLP has no stable speaker IDs in the frozen manifest, so speaker-disjoint evaluation cannot be claimed from this source alone.
- A large block of clips are too short for stable ASR learning.
- Many rows have suspicious word-per-second ratios, implying transcript/audio mismatch or segmentation problems.
- Duplicate transcripts are common enough to distort training and evaluation if not grouped.

## Next Recipe

Do not run another raw GhanaNLP full fine-tune.

Next ASR candidate should:

1. Train on `clean_train_candidates.jsonl` only for GhanaNLP.
2. Keep Waxal replay in every batch or alternating epoch.
3. Keep Waxal dev/test as the regression anchor.
4. Track per-corpus WER separately.
5. Stop on Waxal regression or repetition collapse.
6. Preserve a separate GhanaNLP validation slice for adaptation signal.

The expected codename shape:

```text
serendepify-gsl-asr-ak-waxal-gnlpclean-whisper-small-replay-fullft-v0.6
```

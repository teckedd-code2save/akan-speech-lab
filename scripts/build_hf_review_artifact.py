from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from akan_speech.asr.artifacts import FIRST_ASR_REVIEW


ROOT = Path(__file__).resolve().parents[1]


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def render_readme(stats: dict) -> str:
    spec = FIRST_ASR_REVIEW
    return f"""---
language:
- tw
- ak
license: apache-2.0
tags:
- automatic-speech-recognition
- akan
- twi
- ghanaian-speech-lab
- dataset-manifest
- asr-pipeline
- serendepify-gsl
pretty_name: Serendepify GSL ASR v0.1 Review Artifact
task_categories:
- automatic-speech-recognition
---

# {spec.code_name}

This is the first pushed **Ghanaian Speech Lab ASR pipeline artifact**. It is a
review artifact for the v0.1 Akan ASR pass, not a trained model checkpoint.

Expected future model repo:

```text
{spec.hf_repo}
```

## What This Artifact Contains

- `data/manifest.jsonl`: harmonized Waxal + GhanaNLP manifest references.
- `reports/sanitize.json`: sanitization report and corpus counts.
- `pipeline/latest.json`: pipeline run log.
- `configs/asr.yaml`: planned ASR training config.
- `model_card_draft/README.md`: model-card contract for the future trained model.

The manifest contains references and metadata, not redistributed audio.

## Pipeline State

Completed:

- Pick
- Prepare
- Sanitize

Active next stage:

- Train

## Sanitization Summary

| Metric | Count |
|---|---:|
| Candidate rows | {stats.get("candidate_rows", 0):,} |
| Accepted rows | {stats.get("accepted_rows", 0):,} |
| Quarantined rows | {stats.get("quarantined_rows", 0):,} |
| Waxal accepted | {stats.get("accepted_by_corpus", {}).get("waxal", 0):,} |
| GhanaNLP accepted | {stats.get("accepted_by_corpus", {}).get("gnlp", 0):,} |

Quarantine reasons:

```json
{json.dumps(stats.get("quarantine_reasons", {}), indent=2, ensure_ascii=False)}
```

## Naming

```text
{spec.code_name}
```

- `serendepify-gsl`: Ghanaian Speech Lab
- `asr`: automatic speech recognition
- `ak`: Akan-family target
- `waxal-gnlp`: Waxal + GhanaNLP
- `whisper-small`: planned model family
- `replay-fullft`: planned replay-mixed full fine-tuning
- `v0.1`: first review candidate

## Status

This artifact proves the pipeline has reached the end of the Sanitize stage.
It does not claim ASR improvement yet. The next artifact should be a durable
Modal training job state and checkpoint output.
"""


def build_package(output_dir: Path) -> dict:
    spec = FIRST_ASR_REVIEW
    sanitize_report = ROOT / "evals/reports" / f"{spec.code_name}-sanitize.json"
    manifest = ROOT / "data/manifests" / f"{spec.code_name}.jsonl"
    pipeline_log = ROOT / "outputs/pipeline_runs" / spec.code_name / "latest.json"
    config = ROOT / "configs/asr" / f"{spec.code_name}.yaml"
    model_card = ROOT / "model_cards" / spec.code_name / "README.md"

    required = [sanitize_report, manifest, pipeline_log, config, model_card]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifact files: {missing}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    stats = json.loads(sanitize_report.read_text(encoding="utf-8"))
    (output_dir / "README.md").write_text(render_readme(stats), encoding="utf-8")
    copy_file(manifest, output_dir / "data/manifest.jsonl")
    copy_file(sanitize_report, output_dir / "reports/sanitize.json")
    copy_file(pipeline_log, output_dir / "pipeline/latest.json")
    copy_file(config, output_dir / "configs/asr.yaml")
    copy_file(model_card, output_dir / "model_card_draft/README.md")

    package = {
        "code_name": spec.code_name,
        "dataset_repo": spec.hf_repo,
        "package_dir": str(output_dir.relative_to(ROOT)),
        "files": sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    (output_dir / "package.json").write_text(
        json.dumps(package, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the HF review artifact package.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/hf_packages" / FIRST_ASR_REVIEW.code_name,
    )
    args = parser.parse_args()
    print(json.dumps(build_package(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from akan_speech.asr.artifacts import FIRST_ASR_REVIEW, AsrArtifactSpec


@dataclass(frozen=True)
class PipelineEvidence:
    label: str
    path: str
    required: bool = True


@dataclass(frozen=True)
class PipelineStage:
    key: str
    title: str
    purpose: str
    gate: str
    evidence: tuple[PipelineEvidence, ...]
    next_action: str


@dataclass(frozen=True)
class PipelineStageStatus:
    stage: PipelineStage
    status: str
    present: tuple[PipelineEvidence, ...]
    missing: tuple[PipelineEvidence, ...]


def first_review_pipeline(spec: AsrArtifactSpec = FIRST_ASR_REVIEW) -> tuple[PipelineStage, ...]:
    code = spec.code_name
    return (
        PipelineStage(
            key="pick",
            title="Pick",
            purpose="Lock the artifact name, corpora, base checkpoint, method, and failure hypothesis.",
            gate="The pass has a config, review packet, and planned model card before any training.",
            evidence=(
                PipelineEvidence(
                    "artifact config",
                    f"configs/asr/{code}.yaml",
                ),
                PipelineEvidence(
                    "planned model card",
                    f"model_cards/{code}/README.md",
                ),
                PipelineEvidence(
                    "local review packet",
                    f"outputs/review_packets/{code}/review_packet.json",
                ),
            ),
            next_action="Run scripts/build_asr_review_packet.py and review the artifact contract.",
        ),
        PipelineStage(
            key="prepare",
            title="Prepare",
            purpose="Build source-specific corpus audits and immutable row references.",
            gate="Waxal and GhanaNLP audits exist before harmonization begins.",
            evidence=(
                PipelineEvidence("Waxal corpus audit", "evals/reports/waxal_aka_corpus_audit.json"),
                PipelineEvidence("GhanaNLP corpus audit", "evals/reports/ghana_nlp_corpus_audit.json"),
                PipelineEvidence("GhanaNLP source manifest", "data/manifests/ghana_nlp_twi.jsonl"),
            ),
            next_action="Run Waxal/GhanaNLP audits and inspect transcript, duration, duplicate, and split reports.",
        ),
        PipelineStage(
            key="sanitize",
            title="Sanitize",
            purpose="Create the harmonized multi-corpus manifest with quarantine reasons.",
            gate="Accepted and quarantined rows are explicit; no silent deletion.",
            evidence=(
                PipelineEvidence("v0.1 harmonized manifest", f"data/manifests/{code}.jsonl"),
                PipelineEvidence("v0.1 sanitization report", f"evals/reports/{code}-sanitize.json"),
            ),
            next_action="Build the Waxal+GhanaNLP harmonized manifest with raw, punctuated, normalized, and expressive fields.",
        ),
        PipelineStage(
            key="train",
            title="Train",
            purpose="Run one bounded replay-mixed training job from the approved manifest.",
            gate="Modal call ID, training config, and checkpoint path are persisted.",
            evidence=(
                PipelineEvidence("training job state", f"outputs/modal_jobs/{code}.json"),
                PipelineEvidence("final checkpoint marker", f"outputs/models/{code}/final"),
            ),
            next_action="Implement and submit one durable Modal v0.1 training executor only after the sanitize gate passes.",
        ),
        PipelineStage(
            key="test",
            title="Test",
            purpose="Decode fixed held-out sets and collect paired predictions.",
            gate="WER/CER by corpus plus repetition and failure taxonomy are recorded.",
            evidence=(
                PipelineEvidence("evaluation report", f"evals/reports/{code}-eval.json"),
                PipelineEvidence("failure taxonomy", f"evals/reports/{code}-failures.md"),
            ),
            next_action="Evaluate on matched held-out rows; do not use arbitrary samples as proof.",
        ),
        PipelineStage(
            key="save",
            title="Save",
            purpose="Persist the exact config, manifest hashes, metrics, and predictions.",
            gate="The result can be reviewed without rerunning training.",
            evidence=(
                PipelineEvidence("result packet", f"outputs/review_packets/{code}/result_packet.json"),
                PipelineEvidence("updated model card", f"model_cards/{code}/README.md"),
            ),
            next_action="Update the packet and model card with metrics, hashes, and limitations.",
        ),
        PipelineStage(
            key="publish",
            title="Publish",
            purpose="Publish a Hugging Face review artifact only after evidence exists.",
            gate="The card states what passed and failed.",
            evidence=(
                PipelineEvidence("HF publish receipt", f"outputs/hf/{code}/publish_receipt.json"),
            ),
            next_action=f"Publish to {spec.hf_repo} only after the test and save gates pass.",
        ),
        PipelineStage(
            key="compare",
            title="Compare",
            purpose="Compare against prior artifacts on matched rows.",
            gate="Comparison uses paired rows and records regressions.",
            evidence=(
                PipelineEvidence("paired comparison report", f"evals/reports/{code}-comparison.md"),
            ),
            next_action="Run paired comparison against Round 2 and the old Waxal model.",
        ),
        PipelineStage(
            key="review",
            title="Review",
            purpose="Capture human corrections, punctuation notes, and qualitative Ghanaian review.",
            gate="Corrections include consent, model version, and audio hash.",
            evidence=(
                PipelineEvidence("review notes", f"evals/reports/{code}-ghanaian-review.md"),
                PipelineEvidence("correction export", f"data/manifests/{code}-corrections.jsonl"),
            ),
            next_action="Listen to failures and export corrected transcripts for the next pass.",
        ),
        PipelineStage(
            key="decide",
            title="Decide",
            purpose="Choose learn, relearn, unlearn, or stop.",
            gate="The next pass has one explicit reason.",
            evidence=(
                PipelineEvidence("decision record", f"docs/decisions/{code}.md"),
            ),
            next_action="Write the decision: learn, relearn, unlearn, or stop.",
        ),
    )


def evaluate_pipeline(root: Path, spec: AsrArtifactSpec = FIRST_ASR_REVIEW) -> list[PipelineStageStatus]:
    statuses: list[PipelineStageStatus] = []
    blocked = False
    for stage in first_review_pipeline(spec):
        present = tuple(item for item in stage.evidence if (root / item.path).exists())
        missing = tuple(item for item in stage.evidence if item.required and not (root / item.path).exists())
        if not missing:
            status = "complete"
        elif blocked:
            status = "blocked"
        else:
            status = "active"
            blocked = True
        statuses.append(PipelineStageStatus(stage, status, present, missing))
    return statuses


def pipeline_next_action(root: Path, spec: AsrArtifactSpec = FIRST_ASR_REVIEW) -> str:
    for status in evaluate_pipeline(root, spec):
        if status.status == "active":
            return status.stage.next_action
    return "All stages have evidence. Review the final decision record."


def render_pipeline_markdown(root: Path, spec: AsrArtifactSpec = FIRST_ASR_REVIEW) -> str:
    lines = [
        f"## Pipeline state: `{spec.code_name}`",
        "",
        f"HF target: `{spec.hf_repo}`",
        "",
        f"Next action: **{pipeline_next_action(root, spec)}**",
        "",
        "| Stage | Status | Gate | Missing evidence |",
        "|---|---|---|---|",
    ]
    for stage_status in evaluate_pipeline(root, spec):
        missing = "<br>".join(f"`{item.path}`" for item in stage_status.missing) or "None"
        lines.append(
            "| "
            f"{stage_status.stage.title} | "
            f"{stage_status.status} | "
            f"{stage_status.stage.gate} | "
            f"{missing} |"
        )
    return "\n".join(lines)

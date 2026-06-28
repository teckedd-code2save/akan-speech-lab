from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from akan_speech.asr.artifacts import FIRST_ASR_REVIEW
from akan_speech.asr.pipeline import evaluate_pipeline, pipeline_next_action


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_command(command: list[str]) -> dict:
    started = time.time()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "runtime_seconds": round(time.time() - started, 3),
        "output": proc.stdout,
        "ok": proc.returncode == 0,
    }


def current_stage_status(key: str) -> str:
    for status in evaluate_pipeline(ROOT):
        if status.stage.key == key:
            return status.status
    return "unknown"


def execute_pipeline(*, stop_before_train: bool = False) -> dict:
    code = FIRST_ASR_REVIEW.code_name
    run_dir = ROOT / "outputs/pipeline_runs" / code
    run_dir.mkdir(parents=True, exist_ok=True)
    events = []

    executors: dict[str, list[str] | None] = {
        "pick": [PYTHON, "scripts/build_asr_review_packet.py"],
        "prepare": None,
        "sanitize": [PYTHON, "scripts/sanitize_asr_v01.py"],
        "train": [PYTHON, "scripts/modal_asr_v01_jobs.py", "train"],
        "test": None,
        "save": None,
        "publish": None,
        "compare": None,
        "review": None,
        "decide": None,
    }

    for key, command in executors.items():
        before = current_stage_status(key)
        event = {
            "stage": key,
            "status_before": before,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if before == "complete":
            event.update({"action": "skip_complete", "status_after": before})
            events.append(event)
            continue

        if before == "blocked":
            event.update(
                {
                    "action": "stop_blocked",
                    "reason": pipeline_next_action(ROOT),
                    "status_after": before,
                }
            )
            events.append(event)
            break

        if key == "prepare":
            event.update(
                {
                    "action": "verify_existing_audits",
                    "reason": "Prepare is satisfied by existing Waxal and GhanaNLP audit artifacts.",
                    "status_after": current_stage_status(key),
                }
            )
            events.append(event)
            continue

        if key == "train" and stop_before_train:
            event.update(
                {
                    "action": "stop_before_train",
                    "reason": "Local run requested; no training job submitted.",
                    "status_after": current_stage_status(key),
                }
            )
            events.append(event)
            break

        if command is None:
            event.update(
                {
                    "action": "stop_unimplemented",
                    "reason": f"Stage '{key}' has no v0.1 executor yet. No fake artifact was created.",
                    "status_after": current_stage_status(key),
                }
            )
            events.append(event)
            break

        result = run_command(command)
        event.update(
            {
                "action": "run_command",
                "result": result,
                "status_after": current_stage_status(key),
            }
        )
        events.append(event)
        if not result["ok"]:
            break

    final = {
        "artifact_code": code,
        "hf_repo": FIRST_ASR_REVIEW.hf_repo,
        "events": events,
        "next_action": pipeline_next_action(ROOT),
        "pipeline": [
            {
                "key": status.stage.key,
                "title": status.stage.title,
                "status": status.status,
                "missing": [item.path for item in status.missing],
            }
            for status in evaluate_pipeline(ROOT)
        ],
    }
    output_path = run_dir / "latest.json"
    output_path.write_text(json.dumps(final, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    final["run_log"] = str(output_path.relative_to(ROOT))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ASR pipeline until the next real gate.")
    parser.add_argument(
        "--stop-before-train",
        action="store_true",
        help="Run local CPU stages only and stop before any training executor.",
    )
    args = parser.parse_args()
    print(json.dumps(execute_pipeline(stop_before_train=args.stop_before_train), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

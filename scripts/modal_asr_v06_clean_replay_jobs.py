from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modal_jobs.asr_v06_clean_replay import APP_NAME, CODE_NAME, V06TrainConfig  # noqa: E402


STATE_PATH = Path("outputs/modal_jobs") / f"{CODE_NAME}.json"


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"app": APP_NAME, "artifact_code": CODE_NAME, "deployment": "unknown", "jobs": {}}
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now()
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str) + "\n")


def deploy() -> dict[str, Any]:
    proc = subprocess.run(
        ["modal", "deploy", "modal_jobs/asr_v06_clean_replay.py"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "Modal deployment failed")
    state = load_state()
    state["deployment"] = "ready"
    state["deployed_at"] = now()
    save_state(state)
    return {"status": "ready", "app": APP_NAME}


def refresh() -> dict[str, Any]:
    state = load_state()
    for job in state.get("jobs", {}).values():
        if job.get("status") not in {"submitted", "running"}:
            continue
        try:
            result = modal.FunctionCall.from_id(job["call_id"]).get(timeout=0)
        except TimeoutError:
            job["status"] = "running"
        except Exception as error:
            job["status"] = "failed"
            job["error"] = f"{type(error).__name__}: {error}"
        else:
            job["status"] = "complete"
            job["completed_at"] = now()
            job["result"] = result
            final_marker = Path("outputs/models") / CODE_NAME / "final"
            final_marker.mkdir(parents=True, exist_ok=True)
            (final_marker / "REMOTE_HF_MODEL.json").write_text(
                json.dumps(result, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
    save_state(state)
    return state


def submit() -> dict[str, Any]:
    state = refresh()
    if state.get("deployment") != "ready":
        raise RuntimeError("Deploy the ASR v0.6 app before submitting training")
    existing = state.get("jobs", {}).get("train")
    if existing and existing.get("status") in {"submitted", "running", "complete"}:
        return existing
    function = modal.Function.from_name(APP_NAME, "train_and_publish")
    function.hydrate()
    config = V06TrainConfig()
    call = function.spawn(asdict(config))
    job = {
        "kind": "train",
        "call_id": call.object_id,
        "status": "submitted",
        "submitted_at": now(),
        "config": asdict(config),
    }
    state.setdefault("jobs", {})["train"] = job
    save_state(state)
    return job


def cancel() -> dict[str, Any]:
    state = refresh()
    job = state.get("jobs", {}).get("train")
    if not job:
        raise ValueError("No train job exists")
    if job.get("status") in {"submitted", "running"}:
        modal.FunctionCall.from_id(job["call_id"]).cancel(terminate_containers=True)
        job["status"] = "cancelled"
        job["cancelled_at"] = now()
        save_state(state)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Durable Modal controller for ASR v0.6 cleaned GhanaNLP + Waxal replay"
    )
    parser.add_argument("action", choices=["deploy", "train", "status", "cancel"])
    args = parser.parse_args()
    if args.action == "deploy":
        result = deploy()
    elif args.action == "train":
        result = submit()
    elif args.action == "cancel":
        result = cancel()
    else:
        result = refresh()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

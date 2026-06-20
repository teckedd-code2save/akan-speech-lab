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

from modal_jobs.asr_round2 import APP_NAME, Round2Config  # noqa: E402


STATE_PATH = Path("outputs/modal_jobs/asr_round2.json")


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"app": APP_NAME, "deployment": "unknown", "jobs": {}}
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now()
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str) + "\n")


def deployed_function(name: str) -> modal.Function:
    function = modal.Function.from_name(APP_NAME, name)
    try:
        function.hydrate()
    except Exception as error:
        raise RuntimeError(
            f"Deployed app {APP_NAME!r} is unavailable. Run the deploy action once before submitting jobs."
        ) from error
    return function


def deploy() -> dict[str, Any]:
    proc = subprocess.run(
        ["modal", "deploy", "modal_jobs/asr_round2.py"],
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
    return {"status": "ready", "app": APP_NAME, "message": proc.stdout.strip()}


def config_for(mode: str) -> Round2Config:
    config = Round2Config()
    if mode == "smoke":
        config.run_name = "smoke-whisper-small-waxal-round2-v1"
        config.max_steps = 2
        config.warmup_steps = 0
        config.eval_steps = 1
        config.save_steps = 1
        config.train_batch_size = 2
        config.eval_batch_size = 2
        config.gradient_accumulation_steps = 1
        config.early_stopping_patience = 3
        config.train_limit = 32
        config.eval_limit = 16
    elif mode == "pilot":
        config.run_name = "whisper-small-waxal-round2-pilot-v1"
        config.max_steps = 200
        config.eval_steps = 200
        config.save_steps = 200
    elif mode != "full":
        raise ValueError("mode must be smoke, pilot, or full")
    return config


def poll_job(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") in {"complete", "failed", "cancelled"}:
        return job
    call = modal.FunctionCall.from_id(job["call_id"])
    try:
        result = call.get(timeout=0)
    except TimeoutError:
        job["status"] = "running"
        return job
    except modal.exception.OutputExpiredError:
        job["status"] = "expired"
        job["error"] = "Modal result expired before it was collected."
        return job
    except Exception as error:
        job["status"] = "failed"
        job["error"] = f"{type(error).__name__}: {error}"
        return job
    job["status"] = "complete"
    job["completed_at"] = now()
    job["result"] = result
    return job


def refresh() -> dict[str, Any]:
    state = load_state()
    for key, job in list(state.get("jobs", {}).items()):
        state["jobs"][key] = poll_job(job)
    save_state(state)
    return state


def submit_preparation() -> dict[str, Any]:
    state = refresh()
    existing = state.get("jobs", {}).get("prepare")
    if existing and existing.get("status") in {"submitted", "running", "complete"}:
        return existing
    config = config_for("full")
    call = deployed_function("prepare_round2_data").spawn(asdict(config))
    job = {
        "kind": "prepare",
        "call_id": call.object_id,
        "status": "submitted",
        "submitted_at": now(),
        "config": asdict(config),
    }
    state.setdefault("jobs", {})["prepare"] = job
    save_state(state)
    return job


def submit_training(mode: str) -> dict[str, Any]:
    state = refresh()
    preparation = state.get("jobs", {}).get("prepare")
    if not preparation or preparation.get("status") != "complete":
        raise RuntimeError("Round 2 CPU preparation must complete before any GPU job is submitted.")
    result = preparation.get("result") or {}
    if not result.get("passed"):
        raise RuntimeError("Round 2 contamination audit did not pass; GPU submission is blocked.")
    key = f"train_{mode}"
    existing = state.get("jobs", {}).get(key)
    if existing and existing.get("status") in {"submitted", "running", "complete"}:
        return existing
    config = config_for(mode)
    call = deployed_function("train_round2").spawn(asdict(config))
    job = {
        "kind": "train",
        "mode": mode,
        "call_id": call.object_id,
        "status": "submitted",
        "submitted_at": now(),
        "config": asdict(config),
    }
    state.setdefault("jobs", {})[key] = job
    save_state(state)
    return job


def cancel(key: str) -> dict[str, Any]:
    state = refresh()
    job = state.get("jobs", {}).get(key)
    if not job:
        raise ValueError(f"Unknown job key: {key}")
    if job.get("status") not in {"submitted", "running"}:
        return job
    modal.FunctionCall.from_id(job["call_id"]).cancel(terminate_containers=True)
    job["status"] = "cancelled"
    job["cancelled_at"] = now()
    save_state(state)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description="Durable Modal controller for ASR Round 2")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("deploy")
    subparsers.add_parser("prepare")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--mode", choices=["smoke", "pilot", "full"], required=True)
    subparsers.add_parser("status")
    cancel_parser = subparsers.add_parser("cancel")
    cancel_parser.add_argument("job_key")
    args = parser.parse_args()

    if args.action == "deploy":
        result = deploy()
    elif args.action == "prepare":
        result = submit_preparation()
    elif args.action == "train":
        result = submit_training(args.mode)
    elif args.action == "cancel":
        result = cancel(args.job_key)
    else:
        result = refresh()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

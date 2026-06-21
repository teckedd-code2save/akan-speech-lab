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

from modal_jobs.asr_round3_lora import APP_NAME, Round3Config  # noqa: E402


STATE_PATH = Path("outputs/modal_jobs/asr_round3.json")
ROUND2_DEV_WER = 0.3213506139154161


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
    save_state(state)
    return state


def deploy() -> dict[str, Any]:
    proc = subprocess.run(
        ["modal", "deploy", "modal_jobs/asr_round3_lora.py"],
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


def config_for(mode: str) -> Round3Config:
    config = Round3Config()
    if mode == "smoke":
        config.run_name = "smoke-whisper-medium-waxal-round3-lora-v1"
        config.max_steps = 2
        config.warmup_steps = 0
        config.eval_steps = 1
        config.save_steps = 1
        config.train_batch_size = 1
        config.eval_batch_size = 1
        config.gradient_accumulation_steps = 1
        config.train_limit = 8
        config.eval_limit = 4
    elif mode == "pilot":
        config.run_name = "pilot-whisper-medium-waxal-round3-lora-v1"
        config.max_steps = 200
        config.warmup_steps = 40
        config.early_stopping_patience = 2
    elif mode != "full":
        raise ValueError("mode must be smoke, pilot, or full")
    return config


def submit(mode: str) -> dict[str, Any]:
    state = refresh()
    if state.get("deployment") != "ready":
        raise RuntimeError("Deploy the stable Round 3 app before submitting jobs")
    if mode == "full":
        pilot = state.get("jobs", {}).get("train_pilot") or {}
        pilot_wer = ((pilot.get("result") or {}).get("final_metrics") or {}).get("final_wer")
        if pilot.get("status") != "complete" or pilot_wer is None:
            raise RuntimeError("A completed pilot with dev WER is required before full training")
        if pilot_wer >= ROUND2_DEV_WER:
            raise RuntimeError(
                f"Pilot WER {pilot_wer:.4f} did not beat Round 2 dev WER "
                f"{ROUND2_DEV_WER:.4f}; full training is blocked"
            )
    key = f"train_{mode}"
    existing = state.get("jobs", {}).get(key)
    if existing and existing.get("status") in {"submitted", "running", "complete"}:
        return existing
    function = modal.Function.from_name(APP_NAME, "train_lora")
    function.hydrate()
    config = config_for(mode)
    call = function.spawn(asdict(config))
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
    if job.get("status") in {"submitted", "running"}:
        modal.FunctionCall.from_id(job["call_id"]).cancel(terminate_containers=True)
        job["status"] = "cancelled"
        job["cancelled_at"] = now()
        save_state(state)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description="Durable Modal controller for ASR Round 3 LoRA")
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("deploy")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--mode", choices=["smoke", "pilot", "full"], required=True)
    subparsers.add_parser("status")
    cancel_parser = subparsers.add_parser("cancel")
    cancel_parser.add_argument("job_key")
    args = parser.parse_args()
    if args.action == "deploy":
        result = deploy()
    elif args.action == "train":
        result = submit(args.mode)
    elif args.action == "cancel":
        result = cancel(args.job_key)
    else:
        result = refresh()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

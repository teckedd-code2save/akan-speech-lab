from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from modal_jobs.tts_train import APP_NAME, TTSConfig


STATE_PATH = Path("outputs/modal_jobs/tts.json")
OVERFIT_TRAINER_STATE = Path("outputs/tts/overfit_trainer_state.json")


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
    overfit = state.get("jobs", {}).get("train_overfit") or {}
    result = overfit.get("result") or {}
    if (
        overfit.get("status") == "complete"
        and not result.get("overfit_gate")
        and OVERFIT_TRAINER_STATE.exists()
    ):
        trainer_state = json.loads(OVERFIT_TRAINER_STATE.read_text())
        losses = [
            {"step": item.get("step"), "loss": item["loss"]}
            for item in trainer_state.get("log_history", [])
            if item.get("loss") is not None
        ]
        converged = bool(losses and losses[-1]["loss"] < losses[0]["loss"] * 0.8)
        result["loss_history"] = losses
        result["overfit_gate"] = {
            "loss_converged": converged,
            "passed": False,
            "status": "needs_human_alignment_review",
            "reason": "Listen for intelligibility, skipped words, and repetition before pilot.",
        }
        overfit["result"] = result
    save_state(state)
    return state


def deploy() -> dict[str, Any]:
    proc = subprocess.run(
        ["modal", "deploy", "modal_jobs/tts_train.py"],
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


def submit_prepare() -> dict[str, Any]:
    state = refresh()
    if state.get("deployment") != "ready":
        raise RuntimeError("Deploy the stable TTS app before submitting preparation")
    existing = state.get("jobs", {}).get("prepare")
    if existing and existing.get("status") in {"submitted", "running", "complete"}:
        return existing
    function = modal.Function.from_name(APP_NAME, "prepare_diagnostic_corpus")
    function.hydrate()
    call = function.spawn()
    job = {
        "kind": "prepare",
        "call_id": call.object_id,
        "status": "submitted",
        "submitted_at": now(),
    }
    state.setdefault("jobs", {})["prepare"] = job
    save_state(state)
    return job


def config_for(mode: str) -> TTSConfig:
    config = TTSConfig()
    if mode == "smoke":
        config.run_name = "smoke-speecht5-farmerline-akosua-v1"
        config.max_steps = 20
        config.warmup_steps = 0
        config.eval_steps = 10
        config.save_steps = 10
        config.train_batch_size = 2
        config.eval_batch_size = 2
        config.gradient_accumulation_steps = 1
        config.train_limit = 64
        config.eval_limit = 16
    elif mode == "overfit":
        config.run_name = "overfit32-speecht5-farmerline-akosua-v1"
        config.max_steps = 300
        config.warmup_steps = 20
        config.eval_steps = 50
        config.save_steps = 50
        config.train_batch_size = 4
        config.eval_batch_size = 4
        config.gradient_accumulation_steps = 1
        config.train_limit = 32
        config.eval_limit = 8
    elif mode == "pilot":
        config.run_name = "pilot-speecht5-farmerline-akosua-v1"
        config.max_steps = 1000
        config.eval_steps = 100
        config.save_steps = 100
    elif mode != "full":
        raise ValueError("mode must be smoke, overfit, pilot, or full")
    return config


def _require_complete(state: dict, key: str) -> dict:
    job = state.get("jobs", {}).get(key) or {}
    if job.get("status") != "complete":
        raise RuntimeError(f"Completed {key} is required")
    return job


def submit_train(mode: str) -> dict[str, Any]:
    state = refresh()
    if state.get("deployment") != "ready":
        raise RuntimeError("Deploy the stable TTS app before submitting training")
    prepare = _require_complete(state, "prepare")
    if not (prepare.get("result") or {}).get("passed"):
        raise RuntimeError("CPU corpus audit did not pass")
    if mode in {"overfit", "pilot", "full"}:
        _require_complete(state, "train_smoke")
    if mode in {"pilot", "full"}:
        overfit = _require_complete(state, "train_overfit")
        if not ((overfit.get("result") or {}).get("overfit_gate") or {}).get("passed"):
            raise RuntimeError("Pilot is blocked until the overfit audio/alignment review passes")
    if mode == "full":
        pilot = _require_complete(state, "train_pilot")
        gate = (pilot.get("result") or {}).get("pilot_gate") or {}
        if not gate.get("passed"):
            raise RuntimeError("Full training is blocked until real pilot listening review passes")
    key = f"train_{mode}"
    existing = state.get("jobs", {}).get(key)
    if existing and existing.get("status") in {"submitted", "running", "complete"}:
        return existing
    config = config_for(mode)
    function = modal.Function.from_name(APP_NAME, "train_speecht5")
    function.hydrate()
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


def cancel(job_key: str) -> dict[str, Any]:
    state = refresh()
    job = state.get("jobs", {}).get(job_key)
    if not job:
        raise ValueError(f"Unknown job key: {job_key}")
    if job.get("status") in {"submitted", "running"}:
        modal.FunctionCall.from_id(job["call_id"]).cancel(terminate_containers=True)
        job["status"] = "cancelled"
        job["cancelled_at"] = now()
        save_state(state)
    return job


def review_overfit(approved: bool, note: str) -> dict[str, Any]:
    state = refresh()
    job = _require_complete(state, "train_overfit")
    result = job.get("result") or {}
    gate = result.get("overfit_gate") or {}
    if not gate.get("loss_converged"):
        raise RuntimeError("Overfit loss did not meet the convergence gate")
    gate.update(
        {
            "passed": bool(approved),
            "status": "approved" if approved else "rejected",
            "reviewed_at": now(),
            "review_note": note.strip(),
        }
    )
    result["overfit_gate"] = gate
    job["result"] = result
    save_state(state)
    return gate


def submit_overfit_diagnostic() -> dict[str, Any]:
    state = refresh()
    _require_complete(state, "train_overfit")
    key = "diagnose_overfit_train"
    existing = state.get("jobs", {}).get(key)
    if existing and existing.get("status") in {"submitted", "running", "complete"}:
        return existing
    function = modal.Function.from_name(APP_NAME, "synthesize_saved_checkpoint")
    function.hydrate()
    call = function.spawn("overfit32-speecht5-farmerline-akosua-v1", "train", 0)
    job = {
        "kind": "diagnostic_synthesis",
        "call_id": call.object_id,
        "status": "submitted",
        "submitted_at": now(),
    }
    state.setdefault("jobs", {})[key] = job
    save_state(state)
    return job


def main() -> None:
    parser = argparse.ArgumentParser(description="Durable gated Modal controller for Akan TTS")
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("deploy")
    commands.add_parser("prepare")
    train = commands.add_parser("train")
    train.add_argument("--mode", choices=["smoke", "overfit", "pilot", "full"], required=True)
    commands.add_parser("status")
    cancel_parser = commands.add_parser("cancel")
    cancel_parser.add_argument("job_key")
    review = commands.add_parser("review-overfit")
    review.add_argument("--approve", action="store_true")
    review.add_argument("--note", required=True)
    commands.add_parser("diagnose-overfit")
    args = parser.parse_args()
    if args.action == "deploy":
        result = deploy()
    elif args.action == "prepare":
        result = submit_prepare()
    elif args.action == "train":
        result = submit_train(args.mode)
    elif args.action == "cancel":
        result = cancel(args.job_key)
    elif args.action == "review-overfit":
        result = review_overfit(args.approve, args.note)
    elif args.action == "diagnose-overfit":
        result = submit_overfit_diagnostic()
    else:
        result = refresh()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

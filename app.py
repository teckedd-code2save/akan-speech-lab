from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

PREVIEW_MANIFEST = ROOT / "evals/samples/waxal_aka_asr_test_preview.jsonl"
LOCAL_MANIFEST = ROOT / "evals/samples/waxal_aka_asr_test_preview_local.jsonl"
REPORTS_DIR = ROOT / "evals/reports"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Quicksand:wght@500;600;700&display=swap');

:root {
  --cream: #FFF6E6; --cream-2: #FFEFD1; --paper: #FFFAF0;
  --line: #2B1E16; --ink: #3B2A1F; --ink-2: #6B4F3F; --ink-soft: #8E7261;
  --shadow: rgba(43,30,22,0.18);
  --peach: #FFB68A; --peach-deep: #FF8E5C;
  --butter: #FFD86B; --butter-deep: #F5B73D;
  --mint: #A8E5C8; --mint-deep: #5FCFA0;
  --sky: #A6D8FF; --sky-deep: #5AB7F2;
  --rose: #FFB3C0; --rose-deep: #F47A92;
  --r-sm: 14px; --r-md: 22px; --r-lg: 36px; --r-pill: 999px;
  --stroke: 3px; --stroke-thick: 4px;
  --plush: 0 6px 0 var(--line), 0 14px 28px var(--shadow);
  --plush-sm: 0 4px 0 var(--line), 0 8px 16px rgba(43,30,22,0.15);
}

body, .gradio-container {
  background: var(--cream) !important;
  color: var(--ink) !important;
  font-family: 'Nunito', system-ui, sans-serif !important;
}

.ak-shell { max-width: 1240px; margin: 0 auto; }
.ak-hero {
  background: var(--paper);
  border: var(--stroke-thick) solid var(--line);
  border-radius: var(--r-lg);
  box-shadow: var(--plush);
  padding: 24px;
  margin: 12px 0 18px;
}
.ak-kicker {
  font-family: 'Quicksand', sans-serif;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--ink-2);
  font-size: 12px;
}
.ak-hero h1 {
  font-size: clamp(34px, 5vw, 60px);
  line-height: .96;
  margin: 6px 0 10px;
  font-weight: 900;
  letter-spacing: 0;
}
.ak-hero p { max-width: 780px; color: var(--ink-2); font-size: 17px; margin: 0; }
.ak-chiprow { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.ak-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 12px; border: 2.5px solid var(--line);
  border-radius: var(--r-pill); box-shadow: 0 2px 0 var(--line);
  background: white; color: var(--ink); font-weight: 800; font-size: 13px;
}
.ak-chip.mint { background: var(--mint); }
.ak-chip.sky { background: var(--sky); }
.ak-chip.butter { background: var(--butter); }
.ak-chip.peach { background: var(--peach); }

.gradio-container .block, .gradio-container .form, .gradio-container .panel {
  border-radius: var(--r-md) !important;
}
.gradio-container .block:not(.gradio-html) {
  border: var(--stroke) solid var(--line) !important;
  box-shadow: var(--plush-sm) !important;
  background: white !important;
}
.gradio-container button {
  border: var(--stroke) solid var(--line) !important;
  border-radius: var(--r-pill) !important;
  box-shadow: 0 4px 0 var(--line) !important;
  font-weight: 900 !important;
  color: var(--ink) !important;
}
.gradio-container button.primary { background: var(--peach) !important; }
.gradio-container button.secondary { background: var(--butter) !important; }
.gradio-container button:hover { transform: translateY(-2px); }
.gradio-container button:active { transform: translateY(2px); box-shadow: 0 1px 0 var(--line) !important; }

.gradio-container input, .gradio-container textarea, .gradio-container select {
  border: 2.5px solid var(--line) !important;
  border-radius: var(--r-sm) !important;
  background: var(--cream-2) !important;
  color: var(--ink) !important;
}
.gradio-container label, .gradio-container .label-wrap span {
  color: var(--ink) !important;
  font-weight: 800 !important;
}
.ak-card {
  background: white;
  border: var(--stroke) solid var(--line);
  border-radius: var(--r-md);
  box-shadow: var(--plush-sm);
  padding: 16px;
}
.ak-status { font-weight: 900; font-size: 18px; margin-bottom: 6px; }
.ak-muted { color: var(--ink-2); }
"""


def run_command(args: list[str], timeout: int = 1800) -> tuple[bool, str]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=ENV,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    output = "\n".join(part for part in [proc.stdout.strip(), proc.stderr.strip()] if part)
    return proc.returncode == 0, output or "(no output)"


def status_html(ok: bool, title: str, body: str = "") -> str:
    color = "mint" if ok else "rose"
    return f"""
    <div class="ak-card">
      <div class="ak-status">{'Ready' if ok else 'Needs attention'} · {title}</div>
      <div class="ak-muted">{body}</div>
      <div class="ak-chiprow"><span class="ak-chip {color}">{'pass' if ok else 'check logs'}</span></div>
    </div>
    """


def read_jsonl_preview(path: Path, limit: int = 3) -> str:
    if not path.exists():
        return "No manifest yet."
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[:limit]:
        if line.strip():
            rows.append(json.loads(line))
    lines = []
    for idx, row in enumerate(rows, start=1):
        lines.append(f"### Sample {idx}")
        lines.append(f"- Speaker: `{row.get('speaker_id')}`")
        lines.append(f"- Split: `{row.get('split')}`")
        lines.append(f"- Audio: `{row.get('audio_path')}`")
        lines.append("")
        lines.append(row.get("text") or row.get("normalized_text") or "")
        lines.append("")
    return "\n".join(lines) if lines else "Manifest exists, but no readable rows."


def build_preview(split: str, limit: int):
    ok, logs = run_command(
        [
            PYTHON,
            "scripts/build_waxal_viewer_manifest.py",
            "--config",
            "aka_asr",
            "--split",
            split,
            "--limit",
            str(int(limit)),
            "--output",
            str(PREVIEW_MANIFEST.relative_to(ROOT)),
        ],
        timeout=120,
    )
    return status_html(ok, "Preview manifest", str(PREVIEW_MANIFEST.relative_to(ROOT))), logs, read_jsonl_preview(PREVIEW_MANIFEST)


def materialize_audio(limit: int):
    if not PREVIEW_MANIFEST.exists():
        return status_html(False, "Audio materialization", "Build a preview manifest first."), "", None, "No preview manifest."
    ok, logs = run_command(
        [
            PYTHON,
            "scripts/materialize_manifest_audio.py",
            "--manifest",
            str(PREVIEW_MANIFEST.relative_to(ROOT)),
            "--output-manifest",
            str(LOCAL_MANIFEST.relative_to(ROOT)),
            "--limit",
            str(int(limit)),
        ],
        timeout=180,
    )
    first_audio = None
    if LOCAL_MANIFEST.exists():
        first = json.loads(LOCAL_MANIFEST.read_text(encoding="utf-8").splitlines()[0])
        candidate = ROOT / first["audio_path"]
        if candidate.exists():
            first_audio = str(candidate)
    return status_html(ok, "Local audio pack", str(LOCAL_MANIFEST.relative_to(ROOT))), logs, first_audio, read_jsonl_preview(LOCAL_MANIFEST)


def prepare_sample_pack(split: str, limit: int):
    manifest_status, manifest_logs, preview = build_preview(split, limit)
    audio_status, audio_logs, audio, local_preview = materialize_audio(limit)
    return (
        manifest_status + audio_status,
        manifest_logs + "\n\n" + audio_logs,
        audio,
        local_preview if LOCAL_MANIFEST.exists() else preview,
    )


def dry_run_eval(model_id: str, limit: int, language: str):
    manifest = LOCAL_MANIFEST if LOCAL_MANIFEST.exists() else PREVIEW_MANIFEST
    if not manifest.exists():
        return status_html(False, "Eval dry-run", "Prepare a sample pack first."), ""
    ok, logs = run_command(
        [
            PYTHON,
            "scripts/eval_asr_manifest.py",
            "--model-id",
            model_id,
            "--manifest",
            str(manifest.relative_to(ROOT)),
            "--limit",
            str(int(limit)),
            "--language",
            language,
            "--task",
            "transcribe",
            "--dry-run",
        ],
        timeout=120,
    )
    return status_html(ok, "Eval dry-run", f"Manifest: {manifest.relative_to(ROOT)}"), logs


def cache_model(model_id: str):
    ok, logs = run_command([PYTHON, "scripts/cache_hf_model.py", "--model-id", model_id], timeout=1800)
    return status_html(ok, "Model cache", model_id), logs


def run_eval(model_id: str, limit: int, language: str):
    manifest = LOCAL_MANIFEST if LOCAL_MANIFEST.exists() else PREVIEW_MANIFEST
    if not manifest.exists():
        return status_html(False, "ASR eval", "Prepare a sample pack first."), "", "", None
    safe_name = model_id.replace("/", "__").replace(":", "_")
    json_path = REPORTS_DIR / f"{safe_name}_eval_{int(limit)}row.json"
    csv_path = REPORTS_DIR / f"{safe_name}_eval_{int(limit)}row.csv"
    md_path = REPORTS_DIR / f"{safe_name}_eval_{int(limit)}row.md"
    ok, logs = run_command(
        [
            PYTHON,
            "scripts/eval_asr_manifest.py",
            "--model-id",
            model_id,
            "--manifest",
            str(manifest.relative_to(ROOT)),
            "--limit",
            str(int(limit)),
            "--language",
            language,
            "--task",
            "transcribe",
            "--output-json",
            str(json_path.relative_to(ROOT)),
            "--output-csv",
            str(csv_path.relative_to(ROOT)),
        ],
        timeout=3600,
    )
    report_md = ""
    report_file = None
    if ok and json_path.exists():
        ok_render, render_logs = run_command(
            [
                PYTHON,
                "scripts/render_asr_report.py",
                "--input-json",
                str(json_path.relative_to(ROOT)),
                "--output-md",
                str(md_path.relative_to(ROOT)),
            ],
            timeout=120,
        )
        logs = logs + "\n\n" + render_logs
        if ok_render and md_path.exists():
            report_md = md_path.read_text(encoding="utf-8")
            report_file = str(md_path)
    return status_html(ok, "ASR eval", model_id), logs, report_md or "No report rendered.", report_file


def report_choices() -> gr.Dropdown:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(REPORTS_DIR.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
    return gr.Dropdown(choices=[str(path.relative_to(ROOT)) for path in files])


def load_report(path: str):
    if not path:
        return "Choose a report."
    report = ROOT / path
    if not report.exists():
        return "Report not found."
    return report.read_text(encoding="utf-8")


def app_summary() -> str:
    has_pack = LOCAL_MANIFEST.exists()
    report_count = len(list(REPORTS_DIR.glob("*.md"))) if REPORTS_DIR.exists() else 0
    return f"""
    <div class="ak-card">
      <div class="ak-status">Lab state</div>
      <div class="ak-chiprow">
        <span class="ak-chip {'mint' if has_pack else 'butter'}">sample pack {'ready' if has_pack else 'not built'}</span>
        <span class="ak-chip sky">{report_count} rendered report(s)</span>
        <span class="ak-chip peach">Modal paused</span>
      </div>
    </div>
    """


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Akan Speech Lab") as demo:
        with gr.Column(elem_classes=["ak-shell"]):
            gr.HTML(
                """
                <section class="ak-hero">
                  <div class="ak-kicker">Akan Speech Lab</div>
                  <h1>Small, inspectable ASR experiments.</h1>
                  <p>Prepare Waxal Akan samples, materialize audio, run baseline ASR checks, and read reports without touching the terminal.</p>
                  <div class="ak-chiprow">
                    <span class="ak-chip mint">Waxal aka_asr</span>
                    <span class="ak-chip sky">WER / CER</span>
                    <span class="ak-chip butter">Local first</span>
                    <span class="ak-chip peach">Modal later</span>
                  </div>
                </section>
                """
            )
            state = gr.HTML(app_summary())

            with gr.Tabs():
                with gr.Tab("Sample Pack"):
                    with gr.Row():
                        split = gr.Dropdown(["train", "validation", "test"], value="test", label="Waxal split")
                        limit = gr.Slider(1, 10, value=3, step=1, label="Samples")
                    prepare_btn = gr.Button("Build sample pack", variant="primary")
                    sample_status = gr.HTML()
                    sample_logs = gr.Textbox(label="Run log", lines=8, interactive=False)
                    sample_audio = gr.Audio(label="First sample audio", type="filepath")
                    sample_preview = gr.Markdown(label="Manifest preview")
                    prepare_btn.click(
                        prepare_sample_pack,
                        inputs=[split, limit],
                        outputs=[sample_status, sample_logs, sample_audio, sample_preview],
                    ).then(app_summary, outputs=state)

                with gr.Tab("Evaluate"):
                    model_id = gr.Textbox(
                        value="teckedd/whisper_small-waxal_akan-asr-v1",
                        label="Model ID",
                        placeholder="openai/whisper-tiny or teckedd/whisper_small-waxal_akan-asr-v1",
                    )
                    with gr.Row():
                        eval_limit = gr.Slider(1, 5, value=1, step=1, label="Rows to evaluate")
                        language = gr.Dropdown(["yoruba", "english", ""], value="yoruba", label="Whisper language hint")
                    with gr.Row():
                        dry_btn = gr.Button("Dry-run eval", variant="secondary")
                        cache_btn = gr.Button("Cache model", variant="secondary")
                        eval_btn = gr.Button("Run ASR eval", variant="primary")
                    eval_status = gr.HTML()
                    eval_logs = gr.Textbox(label="Run log", lines=10, interactive=False)
                    eval_report = gr.Markdown(label="Rendered report")
                    eval_report_file = gr.File(label="Report file", interactive=False)
                    dry_btn.click(dry_run_eval, inputs=[model_id, eval_limit, language], outputs=[eval_status, eval_logs])
                    cache_btn.click(cache_model, inputs=model_id, outputs=[eval_status, eval_logs])
                    eval_btn.click(
                        run_eval,
                        inputs=[model_id, eval_limit, language],
                        outputs=[eval_status, eval_logs, eval_report, eval_report_file],
                    ).then(app_summary, outputs=state)

                with gr.Tab("Reports"):
                    refresh_reports = gr.Button("Refresh reports", variant="secondary")
                    report_picker = report_choices()
                    report_view = gr.Markdown("Choose a report.")
                    refresh_reports.click(report_choices, outputs=report_picker)
                    report_picker.change(load_report, inputs=report_picker, outputs=report_view)

                with gr.Tab("Notes"):
                    gr.Markdown(
                        """
                        ## What I picked up from MedKit

                        - Use a warm operator-console UI, not a raw script launcher.
                        - Keep voice/model work inspectable: input audio, reference, prediction, normalized text, WER/CER.
                        - Separate cheap local checks from GPU/Modal jobs.
                        - Treat the old Waxal model as a baseline floor, not a target.
                        - Future Modal jobs should run only after the sample pack and eval harness are proven locally.
                        """
                    )
                    gr.File(value=str(ROOT / "docs/medkit_ui_lessons.md"), label="Study notes", interactive=False)

    return demo


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1", server_port=7862, css=CSS)

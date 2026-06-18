from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

PREVIEW_MANIFEST = ROOT / "evals/samples/waxal_aka_asr_test_preview.jsonl"
LOCAL_MANIFEST = ROOT / "evals/samples/waxal_aka_asr_test_preview_local.jsonl"
REPORTS_DIR = ROOT / "evals/reports"
TTS_COMPARISON_DIR = ROOT / "outputs/tts_comparisons"
TTS_MODEL_ID = "facebook/mms-tts-aka"
_TTS_MODEL = None
_TTS_TOKENIZER = None


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Quicksand:wght@500;600;700&display=swap');

:root {
  --cream: #FFF6E6; --cream-2: #FFEFD1; --paper: #FFFDF8; --surface: #FFFFFF;
  --card: #FFEFD1; --card-2: #FFF7E2;
  --line: #2B1E16; --ink: #3B2A1F; --ink-2: #6B4F3F; --ink-soft: #8E7261;
  --hair: rgba(43,30,22,.14); --hair-strong: rgba(43,30,22,.22);
  --shadow: rgba(43,30,22,0.10);
  --peach: #FFB68A; --peach-deep: #FF8E5C;
  --butter: #FFD86B; --butter-deep: #F5B73D;
  --mint: #A8E5C8; --mint-deep: #5FCFA0;
  --sky: #A6D8FF; --sky-deep: #5AB7F2;
  --rose: #FFB3C0; --rose-deep: #F47A92;
  --r-sm: 12px; --r-md: 18px; --r-lg: 26px; --r-pill: 999px;
  --card-shadow: 0 1px 2px rgba(43,30,22,.06), 0 10px 24px rgba(43,30,22,.07);
  --btn-lift: 0 2px 0 rgba(43,30,22,.55);
}

/* ---- Kill Gradio dark-mode bleed: drive its own theme vars to light ---- */
/* Without this, File/Audio render as black boxes and blocks get heavy dark frames. */
.gradio-container,
.gradio-container.dark,
.dark {
  color-scheme: light;
  --body-background-fill: var(--cream);
  --background-fill-primary: var(--cream);
  --background-fill-secondary: var(--paper);
  --block-background-fill: var(--surface);
  --block-border-color: var(--hair);
  --block-border-width: 1px;
  --block-label-background-fill: var(--surface);
  --block-label-text-color: var(--ink-2);
  --block-title-text-color: var(--ink);
  --block-info-text-color: var(--ink-2);
  --border-color-primary: var(--hair);
  --border-color-accent: var(--peach-deep);
  --input-background-fill: var(--paper);
  --input-background-fill-focus: var(--surface);
  --input-border-color: var(--hair-strong);
  --input-border-color-focus: var(--peach-deep);
  --input-placeholder-color: var(--ink-soft);
  --body-text-color: var(--ink);
  --body-text-color-subdued: var(--ink-2);
  --neutral-400: var(--ink-soft);
  --neutral-500: var(--ink-2);
  --color-accent: var(--peach-deep);
  --color-accent-soft: var(--cream-2);
  --slider-color: var(--peach-deep);
  --panel-background-fill: var(--surface);
  --table-even-background-fill: var(--surface);
  --table-odd-background-fill: var(--paper);
  --table-row-focus: var(--cream-2);
  --table-border-color: var(--hair);
  --table-text-color: var(--ink);
  --link-text-color: var(--peach-deep);
  --code-background-fill: var(--cream-2);
}

html, body, gradio-app, .gradio-container {
  background: var(--cream) !important;
  color: var(--ink) !important;
  font-family: 'Nunito', system-ui, sans-serif !important;
}

.ak-shell { max-width: 1180px; margin: 0 auto; padding: 8px 4px 32px; }

/* ---- Consistent surface system: hero + status share one language ---- */
.ak-hero, .ak-card {
  background: var(--card);
  border: 2px solid var(--line);
  box-shadow: 0 4px 0 rgba(43,30,22,.7), 0 14px 26px rgba(43,30,22,.10);
  color: var(--ink) !important;
}
.ak-hero * , .ak-card * { color: inherit; }
.ak-hero {
  border-radius: var(--r-lg);
  padding: 30px 32px;
  margin: 10px 0 14px;
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
  font-size: clamp(30px, 4.4vw, 52px);
  line-height: 1.02;
  margin: 8px 0 12px;
  font-weight: 900;
  letter-spacing: -0.01em;
  color: var(--ink) !important;
}
.ak-hero p { max-width: 780px; color: var(--ink-2); font-size: 18px; line-height: 1.55; margin: 0; }

.ak-chiprow { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
.ak-chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 13px; border: 2px solid var(--line);
  border-radius: var(--r-pill); box-shadow: 0 2px 0 var(--line);
  background: #FFFFFF; color: var(--ink); font-weight: 800; font-size: 13px;
}
.ak-chip.mint { background: var(--mint); }
.ak-chip.sky { background: var(--sky); }
.ak-chip.butter { background: var(--butter); }
.ak-chip.peach { background: var(--peach); }
.ak-chip.rose { background: var(--rose); }

.ak-card { border-radius: var(--r-md); padding: 18px 20px; margin: 4px 0 10px; }
.ak-status { font-weight: 900; font-size: 17px; margin-bottom: 10px; color: var(--ink); }
.ak-muted { color: var(--ink-2); }

/* ---- Headings / body text inherit ink ---- */
.gradio-container h1, .gradio-container h2, .gradio-container h3,
.gradio-container h4, .gradio-container p, .gradio-container li {
  color: var(--ink);
}

/* Bare HTML/Markdown wrappers carry no chrome */
.gradio-container .gradio-html,
.gradio-container .html-container,
.gradio-container .prose {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}
/* HTML blocks draw their own .ak-card (or are empty placeholders): no outer frame */
.gradio-container .block:has(.html-container) {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
}

/* ---- Markdown / prose: bold brand headings, readable body, on-brand code ---- */
.gradio-container .md,
.gradio-container .prose {
  color: var(--ink) !important;
  font-size: 15px !important;
  line-height: 1.62 !important;
}
.gradio-container .md :is(h1, h2, h3, h4),
.gradio-container .prose :is(h1, h2, h3, h4) {
  color: var(--ink) !important;
  font-weight: 900 !important;
  letter-spacing: -0.01em !important;
}
.gradio-container .md h2, .gradio-container .prose h2 { font-size: 22px !important; margin: 8px 0 10px !important; }
.gradio-container .md h3, .gradio-container .prose h3 { font-size: 18px !important; margin: 16px 0 6px !important; }
.gradio-container .md :is(p, li),
.gradio-container .prose :is(p, li) {
  color: var(--ink) !important;
  font-weight: 500 !important;
}
.gradio-container .md strong, .gradio-container .prose strong {
  color: var(--ink) !important;
  font-weight: 800 !important;
}
.gradio-container .md code, .gradio-container .prose code {
  background: var(--cream-2) !important;
  color: var(--ink) !important;
  border: 1px solid var(--hair-strong) !important;
  border-radius: 6px !important;
  padding: 1px 6px !important;
  font-weight: 700 !important;
  font-size: 13px !important;
}
.gradio-container .md a, .gradio-container .prose a {
  color: var(--peach-deep) !important;
  font-weight: 700 !important;
}

/* ---- File component row: readable, not a black bar ---- */
.gradio-container .file-preview,
.gradio-container .file-preview *,
.gradio-container [data-testid="file"] td,
.gradio-container [data-testid="file"] tr {
  background: var(--paper) !important;
  color: var(--ink) !important;
}
.gradio-container .file-preview a,
.gradio-container .download-link { color: var(--peach-deep) !important; font-weight: 700 !important; }

/* ---- One flat surface per block (no nested double-frame) ---- */
.gradio-container .block {
  border-radius: var(--r-md) !important;
  border: 1px solid var(--hair) !important;
  background: var(--surface) !important;
  box-shadow: none !important;
  padding: 14px 16px !important;
}
.gradio-container .form { border: 0 !important; background: transparent !important; gap: 14px !important; }
.gradio-container .gap { gap: 14px !important; }

/* Block labels: quiet, readable */
.gradio-container .block-title,
.gradio-container label > span,
.gradio-container .label-wrap span {
  color: var(--ink-2) !important;
  font-weight: 700 !important;
  font-size: 13px !important;
}

/* ---- Inputs: light field, subtle border, no peach pill ---- */
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container select,
.gradio-container .gr-input,
.gradio-container .secondary-wrap {
  border: 1px solid var(--hair-strong) !important;
  border-radius: var(--r-sm) !important;
  background: var(--paper) !important;
  color: var(--ink) !important;
  box-shadow: none !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: var(--peach-deep) !important;
  background: var(--surface) !important;
  outline: none !important;
}
.gradio-container textarea { line-height: 1.5 !important; }

/* ---- Buttons: keep the playful plush lift, lighter + consistent ---- */
.gradio-container button.primary,
.gradio-container button.secondary,
.gradio-container button.lg,
.gradio-container button.sm {
  border: 1px solid var(--line) !important;
  border-radius: var(--r-pill) !important;
  box-shadow: var(--btn-lift) !important;
  font-weight: 800 !important;
  color: var(--ink) !important;
  transition: transform .08s ease, box-shadow .08s ease;
}
.gradio-container button.primary { background: var(--peach) !important; }
.gradio-container button.secondary { background: var(--butter) !important; }
.gradio-container button.primary:hover,
.gradio-container button.secondary:hover { transform: translateY(-1px); box-shadow: 0 3px 0 rgba(43,30,22,.55) !important; }
.gradio-container button.primary:active,
.gradio-container button.secondary:active { transform: translateY(1px); box-shadow: 0 1px 0 rgba(43,30,22,.55) !important; }

/* ---- Tabs: clean underline ---- */
.gradio-container .tab-nav {
  border-bottom: 1px solid var(--hair-strong) !important;
  gap: 4px !important;
  margin-bottom: 6px !important;
}
.gradio-container button[role="tab"] {
  background: transparent !important;
  box-shadow: none !important;
  border: 0 !important;
  border-bottom: 3px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 16px !important;
  font-weight: 800 !important;
  color: var(--ink-2) !important;
}
.gradio-container button[role="tab"]:hover { color: var(--ink) !important; }
.gradio-container button[role="tab"][aria-selected="true"] {
  color: var(--ink) !important;
  border-bottom: 3px solid var(--peach-deep) !important;
}

/* ---- Audio / File drop zones: light, not black ---- */
.gradio-container .audio-container,
.gradio-container .file-preview,
.gradio-container [data-testid="audio"],
.gradio-container .empty,
.gradio-container .wrap.default {
  background: var(--paper) !important;
  color: var(--ink-2) !important;
}

/* ---- Slider min/max + number box: readable ---- */
.gradio-container .min_value,
.gradio-container .max_value,
.gradio-container .head span { color: var(--ink-2) !important; }

/* ---- Footer: quiet utility links, not primary buttons ---- */
footer { color: var(--ink-2) !important; }
footer button,
footer a,
.gradio-container .reset-button {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  color: var(--ink-2) !important;
  font-weight: 700 !important;
  transform: none !important;
}
footer button:hover, footer a:hover { color: var(--peach-deep) !important; }
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


def read_manifest_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_choices(rows: list[dict]) -> list[tuple[str, str]]:
    choices = []
    for idx, row in enumerate(rows):
        text = row.get("text") or row.get("normalized_text") or "(empty transcript)"
        short_text = text if len(text) <= 64 else text[:61] + "..."
        choices.append((f"{idx + 1}. {row.get('speaker_id') or 'unknown speaker'} · {short_text}", str(idx)))
    return choices


def sample_table(rows: list[dict]) -> list[list[str]]:
    return [
        [
            str(idx + 1),
            str(row.get("speaker_id") or "unknown"),
            str(row.get("language") or "aka"),
            str(row.get("split") or ""),
            str(row.get("text") or ""),
            str(row.get("normalized_text") or ""),
        ]
        for idx, row in enumerate(rows)
    ]


def selected_sample(index: str | int):
    rows = read_manifest_rows(LOCAL_MANIFEST)
    if not rows:
        return None, "_Build a sample pack first._"
    try:
        row_index = max(0, min(int(index), len(rows) - 1))
    except (TypeError, ValueError):
        row_index = 0
    row = rows[row_index]
    audio_path = ROOT / str(row.get("audio_path") or "")
    audio = str(audio_path) if audio_path.exists() else None
    details = f"""
### Sample {row_index + 1} of {len(rows)}

**Reference**

{row.get('text') or '(empty)'}

**Normalized for WER**

{row.get('normalized_text') or '(empty)'}

`speaker: {row.get('speaker_id') or 'unknown'}` · `language: {row.get('language') or 'aka'}` · `split: {row.get('split') or 'unknown'}`
"""
    return audio, details


def pack_quality(rows: list[dict]) -> str:
    if not rows:
        return "_No sample pack built._"
    speakers = Counter(str(row.get("speaker_id") or "unknown") for row in rows)
    changed = sum(
        1 for row in rows if (row.get("text") or "").strip() != (row.get("normalized_text") or "").strip()
    )
    return f"""
### Pack checks

- **{len(rows)}** requested and materialized
- **{len(speakers)}** unique speakers
- **{changed}** transcripts changed by conservative Akan normalization
- **{sum(1 for row in rows if not row.get('text'))}** empty references
- **{sum(1 for row in rows if not row.get('audio_path'))}** missing audio paths

This preview audits examples only. Speaker-safe split generation and duration filtering belong to the full training-manifest step.
"""


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
    rows = read_manifest_rows(LOCAL_MANIFEST) if LOCAL_MANIFEST.exists() else []
    choices = sample_choices(rows)
    first_audio, first_details = selected_sample("0") if rows else (None, "_No sample selected._")
    return (
        manifest_status + audio_status,
        manifest_logs + "\n\n" + audio_logs,
        gr.Dropdown(choices=choices, value="0" if choices else None),
        first_audio,
        first_details,
        sample_table(rows),
        pack_quality(rows),
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


def eval_json_choices() -> gr.Dropdown:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for path in REPORTS_DIR.glob("*_eval_*row.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("predictions"):
            files.append(path)
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    choices = [(path.stem.replace("__", "/"), str(path.relative_to(ROOT))) for path in files]
    return gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def resolve_eval_audio(payload: dict, row: dict) -> str | None:
    candidate = row.get("audio_path")
    if not candidate:
        manifest_path = ROOT / str(payload.get("report", {}).get("manifest") or "")
        manifest_rows = read_manifest_rows(manifest_path)
        index = int(row.get("idx", 0))
        if index < len(manifest_rows):
            candidate = manifest_rows[index].get("audio_path")
    if not candidate:
        return None
    path = Path(str(candidate))
    if not path.is_absolute():
        path = ROOT / path
    return str(path) if path.exists() else None


def load_eval_row(report_path: str, row_index: str | int):
    if not report_path:
        return None, "", "", "_Choose an evaluation report._"
    path = ROOT / report_path
    if not path.exists():
        return None, "", "", "_Evaluation report not found._"
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("predictions", [])
    if not rows:
        return None, "", "", "_This report has no saved predictions._"
    try:
        index = max(0, min(int(row_index), len(rows) - 1))
    except (TypeError, ValueError):
        index = 0
    row = rows[index]
    if "wer" not in row:
        from akan_speech.eval.wer import speech_error_rates

        row.update(speech_error_rates([row.get("reference", "")], [row.get("prediction", "")]))
    metrics = (
        f"**Row {index + 1} of {len(rows)} · speaker {row.get('speaker_id') or 'unknown'}**  \n"
        f"WER **{float(row.get('wer', 0)) * 100:.2f}%** · CER **{float(row.get('cer', 0)) * 100:.2f}%** · "
        f"{row.get('substitutions', 0)} substitutions · {row.get('deletions', 0)} deletions · "
        f"{row.get('insertions', 0)} insertions"
    )
    return (
        resolve_eval_audio(payload, row),
        row.get("reference", ""),
        row.get("prediction", ""),
        metrics,
    )


def load_eval_inspector(report_path: str):
    if not report_path or not (ROOT / report_path).exists():
        return gr.Dropdown(choices=[], value=None), None, "", "", "_Choose an evaluation report._"
    payload = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
    rows = payload.get("predictions", [])
    choices = [
        (f"Row {idx + 1} · speaker {row.get('speaker_id') or 'unknown'}", str(idx))
        for idx, row in enumerate(rows)
    ]
    audio, reference, prediction, metrics = load_eval_row(report_path, "0")
    return gr.Dropdown(choices=choices, value="0" if choices else None), audio, reference, prediction, metrics


def _load_akan_tts():
    global _TTS_MODEL, _TTS_TOKENIZER
    if _TTS_MODEL is None or _TTS_TOKENIZER is None:
        from transformers import AutoTokenizer, VitsModel

        _TTS_TOKENIZER = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
        _TTS_MODEL = VitsModel.from_pretrained(TTS_MODEL_ID)
        _TTS_MODEL.eval()
    return _TTS_MODEL, _TTS_TOKENIZER


def synthesize_akan_text(text: str, role: str) -> str:
    if not (text or "").strip():
        raise ValueError(f"The {role} text is empty.")
    import soundfile as sf
    import torch

    model, tokenizer = _load_akan_tts()
    cache_key = hashlib.sha256(f"{TTS_MODEL_ID}\n{text}".encode("utf-8")).hexdigest()[:20]
    TTS_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TTS_COMPARISON_DIR / f"{role}_{cache_key}.wav"
    if not output_path.exists():
        inputs = tokenizer(text, return_tensors="pt")
        torch.manual_seed(42)
        with torch.no_grad():
            waveform = model(**inputs).waveform.squeeze().cpu().float().numpy()
        sf.write(output_path, waveform, model.config.sampling_rate)
    return str(output_path)


def synthesize_eval_pair(reference: str, prediction: str):
    try:
        reference_audio = synthesize_akan_text(reference, "reference")
        prediction_audio = synthesize_akan_text(prediction, "prediction")
    except Exception as exc:
        return None, None, status_html(False, "Akan TTS comparison", str(exc))
    return (
        reference_audio,
        prediction_audio,
        status_html(
            True,
            "Akan TTS comparison",
            f"Both WAVs generated with {TTS_MODEL_ID}. Compare pronunciation differences, not speaker identity.",
        ),
    )


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


def training_plan() -> str:
    manifest_ready = LOCAL_MANIFEST.exists() and bool(read_manifest_rows(LOCAL_MANIFEST))
    return f"""
<div class="ak-card">
  <div class="ak-status">Training is gated</div>
  <div class="ak-muted">The preview pack is not training data. Before Modal can launch, we still need a full Waxal manifest, duration measurements, filtering, and speaker-overlap checks.</div>
  <div class="ak-chiprow">
    <span class="ak-chip {'mint' if manifest_ready else 'butter'}">preview {'ready' if manifest_ready else 'needed'}</span>
    <span class="ak-chip butter">full manifest pending</span>
    <span class="ak-chip butter">speaker-safe split pending</span>
    <span class="ak-chip peach">Modal launch locked</span>
  </div>
</div>
"""


def comparison_board() -> str:
    reports = []
    for path in REPORTS_DIR.glob("*.json") if REPORTS_DIR.exists() else []:
        try:
            report = json.loads(path.read_text(encoding="utf-8")).get("report", {})
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("model_id") and report.get("wer") is not None:
            reports.append(report)
    reports.sort(key=lambda item: float(item.get("wer", 99)))
    lines = [
        "### Promotion gate",
        "",
        "The current floor is **34.28% WER**. A candidate is promoted only below **30.86% WER** (10% relative improvement), with qualitative review of Akan spelling and code-switching.",
        "",
        "| Model | Rows | WER | CER | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for report in reports:
        wer_percent = float(report["wer"]) * 100
        status = "candidate" if wer_percent < 30.856 else "below gate"
        lines.append(
            f"| `{report['model_id']}` | {report.get('rows', 0)} | {wer_percent:.2f}% | {float(report.get('cer', 0)) * 100:.2f}% | {status} |"
        )
    if not reports:
        lines.append("| No local evaluations yet | - | - | - | run the baseline first |")
    lines.extend(
        [
            "",
            "Full-dataset reports will also break results down by speaker and utterance duration. Preview evaluations are deliberately not promotion evidence.",
        ]
    )
    return "\n".join(lines)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Akan Speech Lab") as demo:
        with gr.Column(elem_classes=["ak-shell"]):
            gr.HTML(
                """
                <section class="ak-hero">
                  <div class="ak-kicker">Akan Speech Lab</div>
                  <h1>Beat the 34.28% Akan ASR baseline.</h1>
                  <p>Audit the data, reproduce the existing Whisper result, train stronger configurations, and promote only a measured improvement.</p>
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

            gr.HTML(
                """
                <div class="ak-chiprow">
                  <span class="ak-chip mint">1 · Prepare data</span>
                  <span class="ak-chip sky">2 · Reproduce baseline</span>
                  <span class="ak-chip butter">3 · Train candidates</span>
                  <span class="ak-chip peach">4 · Compare & promote</span>
                </div>
                """
            )

            with gr.Tabs():
                with gr.Tab("1 · Prepare"):
                    gr.Markdown("## Inspect one coherent sample pack\nEvery row below is one audio/reference pair. Choose a row to hear exactly the transcript shown beside it.")
                    with gr.Row():
                        split = gr.Dropdown(["train", "validation", "test"], value="test", label="Waxal split")
                        limit = gr.Slider(1, 10, value=5, step=1, label="Samples")
                    prepare_btn = gr.Button("Build sample pack", variant="primary")
                    sample_status = gr.HTML()
                    sample_picker = gr.Dropdown(label="Sample to inspect", choices=[])
                    with gr.Row():
                        sample_audio = gr.Audio(label="Selected sample audio", type="filepath", interactive=False)
                        sample_details = gr.Markdown("_Build a sample pack, then select a row._")
                    sample_rows = gr.Dataframe(
                        headers=["#", "Speaker", "Language", "Split", "Reference", "Normalized"],
                        datatype=["str", "str", "str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        wrap=True,
                    )
                    sample_quality = gr.Markdown("_Pack checks appear here after preparation._")
                    with gr.Accordion("Technical log", open=False):
                        sample_logs = gr.Textbox(label="Run log", lines=8, interactive=False)
                    prepare_btn.click(
                        prepare_sample_pack,
                        inputs=[split, limit],
                        outputs=[sample_status, sample_logs, sample_picker, sample_audio, sample_details, sample_rows, sample_quality],
                    ).then(app_summary, outputs=state)
                    sample_picker.change(selected_sample, inputs=sample_picker, outputs=[sample_audio, sample_details])

                with gr.Tab("2 · Baseline"):
                    gr.Markdown("## Reproduce the existing model\nEvaluate the published Waxal fine-tune on the exact sample pack prepared in step 1. The Yoruba hint is retained as a baseline experiment, not assumed to be optimal Akan handling.")
                    model_id = gr.Textbox(
                        value="teckedd/whisper_small-waxal_akan-asr-v1",
                        label="Model ID",
                        placeholder="openai/whisper-tiny or teckedd/whisper_small-waxal_akan-asr-v1",
                    )
                    with gr.Row():
                        eval_limit = gr.Slider(1, 5, value=1, step=1, label="Rows to evaluate")
                        language = gr.Dropdown(
                            choices=[
                                ("No forced language", ""),
                                ("Yoruba proxy", "yoruba"),
                                ("English proxy", "english"),
                            ],
                            value="",
                            label="Decoder strategy",
                            info="Run the same fixed rows under each strategy before choosing one.",
                        )
                    with gr.Row():
                        dry_btn = gr.Button("Dry-run eval", variant="secondary")
                        cache_btn = gr.Button("Cache model", variant="secondary")
                        eval_btn = gr.Button("Run ASR eval", variant="primary")
                    eval_status = gr.HTML()
                    with gr.Accordion("Technical log", open=False):
                        eval_logs = gr.Textbox(label="Run log", lines=10, interactive=False)
                    eval_report = gr.Markdown(
                        "_Rendered report appears here after a run._",
                        label="Rendered report",
                    )
                    eval_report_file = gr.File(label="Report file", interactive=False)
                    dry_btn.click(dry_run_eval, inputs=[model_id, eval_limit, language], outputs=[eval_status, eval_logs])
                    cache_btn.click(cache_model, inputs=model_id, outputs=[eval_status, eval_logs])
                    eval_btn.click(
                        run_eval,
                        inputs=[model_id, eval_limit, language],
                        outputs=[eval_status, eval_logs, eval_report, eval_report_file],
                    ).then(app_summary, outputs=state)

                    gr.Markdown(
                        "## Listen row by row\nPlay the original recording, then generate the reference and prediction with the same Akan TTS model. This makes text differences audible without changing the ASR score."
                    )
                    with gr.Row():
                        listen_report = eval_json_choices()
                        refresh_listen = gr.Button("Refresh evaluations", variant="secondary")
                    listen_row = gr.Dropdown(label="Evaluation row", choices=[])
                    listen_metrics = gr.Markdown("_Choose an evaluation report._")
                    original_audio = gr.Audio(label="Original utterance", type="filepath", interactive=False)
                    with gr.Row():
                        with gr.Column():
                            reference_text = gr.Textbox(label="Reference", lines=6, interactive=False)
                            reference_audio = gr.Audio(label="Synthesized reference", type="filepath", interactive=False)
                        with gr.Column():
                            prediction_text = gr.Textbox(label="Prediction", lines=6, interactive=False)
                            prediction_audio = gr.Audio(label="Synthesized prediction", type="filepath", interactive=False)
                    synthesize_pair = gr.Button("Generate both Akan audios", variant="primary")
                    synthesis_status = gr.HTML()
                    listen_report.change(
                        load_eval_inspector,
                        inputs=listen_report,
                        outputs=[listen_row, original_audio, reference_text, prediction_text, listen_metrics],
                    ).then(
                        lambda: (None, None, ""),
                        outputs=[reference_audio, prediction_audio, synthesis_status],
                    )
                    listen_row.change(
                        load_eval_row,
                        inputs=[listen_report, listen_row],
                        outputs=[original_audio, reference_text, prediction_text, listen_metrics],
                    ).then(
                        lambda: (None, None, ""),
                        outputs=[reference_audio, prediction_audio, synthesis_status],
                    )
                    refresh_listen.click(eval_json_choices, outputs=listen_report)
                    synthesize_pair.click(
                        synthesize_eval_pair,
                        inputs=[reference_text, prediction_text],
                        outputs=[reference_audio, prediction_audio, synthesis_status],
                    )
                    demo.load(
                        load_eval_inspector,
                        inputs=listen_report,
                        outputs=[listen_row, original_audio, reference_text, prediction_text, listen_metrics],
                    )

                with gr.Tab("3 · Train"):
                    gr.HTML(training_plan())
                    gr.Markdown(
                        """
                        ### Candidate matrix

                        | Experiment | What changes | Why |
                        |---|---|---|
                        | No forced language | Remove the Yoruba decoder prompt | Test whether the proxy is constraining Akan output |
                        | Yoruba proxy | Reproduce the old strategy | Controlled comparison with the published baseline |
                        | English proxy | Preserve code-switched English more naturally | Waxal contains English terms and names |
                        | Data-cleaned best | Best decoder strategy + filtered speaker-safe data | Measure the value of preprocessing separately |

                        **Required before launch:** full manifest, audio durations, empty/duplicate filtering, speaker-overlap report, fixed held-out evaluation set, and a Modal cost estimate.
                        """
                    )
                    gr.Button("Launch Modal training (locked)", interactive=False)

                with gr.Tab("4 · Compare"):
                    compare_board = gr.Markdown(comparison_board())
                    refresh_compare = gr.Button("Refresh comparison", variant="secondary")
                    refresh_compare.click(comparison_board, outputs=compare_board)
                    with gr.Accordion("Open an individual report", open=False):
                        refresh_reports = gr.Button("Refresh reports", variant="secondary")
                        report_picker = report_choices()
                        report_view = gr.Markdown("Choose a report.")
                        refresh_reports.click(report_choices, outputs=report_picker)
                        report_picker.change(load_report, inputs=report_picker, outputs=report_view)

    return demo


if __name__ == "__main__":
    build_app().queue().launch(server_name="127.0.0.1", server_port=7862, css=CSS)

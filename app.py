from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import gradio as gr


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

PREVIEW_MANIFEST = ROOT / "evals/samples/waxal_aka_asr_test_preview.jsonl"
LOCAL_MANIFEST = ROOT / "evals/samples/waxal_aka_asr_test_preview_local.jsonl"
REPORTS_DIR = ROOT / "evals/reports"
BENCHMARK_INDEX = ROOT / "evals/waxal_aka_benchmark_v1.json"
CORPUS_AUDIT = REPORTS_DIR / "waxal_aka_corpus_audit.json"
GHANA_NLP_MANIFEST = ROOT / "data/manifests/ghana_nlp_twi.jsonl"
GHANA_NLP_AUDIT = REPORTS_DIR / "ghana_nlp_corpus_audit.json"
GHANA_NLP_TEST_REPORT = REPORTS_DIR / "ghana-nlp-test-continuation-v1.json"
GHANA_WAXAL_TEST_REPORT = REPORTS_DIR / "waxal-test-ghana-nlp-continuation-v1.json"
ROUND2_SPLIT_AUDIT = REPORTS_DIR / "waxal_round2_split_audit.json"
ROUND2_JOB_STATE = ROOT / "outputs/modal_jobs/asr_round2.json"
TTS_JOB_STATE = ROOT / "outputs/modal_jobs/tts.json"
SMOKE_TTS_AUDIO = ROOT / "outputs/tts/smoke_validation_sample.wav"
OVERFIT_TTS_AUDIO = ROOT / "outputs/tts/overfit_validation_sample.wav"
DECODER_ANALYSIS = REPORTS_DIR / "waxal_decoder_analysis.json"
SMOKE_RUN_NAME = "smoke-whisper-small-waxal-aka-no-language-v5"
SMOKE_SUMMARY = ROOT / "outputs/modal" / SMOKE_RUN_NAME / "summary.json"
GHANA_SMOKE_RUN_NAME = "smoke-ghana-nlp-only-v1"
GHANA_SMOKE_SUMMARY = ROOT / "outputs/modal" / GHANA_SMOKE_RUN_NAME / "summary.json"
CANDIDATE_MODEL_ID = "teckedd/whisper-small-waxal-akan-continuation-v1"
LOCAL_CANDIDATE_MODEL = ROOT / "outputs/models/whisper-small-waxal-akan-continuation-v1/final"
ROUND2_MODEL_ID = "teckedd/whisper-small-waxal-round2-specaug-v1"
LOCAL_ROUND2_MODEL = ROOT / "outputs/verification/whisper-small-waxal-round2-specaug-v1"
ORIGINAL_MODEL_ID = "teckedd/whisper_small-waxal_akan-asr-v1"
LIVE_ASR_MODELS = (ROUND2_MODEL_ID, CANDIDATE_MODEL_ID, ORIGINAL_MODEL_ID)
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

/* ---- Radio options: light segmented control with an obvious selected state ---- */
.gradio-container [role="radiogroup"] label {
  background: var(--paper) !important;
  border: 1px solid var(--hair-strong) !important;
  color: var(--ink) !important;
  border-radius: var(--r-sm) !important;
}
.gradio-container [role="radiogroup"] label:has(input:checked) {
  background: var(--mint) !important;
  border-color: var(--line) !important;
}
.gradio-container [role="radiogroup"] label span { color: var(--ink) !important; }

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


def ghana_nlp_audit_summary() -> str:
    if not GHANA_NLP_AUDIT.exists():
        return "_Run the metadata audit before enabling a GhanaNLP training arm._"
    report = json.loads(GHANA_NLP_AUDIT.read_text(encoding="utf-8"))
    counts = report.get("split_counts", {})
    durations = report.get("durations", {})
    return f"""
### GhanaNLP readiness

| Check | Result |
|---|---|
| Current Viewer rows | **{report.get('viewer_count', 0):,}** |
| Usable transcript rows | **{report.get('usable_transcript_rows', report.get('records', 0)):,}** · {report.get('dropped_empty_transcript_rows', 0):,} dropped |
| Audio hours from metadata | **{float(durations.get('total_hours', 0)):.2f} h** |
| Training-duration filter | **{durations.get('training_eligible', 0):,} eligible** · {durations.get('below_training_minimum', 0):,} below 0.4 s |
| Deterministic partitions | train **{counts.get('train', 0):,}** · validation **{counts.get('validation', 0):,}** · test **{counts.get('test', 0):,}** |
| Duplicate transcript groups | **{report.get('duplicate_text_groups', 0):,}** kept within one partition |
| Empty transcripts retained | **{report.get('empty_text', 0):,}** |
| Speaker-safe split | **Not claimable**: the published schema has no speaker identifier |
| Dataset-card consistency | **Mismatch**: card says 21,138 pairs; current Viewer exposes {report.get('viewer_count', 0):,} |

The test partition is assigned by a stable SHA-256 hash of normalized transcript and must remain excluded from model selection. This prevents duplicate-text leakage, but cannot prevent speaker leakage without speaker metadata.
"""


def run_ghana_nlp_audit():
    ok, logs = run_command([PYTHON, "scripts/prepare_ghana_nlp.py"], timeout=1200)
    return (
        status_html(ok, "GhanaNLP metadata audit", str(GHANA_NLP_AUDIT.relative_to(ROOT))),
        logs,
        ghana_nlp_audit_summary(),
    )


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
            str(row.get("dataset_row") if row.get("dataset_row") is not None else "legacy"),
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


def build_preview(split: str, limit: int, sampling_mode: str, start_row: int, seed: int):
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
            "--mode",
            sampling_mode,
            "--start-row",
            str(int(start_row)),
            "--seed",
            str(int(seed)),
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


def prepare_sample_pack(
    split: str, limit: int, sampling_mode: str, start_row: int, seed: int
):
    manifest_status, manifest_logs, preview = build_preview(
        split, limit, sampling_mode, start_row, seed
    )
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


def new_sample_seed() -> int:
    return secrets.randbelow(1_000_000_000)


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
    if model_id == CANDIDATE_MODEL_ID and LOCAL_CANDIDATE_MODEL.exists():
        logs = f"Using verified local checkpoint: {LOCAL_CANDIDATE_MODEL}"
        return status_html(True, "Model cache", model_id), logs
    ok, logs = run_command([PYTHON, "scripts/cache_hf_model.py", "--model-id", model_id], timeout=1800)
    return status_html(ok, "Model cache", model_id), logs


@lru_cache(maxsize=3)
def live_asr_pipeline(model_id: str):
    if model_id not in LIVE_ASR_MODELS:
        raise ValueError("Choose one of the three audited Akan ASR checkpoints.")

    import torch
    from transformers import pipeline

    if model_id == ROUND2_MODEL_ID and LOCAL_ROUND2_MODEL.exists():
        model_source = str(LOCAL_ROUND2_MODEL)
    elif model_id == CANDIDATE_MODEL_ID and LOCAL_CANDIDATE_MODEL.exists():
        model_source = str(LOCAL_CANDIDATE_MODEL)
    else:
        model_source = model_id
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=model_source,
        tokenizer=model_source,
        feature_extractor=model_source,
        device=device,
        torch_dtype=dtype,
    )
    recognizer.model.generation_config.language = None
    recognizer.model.generation_config.forced_decoder_ids = None
    recognizer.model.config.forced_decoder_ids = None
    return recognizer, device


def transcribe_live_audio(audio_path: str | None, model_id: str, reference: str = ""):
    if not audio_path:
        return "", "", model_id, "", status_html(False, "Live ASR", "Record or upload audio first.")
    if model_id not in LIVE_ASR_MODELS:
        return "", "", model_id, "", status_html(False, "Live ASR", "Unsupported model selection.")

    try:
        started = time.perf_counter()
        recognizer, device = live_asr_pipeline(model_id)
        result = recognizer(audio_path, generate_kwargs={"task": "transcribe"})
        runtime = time.perf_counter() - started
        transcript = str(result.get("text", "")).strip()
        warning = (
            "Experimental checkpoint: one immutable-test repetition collapse and a speaker-4430 "
            "regression remain. This arbitrary clip is not benchmark evidence."
            if model_id == ROUND2_MODEL_ID
            else "Arbitrary live audio is diagnostic and is not benchmark evidence."
        )
        if reference.strip():
            from akan_speech.eval.wer import speech_error_rates

            metrics = speech_error_rates([reference], [transcript])
            accuracy = f"WER {metrics['wer']:.2%} · CER {metrics['cer']:.2%}"
        else:
            accuracy = "No reference supplied · qualitative test only"
        return (
            transcript,
            f"{runtime:.2f} s on {device}",
            model_id,
            accuracy,
            status_html(bool(transcript), "Live ASR", warning),
        )
    except Exception as exc:  # Gradio must return a useful failure instead of dropping the event.
        return "", "", model_id, "", status_html(False, "Live ASR failed", str(exc))


def run_eval(model_id: str, limit: int, language: str):
    manifest = LOCAL_MANIFEST if LOCAL_MANIFEST.exists() else PREVIEW_MANIFEST
    if not manifest.exists():
        return status_html(False, "ASR eval", "Prepare a sample pack first."), "", "", None, None
    safe_name = model_id.replace("/", "__").replace(":", "_")
    json_path = REPORTS_DIR / f"{safe_name}_eval_{int(limit)}row.json"
    csv_path = REPORTS_DIR / f"{safe_name}_eval_{int(limit)}row.csv"
    md_path = REPORTS_DIR / f"{safe_name}_eval_{int(limit)}row.md"
    model_source = (
        str(LOCAL_CANDIDATE_MODEL)
        if model_id == CANDIDATE_MODEL_ID and LOCAL_CANDIDATE_MODEL.exists()
        else model_id
    )
    ok, logs = run_command(
        [
            PYTHON,
            "scripts/eval_asr_manifest.py",
            "--model-id",
            model_source,
            "--display-model-id",
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
    report_path = str(json_path.relative_to(ROOT)) if ok and json_path.exists() else None
    return status_html(ok, "ASR eval", model_id), logs, report_md or "No report rendered.", report_file, report_path


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


def refresh_eval_inspector(report_path: str):
    dropdown = eval_json_choices()
    if report_path and (ROOT / report_path).exists():
        dropdown = gr.Dropdown(choices=dropdown.choices, value=report_path)
    row, audio, reference, prediction, metrics = load_eval_inspector(report_path)
    return dropdown, row, audio, reference, prediction, metrics, None, None, ""


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
        <span class="ak-chip peach">Modal scales to zero</span>
      </div>
    </div>
    """


def training_plan() -> str:
    benchmark_ready = BENCHMARK_INDEX.exists()
    audit_ready = CORPUS_AUDIT.exists()
    decoder_ready = DECODER_ANALYSIS.exists()
    smoke_ready = SMOKE_SUMMARY.exists()
    ghana_audit_ready = GHANA_NLP_AUDIT.exists()
    ghana_smoke_ready = GHANA_SMOKE_SUMMARY.exists()
    return f"""
<div class="ak-card">
  <div class="ak-status">Training gates</div>
  <div class="ak-muted">Waxal promotion evidence and GhanaNLP preparation stay separate. Each arm requires an audit and a two-step smoke before a full L4 run.</div>
  <div class="ak-chiprow">
    <span class="ak-chip {'mint' if audit_ready else 'butter'}">corpus audit {'ready' if audit_ready else 'needed'}</span>
    <span class="ak-chip {'mint' if benchmark_ready else 'butter'}">99-row benchmark {'frozen' if benchmark_ready else 'needed'}</span>
    <span class="ak-chip {'mint' if decoder_ready else 'butter'}">decoder {'selected' if decoder_ready else 'needed'}</span>
    <span class="ak-chip {'mint' if smoke_ready else 'peach'}">smoke {'passed' if smoke_ready else 'pending'}</span>
    <span class="ak-chip {'mint' if ghana_audit_ready else 'butter'}">GhanaNLP audit {'ready' if ghana_audit_ready else 'needed'}</span>
    <span class="ak-chip {'mint' if ghana_smoke_ready else 'peach'}">GhanaNLP smoke {'passed' if ghana_smoke_ready else 'pending'}</span>
  </div>
</div>
"""


def round2_training_board() -> str:
    split_audit = {}
    if ROUND2_SPLIT_AUDIT.exists():
        split_audit = json.loads(ROUND2_SPLIT_AUDIT.read_text(encoding="utf-8"))
    state = {"deployment": "unknown", "jobs": {}}
    if ROUND2_JOB_STATE.exists():
        state = json.loads(ROUND2_JOB_STATE.read_text(encoding="utf-8"))
    jobs = state.get("jobs", {})

    def job_status(key: str) -> str:
        job = jobs.get(key) or {}
        status = job.get("status", "not started")
        call_id = job.get("call_id")
        return f"{status}{f' · {call_id}' if call_id else ''}"

    rows = split_audit.get("rows", {})
    speakers = split_audit.get("speakers", {})
    return f"""
<div class="ak-card">
  <div class="ak-status">Round 2 · contamination-safe Waxal</div>
  <div class="ak-muted">One deployed scale-to-zero runner. Jobs persist if this UI disconnects; saved call IDs reconnect status without spawning duplicates.</div>
  <div class="ak-chiprow">
    <span class="ak-chip {'mint' if split_audit.get('passed') else 'butter'}">metadata {'passed' if split_audit.get('passed') else 'needed'}</span>
    <span class="ak-chip {'mint' if state.get('deployment') == 'ready' else 'butter'}">runner {state.get('deployment', 'unknown')}</span>
    <span class="ak-chip sky">prepare {jobs.get('prepare', {}).get('status', 'not started')}</span>
    <span class="ak-chip peach">smoke {jobs.get('train_smoke', {}).get('status', 'not started')}</span>
    <span class="ak-chip rose">pilot {jobs.get('train_pilot', {}).get('status', 'locked')}</span>
    <span class="ak-chip rose">full {jobs.get('train_full', {}).get('status', 'locked')}</span>
    <span class="ak-chip sky">test {jobs.get('evaluate_test', {}).get('status', 'locked')}</span>
  </div>
</div>

| Gate | Evidence |
|---|---|
| Frozen split | {rows.get('train', 0):,} train / {rows.get('dev', 0):,} dev / {rows.get('test', 0):,} test |
| Speaker isolation | {speakers.get('train', 0)} / {speakers.get('dev', 0)} / {speakers.get('test', 0)} speakers; zero overlap |
| Deployed runner | `{state.get('deployment', 'unknown')}` |
| CPU audio audit | {job_status('prepare')} |
| Two-step smoke | {job_status('train_smoke')} |
| 200-step pilot | {job_status('train_pilot')} |
| Full training | {job_status('train_full')} |
| Immutable test | {job_status('evaluate_test')} |
"""


def run_round2_action(action: str):
    commands = {
        "deploy": [PYTHON, "scripts/modal_round2_jobs.py", "deploy"],
        "prepare": [PYTHON, "scripts/modal_round2_jobs.py", "prepare"],
        "smoke": [PYTHON, "scripts/modal_round2_jobs.py", "train", "--mode", "smoke"],
        "pilot": [PYTHON, "scripts/modal_round2_jobs.py", "train", "--mode", "pilot"],
        "full": [PYTHON, "scripts/modal_round2_jobs.py", "train", "--mode", "full"],
        "evaluate-test": [PYTHON, "scripts/modal_round2_jobs.py", "evaluate-test"],
        "refresh": [PYTHON, "scripts/modal_round2_jobs.py", "status"],
    }
    if action not in commands:
        raise ValueError(f"Unsupported Round 2 action: {action}")
    ok, output = run_command(commands[action], timeout=600)
    detail = "Action accepted." if ok else output.splitlines()[-1] if output else "Action failed."
    return round2_training_board(), status_html(ok, f"Round 2 {action}", detail)


def cancel_round2_job():
    state = json.loads(ROUND2_JOB_STATE.read_text()) if ROUND2_JOB_STATE.exists() else {}
    jobs = state.get("jobs", {})
    active = next(
        (key for key, job in reversed(list(jobs.items())) if job.get("status") in {"submitted", "running"}),
        None,
    )
    if not active:
        return round2_training_board(), status_html(False, "Round 2 cancel", "No active job.")
    ok, output = run_command(
        [PYTHON, "scripts/modal_round2_jobs.py", "cancel", active], timeout=60
    )
    detail = f"Cancelled {active}." if ok else output.splitlines()[-1]
    return round2_training_board(), status_html(ok, "Round 2 cancel", detail)


def tts_training_board() -> str:
    state = {"deployment": "unknown", "jobs": {}}
    if TTS_JOB_STATE.exists():
        state = json.loads(TTS_JOB_STATE.read_text(encoding="utf-8"))
    jobs = state.get("jobs", {})

    def status(key: str, fallback: str = "locked") -> str:
        job = jobs.get(key) or {}
        value = job.get("status", fallback)
        call_id = job.get("call_id")
        return f"{value}{f' · {call_id}' if call_id else ''}"

    return f"""
<div class="ak-card">
  <div class="ak-status">Asante Twi TTS · active milestone</div>
  <div class="ak-muted">SpeechT5 first. Farmerline/Akosua is diagnostic only; commercial promotion requires the owned, consented 12-hour voice corpus.</div>
  <div class="ak-chiprow">
    <span class="ak-chip {'mint' if state.get('deployment') == 'ready' else 'butter'}">runner {state.get('deployment', 'unknown')}</span>
    <span class="ak-chip sky">data {jobs.get('prepare', {}).get('status', 'not started')}</span>
    <span class="ak-chip peach">smoke {jobs.get('train_smoke', {}).get('status', 'locked')}</span>
    <span class="ak-chip peach">overfit {jobs.get('train_overfit', {}).get('status', 'locked')}</span>
    <span class="ak-chip rose">pilot {jobs.get('train_pilot', {}).get('status', 'locked')}</span>
    <span class="ak-chip rose">full {jobs.get('train_full', {}).get('status', 'locked')}</span>
  </div>
</div>

| Ordered gate | Durable state |
|---|---|
| 1. Decode, VAD/loudness QA, hashes and text-disjoint split | {status('prepare', 'not started')} |
| 2. Tokenizer audit + 20-step smoke | {status('train_smoke')} |
| 3. Overfit 32 examples | {status('train_overfit')} |
| 4. 1,000-step diagnostic pilot | {status('train_pilot')} |
| 5. Ghanaian listener review | required before full |
| 6. Up to 8,000 steps | {status('train_full')} |

`ɛ` and `ɔ` are preserved by extending the SpeechT5 character tokenizer. Numeric forms are quarantined until their spoken-Twi expansion is reviewed.
"""


def run_tts_action(action: str):
    commands = {
        "deploy": [PYTHON, "scripts/modal_tts_jobs.py", "deploy"],
        "prepare": [PYTHON, "scripts/modal_tts_jobs.py", "prepare"],
        "smoke": [PYTHON, "scripts/modal_tts_jobs.py", "train", "--mode", "smoke"],
        "overfit": [PYTHON, "scripts/modal_tts_jobs.py", "train", "--mode", "overfit"],
        "pilot": [PYTHON, "scripts/modal_tts_jobs.py", "train", "--mode", "pilot"],
        "full": [PYTHON, "scripts/modal_tts_jobs.py", "train", "--mode", "full"],
        "refresh": [PYTHON, "scripts/modal_tts_jobs.py", "status"],
    }
    if action not in commands:
        raise ValueError(f"Unsupported TTS action: {action}")
    ok, output = run_command(commands[action], timeout=600)
    detail = "Action accepted." if ok else output.splitlines()[-1] if output else "Action failed."
    return tts_training_board(), status_html(ok, f"TTS {action}", detail)


def cancel_tts_job():
    state = json.loads(TTS_JOB_STATE.read_text()) if TTS_JOB_STATE.exists() else {}
    jobs = state.get("jobs", {})
    active = next(
        (key for key, job in reversed(list(jobs.items())) if job.get("status") in {"submitted", "running"}),
        None,
    )
    if not active:
        return tts_training_board(), status_html(False, "TTS cancel", "No active job.")
    ok, output = run_command([PYTHON, "scripts/modal_tts_jobs.py", "cancel", active], timeout=60)
    detail = f"Cancelled {active}." if ok else output.splitlines()[-1]
    return tts_training_board(), status_html(ok, "TTS cancel", detail)


def review_tts_overfit(note: str):
    if not note.strip():
        return tts_training_board(), status_html(False, "Overfit review", "Add a listening note.")
    ok, output = run_command(
        [PYTHON, "scripts/modal_tts_jobs.py", "review-overfit", "--approve", "--note", note],
        timeout=60,
    )
    detail = "Overfit gate approved; pilot is unlocked." if ok else output.splitlines()[-1]
    return tts_training_board(), status_html(ok, "Overfit review", detail)


def benchmark_board() -> str:
    if not DECODER_ANALYSIS.exists():
        return "_Run the frozen benchmark before selecting a decoder strategy._"
    analysis = json.loads(DECODER_ANALYSIS.read_text(encoding="utf-8"))
    strategy = analysis["strategies"][analysis["selected_strategy"]]
    metrics = strategy["metrics"]
    interval = strategy["wer_95_percent_interval"]
    return f"""
### Full held-out Waxal test

**1,522 utterances · 52,205 reference words · excluded from training and model selection**

| Model | WER | CER |
|---|---:|---:|
| Published baseline | 33.84% | 12.74% |
| Continuation v1 | **32.77%** | **12.47%** |

Paired bootstrap: **99.86% probability of improvement**; 95% candidate-minus-baseline WER interval **-1.90 to -0.33 points**. The candidate improved 524 rows, tied 603, worsened 395, and produced two severe repetition loops.

### Frozen benchmark v1

**99 utterances · 33 held-out speakers · 0 speaker overlap · 0 duplicate audio**

| Model | Decoder | WER | CER | Status |
|---|---|---:|---:|---:|
| Published baseline | No forced language | {metrics['wer'] * 100:.2f}% | {metrics['cer'] * 100:.2f}% | 95% WER: {interval['low'] * 100:.2f}%–{interval['high'] * 100:.2f}% |
| Continuation v1 | No forced language | **32.65%** | **12.26%** | Experimental; listening review due |

The candidate improves WER by 0.97 points. A 5,000-sample paired bootstrap gives 94.1% probability of improvement, but its 95% difference interval crosses zero. No forced language and the stored Yoruba prompt produced identical outputs.
"""


def ghana_experiment_board() -> str:
    if not GHANA_NLP_TEST_REPORT.exists():
        return "_GhanaNLP held-out evaluation is not complete yet._"
    report = json.loads(GHANA_NLP_TEST_REPORT.read_text(encoding="utf-8"))
    baseline = report["runs"]["waxal_continuation_v1"]["metrics"]
    candidate = report["runs"]["ghana_nlp_continuation_v1"]["metrics"]
    waxal_result = "running"
    if GHANA_WAXAL_TEST_REPORT.exists():
        waxal_report = json.loads(GHANA_WAXAL_TEST_REPORT.read_text(encoding="utf-8"))
        metrics = waxal_report["runs"]["ghana_nlp_continuation_v1"]["metrics"]
        waxal_result = f"{metrics['wer'] * 100:.2f}% WER / {metrics['cer'] * 100:.2f}% CER"
    return f"""
### GhanaNLP-only experiment

| Evidence | Before adaptation | GhanaNLP continuation |
|---|---:|---:|
| Validation, 602 rows | 165.53% WER | **84.58% WER** |
| Untouched test, 571 rows | {baseline['wer'] * 100:.2f}% WER | **{candidate['wer'] * 100:.2f}% WER** |

**Waxal regression check:** {waxal_result}

This is a research checkpoint, not a release: the GhanaNLP test gain is large, but absolute WER remains too high and the public dataset does not expose speaker IDs.
"""


def sync_smoke_status():
    SMOKE_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    ok, logs = run_command(
        [
            "modal",
            "volume",
            "get",
            "akan-speech-checkpoints",
            f"{SMOKE_RUN_NAME}/summary.json",
            str(SMOKE_SUMMARY),
        ],
        timeout=60,
    )
    if not ok or not SMOKE_SUMMARY.exists():
        return training_plan(), status_html(False, "Smoke run", "No completed summary yet."), logs
    summary = json.loads(SMOKE_SUMMARY.read_text(encoding="utf-8"))
    baseline = float(summary.get("baseline_metrics", {}).get("baseline_wer", 0)) * 100
    final = float(summary.get("final_metrics", {}).get("final_wer", 0)) * 100
    body = f"2 steps completed on {summary.get('cuda', 'Modal GPU')}. Validation WER moved {baseline:.2f}% → {final:.2f}%. This proves wiring only; it is not model evidence."
    return training_plan(), status_html(True, "Smoke run passed", body), logs


def launch_smoke_training():
    ok, logs = run_command(
        ["modal", "run", "modal_jobs/asr_train.py", "--smoke"], timeout=1800
    )
    plan, status, sync_logs = sync_smoke_status()
    if not ok:
        status = status_html(False, "Smoke run failed", "Inspect the technical log before any full run.")
    return plan, status, logs + "\n\n" + sync_logs


def smoke_artifact(arm: str) -> tuple[str, Path, list[str]]:
    if arm == "ghana_nlp_only":
        return (
            GHANA_SMOKE_RUN_NAME,
            GHANA_SMOKE_SUMMARY,
            ["modal", "run", "modal_jobs/asr_train.py", "--smoke", "--arm", arm],
        )
    return SMOKE_RUN_NAME, SMOKE_SUMMARY, ["modal", "run", "modal_jobs/asr_train.py", "--smoke"]


def sync_selected_smoke(arm: str):
    run_name, summary_path, _ = smoke_artifact(arm)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ok, logs = run_command(
        [
            "modal",
            "volume",
            "get",
            "akan-speech-checkpoints",
            f"{run_name}/summary.json",
            str(summary_path),
        ],
        timeout=60,
    )
    if not ok or not summary_path.exists():
        return training_plan(), status_html(False, "Smoke run", "No completed summary yet."), logs
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline = float(summary.get("baseline_metrics", {}).get("baseline_wer", 0)) * 100
    final = float(summary.get("final_metrics", {}).get("final_wer", 0)) * 100
    body = (
        f"{run_name}: 2 steps completed on {summary.get('cuda', 'Modal GPU')} with "
        f"{summary.get('train_rows', 0)} train, {summary.get('validation_rows', 0)} validation, "
        f"and {summary.get('test_rows', 0)} untouched test rows. Validation WER "
        f"{baseline:.2f}% → {final:.2f}%. Wiring evidence only."
    )
    return training_plan(), status_html(True, "Smoke run passed", body), logs


def launch_selected_smoke(arm: str):
    _, _, command = smoke_artifact(arm)
    ok, logs = run_command(command, timeout=1800)
    plan, status, sync_logs = sync_selected_smoke(arm)
    if not ok:
        status = status_html(False, "Smoke run failed", "Inspect the technical log before any full run.")
    return plan, status, logs + "\n\n" + sync_logs


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
    with gr.Blocks(title="Akan Speech Lab", css=CSS) as demo:
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
                    with gr.Row():
                        sampling_mode = gr.Radio(
                            choices=[("Seeded random", "random"), ("Sequential window", "sequential")],
                            value="random",
                            label="Sampling mode",
                        )
                        start_row = gr.Number(
                            value=0,
                            minimum=0,
                            precision=0,
                            label="Sequential start row",
                            info="Used only for sequential sampling.",
                        )
                        sample_seed = gr.Number(
                            value=42,
                            minimum=0,
                            precision=0,
                            label="Random seed",
                            info="Same seed reproduces the same rows.",
                        )
                    with gr.Row():
                        new_seed_btn = gr.Button("New seed", variant="secondary", size="sm", scale=1)
                        prepare_btn = gr.Button("Build sample pack", variant="primary", scale=3)
                    sample_status = gr.HTML()
                    sample_picker = gr.Dropdown(label="Sample to inspect", choices=[])
                    with gr.Row():
                        sample_audio = gr.Audio(label="Selected sample audio", type="filepath", interactive=False)
                        sample_details = gr.Markdown("_Build a sample pack, then select a row._")
                    sample_rows = gr.Dataframe(
                        headers=["#", "Dataset row", "Speaker", "Language", "Split", "Reference", "Normalized"],
                        datatype=["str", "str", "str", "str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        wrap=True,
                    )
                    sample_quality = gr.Markdown("_Pack checks appear here after preparation._")
                    with gr.Accordion("Technical log", open=False):
                        sample_logs = gr.Textbox(label="Run log", lines=8, interactive=False)
                    prepare_btn.click(
                        prepare_sample_pack,
                        inputs=[split, limit, sampling_mode, start_row, sample_seed],
                        outputs=[sample_status, sample_logs, sample_picker, sample_audio, sample_details, sample_rows, sample_quality],
                    ).then(app_summary, outputs=state)
                    new_seed_btn.click(new_sample_seed, outputs=sample_seed)
                    sample_picker.change(selected_sample, inputs=sample_picker, outputs=[sample_audio, sample_details])

                    gr.Markdown("## Audit GhanaNLP before training\nInspect every published row without downloading the 1.08 GB audio corpus. The audit freezes transcript-group partitions and surfaces limitations that affect evaluation credibility.")
                    ghana_audit_btn = gr.Button("Audit all GhanaNLP metadata", variant="primary")
                    ghana_audit_status = gr.HTML()
                    ghana_audit_report = gr.Markdown(ghana_nlp_audit_summary())
                    with gr.Accordion("GhanaNLP audit log", open=False):
                        ghana_audit_logs = gr.Textbox(label="Audit log", lines=8, interactive=False)
                    ghana_audit_btn.click(
                        run_ghana_nlp_audit,
                        outputs=[ghana_audit_status, ghana_audit_logs, ghana_audit_report],
                    ).then(app_summary, outputs=state)

                with gr.Tab("2 · Test ASR"):
                    gr.Markdown(
                        "## Record and transcribe\nSpeak naturally in Twi/Akan or upload a clip. "
                        "Add the words you said if you want clip-level WER/CER."
                    )
                    with gr.Row():
                        live_audio = gr.Audio(
                            label="Record or upload Akan speech",
                            sources=["microphone", "upload"],
                            type="filepath",
                            format="wav",
                        )
                        with gr.Column():
                            live_model = gr.Dropdown(
                                choices=list(LIVE_ASR_MODELS),
                                value=ROUND2_MODEL_ID,
                                label="Checkpoint",
                            )
                            live_reference = gr.Textbox(
                                label="What you said (optional)",
                                lines=3,
                                placeholder="Enter the exact Twi/Akan words for WER/CER.",
                            )
                            live_transcribe = gr.Button("Transcribe recording", variant="primary")
                            live_status = gr.HTML()
                    live_transcript = gr.Textbox(label="Model transcript", lines=4, interactive=False)
                    with gr.Row():
                        live_accuracy = gr.Textbox(label="Clip accuracy", interactive=False)
                        live_runtime = gr.Textbox(label="Runtime", interactive=False)
                        live_model_used = gr.Textbox(label="Model ID", interactive=False)
                    live_transcribe.click(
                        transcribe_live_audio,
                        inputs=[live_audio, live_model, live_reference],
                        outputs=[
                            live_transcript,
                            live_runtime,
                            live_model_used,
                            live_accuracy,
                            live_status,
                        ],
                    )

                    gr.Markdown("## Fixed sample evaluation\nEvaluate the published models on the exact sample pack prepared in step 1. This benchmark workflow remains separate from arbitrary recordings.")
                    model_id = gr.Dropdown(
                        choices=[
                            ROUND2_MODEL_ID,
                            CANDIDATE_MODEL_ID,
                            ORIGINAL_MODEL_ID,
                        ],
                        value=ROUND2_MODEL_ID,
                        label="Model",
                        allow_custom_value=True,
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
                    eval_report_path = gr.State()
                    dry_btn.click(dry_run_eval, inputs=[model_id, eval_limit, language], outputs=[eval_status, eval_logs])
                    cache_btn.click(cache_model, inputs=model_id, outputs=[eval_status, eval_logs])

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
                    eval_btn.click(
                        run_eval,
                        inputs=[model_id, eval_limit, language],
                        outputs=[eval_status, eval_logs, eval_report, eval_report_file, eval_report_path],
                    ).then(
                        refresh_eval_inspector,
                        inputs=eval_report_path,
                        outputs=[
                            listen_report,
                            listen_row,
                            original_audio,
                            reference_text,
                            prediction_text,
                            listen_metrics,
                            reference_audio,
                            prediction_audio,
                            synthesis_status,
                        ],
                    ).then(app_summary, outputs=state)
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
                    tts_board = gr.Markdown(tts_training_board())
                    tts_status = gr.HTML()
                    with gr.Row():
                        deploy_tts = gr.Button("Deploy TTS runner", variant="secondary")
                        prepare_tts = gr.Button("Audit corpus", variant="primary")
                        smoke_tts = gr.Button("20-step smoke", variant="secondary")
                        overfit_tts = gr.Button("Overfit 32", variant="secondary")
                        pilot_tts = gr.Button("1,000-step pilot", variant="secondary")
                        full_tts = gr.Button("Full run", variant="secondary")
                        refresh_tts = gr.Button("Refresh", variant="secondary")
                        cancel_tts = gr.Button("Cancel active", variant="stop")
                    deploy_tts.click(
                        lambda: run_tts_action("deploy"), outputs=[tts_board, tts_status]
                    )
                    prepare_tts.click(
                        lambda: run_tts_action("prepare"), outputs=[tts_board, tts_status]
                    )
                    smoke_tts.click(
                        lambda: run_tts_action("smoke"), outputs=[tts_board, tts_status]
                    )
                    overfit_tts.click(
                        lambda: run_tts_action("overfit"), outputs=[tts_board, tts_status]
                    )
                    pilot_tts.click(
                        lambda: run_tts_action("pilot"), outputs=[tts_board, tts_status]
                    )
                    full_tts.click(
                        lambda: run_tts_action("full"), outputs=[tts_board, tts_status]
                    )
                    refresh_tts.click(
                        lambda: run_tts_action("refresh"), outputs=[tts_board, tts_status]
                    )
                    cancel_tts.click(
                        cancel_tts_job, outputs=[tts_board, tts_status]
                    )

                    gr.Markdown(
                        "### TTS quality check\nThe 1,000-step run is intentionally paused until "
                        "the short overfit sample sounds intelligible. This avoids spending Modal "
                        "credits on a broken alignment."
                    )
                    with gr.Row():
                        gr.Audio(
                            value=str(SMOKE_TTS_AUDIO) if SMOKE_TTS_AUDIO.exists() else None,
                            label="20-step smoke sample",
                            interactive=False,
                        )
                        gr.Audio(
                            value=str(OVERFIT_TTS_AUDIO) if OVERFIT_TTS_AUDIO.exists() else None,
                            label="32-example overfit sample",
                            interactive=False,
                        )
                    overfit_note = gr.Textbox(
                        label="Ghanaian listening note",
                        placeholder="Record intelligibility, skipped/repeated words, and pronunciation.",
                    )
                    approve_overfit = gr.Button(
                        "Audio is intelligible · continue to pilot", variant="primary"
                    )
                    approve_overfit.click(
                        review_tts_overfit,
                        inputs=overfit_note,
                        outputs=[tts_board, tts_status],
                    )

                    with gr.Accordion("Completed ASR training record", open=False):
                        gr.Markdown(round2_training_board())
                        gr.Markdown(
                            "Round 2 is published experimentally at "
                            "[teckedd/whisper-small-waxal-round2-specaug-v1]"
                            "(https://huggingface.co/teckedd/whisper-small-waxal-round2-specaug-v1): "
                            "32.84% immutable-test WER / 11.79% CER. ASR training is closed."
                        )

                    with gr.Accordion("Previous experiments", open=False):
                        gr.HTML(training_plan())
                        gr.Markdown(benchmark_board())
                        gr.Markdown(ghana_experiment_board())
                        gr.Markdown(
                            "Waxal continuation remains the current benchmark at 32.77% test WER. "
                            "The GhanaNLP-only checkpoint failed promotion after Waxal regressed to 37.80%."
                        )

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
    build_app().queue().launch(server_name="127.0.0.1", server_port=7862)

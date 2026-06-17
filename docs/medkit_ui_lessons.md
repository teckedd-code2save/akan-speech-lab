# MedKit UI/Voice Lessons For Akan Speech Lab

Source studied:

- `teckedd-code2save/medkit-app`, branch `claude/ui-library-identification-ck14sy`
- `.claude/skills/cozy-cartoon-ui/SKILL.md`
- `backend/voice_agent.py`
- `backend/server.py`
- `src/styles/global.css`
- `src/components/primitives.tsx`

## Transferable Patterns

- Keep the interface as an operator console: clear stages, status, artifacts, and next action.
- Separate long-running voice/model work from UI state. The UI should launch controlled jobs and show reports.
- Make backend/secret work explicit and server-side. Do not leak tokens or signed URLs into committed files.
- Render model outputs as cards and reports, not raw terminal dumps.
- Use a warm, playful design language only if it clarifies state rather than decorating the workflow.
- Verification matters: MedKit ships fast tests and deterministic verification scripts; this repo should do the same for manifests, evals, and future training jobs.

## Voice/ASR Takeaways

- Voice systems need a pipeline view: audio input, STT/ASR, language/model choice, transcript, evaluation, output artifact.
- The winning app made voice feel live and inspectable. Our lab should make ASR experiments inspectable: sample audio, reference text, prediction, normalized text, WER/CER.
- Separate demo-time interaction from heavy jobs. Local Gradio should prove and inspect; Modal should run expensive training/eval only after local artifacts are valid.


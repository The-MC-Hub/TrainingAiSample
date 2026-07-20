# CLAUDE.md — TrainingAiSample

Guidance for Claude Code when working in this repo. Written from a direct code audit (2026-07-20).

## What this is

The **working dev/training source** for the MC Voice Training AI service — not a sample, not a notebook. Holds the same core FastAPI voice-analysis service as `HF-Space-Deploy` (sibling repo, production), plus the full offline model-training pipeline (Whisper fine-tuning, GPT-SoVITS voice cloning, raw speech corpora) that never ships to production.

**Read [README.md](./README.md) first** for the endpoint/structure reference — it has the detail this file assumes.

## Critical: this repo's `main.py` currently lacks TTS endpoints

`main.py` right now exposes only `POST /analyze-voice`, `GET /logs`, `GET /`. **`/generate-mc-voice` and `/tts/stream` (present in `HF-Space-Deploy`) do not exist here.**

Why: `requirements.txt` already dropped `transformers`/`accelerate` (MMS-TTS-VIE's dependencies) in favor of `supertonic-py`+`soundfile`, but the TTS route handlers haven't been rewritten against the new library yet. This is a genuine in-progress migration, not a documentation gap — verify with `grep "^@app\." main.py` before assuming otherwise.

**Do not copy this repo's `main.py` into `HF-Space-Deploy` until the Supertonic TTS endpoints are added back.** Doing so would remove working TTS from production.

**Do not assume `AI_ANALYSIS_WORKFLOW.md`'s §15 (TTS/Supertonic) describes working code in this repo** — it describes the target design after migration completes. It's explicitly flagged inside that file now; don't remove the flag without re-verifying `main.py`.

## Structure

```
main.py                    — FastAPI service, analysis-only right now (~1130 lines)
requirements.txt
Dockerfile                 — generic python:3.11-slim, no HF-Spaces-specific bits
Procfile / render.yaml     — Render.com deploy target (service mc-voice-ai, region Oregon)
.env                       — checked in with real threshold values (see security note below)

download_model.py           one-time TTS model weight download
download_dataset.py          interactive corpus downloader (VietSuperSpeech, VIVOS/OpenSLR)
preprocess_audio.py          resample/normalize raw audio → TTS (22050Hz) + STT (16000Hz) training sets
finetune_whisper.py          fine-tune Whisper small on Vietnamese MC speech
launch_sovits_training.py    pre-flight launcher for GPT-SoVITS WebUI (recommended path)
run_sovits_training.py       headless CLI alternative to the WebUI

library/                     raw speech corpora
training_data/               preprocessed metadata + wav sets
models/mms-tts-vie/          downloaded TTS weights
GPT-SoVITS/                  vendored third-party voice-cloning repo (own .git, own venv)
```

`FINETUNE_GUIDE.md` currently only documents the WebUI path (`launch_sovits_training.py`'s recommended flow) — it doesn't mention `finetune_whisper.py` or `run_sovits_training.py` explicitly. If you're asked to fine-tune, check both the guide and these scripts directly; the guide may be behind the actual tooling.

## Security note

A real `.env` with actual threshold values is checked into this repo (not just `.gitignore`d and locally present — it's tracked). Don't add new secrets to it without first confirming whether this repo is private; if it's ever made public, treat anything in that `.env` as already exposed.

## Local dev

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python download_model.py                                    # one-time
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
curl http://localhost:8000/
```

`main.py` forces stdout to UTF-8 (`sys.stdout = TextIOWrapper(...)`) — this is a Windows-console compatibility fix, confirming this file is actively run locally on Windows during development, not just edited-then-copied to the deploy repo.

## Deploy target ambiguity

`Procfile`/`render.yaml` point at Render.com (`mc-voice-ai`, `ALLOWED_ORIGINS=https://mc-voice-training.vercel.app,http://localhost:5173`) — separate from `HF-Space-Deploy`'s Hugging Face target. Whether this Render service is actually live-serving the backend's `AI_ANALYZE_URL`/`AI_TTS_URL`, or is just a staging/scratch deploy, isn't resolvable from code alone. Check the backend's actual `.env` value and the Render dashboard before assuming this is (or isn't) production.

## Relationship to `HF-Space-Deploy`

This is upstream; `HF-Space-Deploy` is the downstream deploy-only trim. Changes here are ported manually, not auto-synced — when editing something that should eventually ship, check whether it also needs a corresponding edit in `HF-Space-Deploy/main.py`, and don't assume the two are in sync just because they look similar.

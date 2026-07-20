---
title: Voice AI Training (dev/training source)
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# MC Hub AI Service — dev/training working copy

> This is the **working source repo** for the MC Voice Training AI service — not a sample, not a notebook. `HF-Space-Deploy`'s `main.py` is trimmed/copied from this repo when a change is ready for production. This repo additionally holds the full offline model-training pipeline (Whisper fine-tuning, GPT-SoVITS voice cloning, raw speech corpora) that never ships to the deploy target.

Written from a direct code audit (2026-07-20). **This repo's `main.py` has diverged from `HF-Space-Deploy`'s in a way that currently drops TTS functionality** — see "Current state vs. HF-Space-Deploy" below before assuming both repos expose the same API.

---

## Structure

```
main.py                    (~1130 lines — FastAPI service, analysis-only right now)
requirements.txt
Dockerfile                 (generic python:3.11-slim, no HF-Spaces-specific bits)
Procfile                   (Render.com: uvicorn main:app --host 0.0.0.0 --port $PORT)
render.yaml                (Render.com deploy config, service name mc-voice-ai)
.env                       (checked in — see note below)
AI_ANALYSIS_WORKFLOW.md    (13-stage deep dive into the scoring pipeline)
FINETUNE_GUIDE.md          (4-phase guide: collect audio → preprocess → GPT-SoVITS fine-tune → deploy)

download_model.py           downloads TTS model weights (one-time setup)
download_dataset.py         interactive corpus downloader (VietSuperSpeech, VIVOS/OpenSLR)
preprocess_audio.py         resamples/normalizes raw audio into TTS (22050Hz) + STT (16000Hz) training sets
finetune_whisper.py         fine-tunes Whisper small on Vietnamese MC-style speech
launch_sovits_training.py   pre-flight launcher for GPT-SoVITS WebUI voice-cloning training
run_sovits_training.py      headless CLI alternative to the WebUI launcher

library/                    raw speech corpora (bud500, MC_BTV_Bac, MC_BTV_Nam, MC_Nguyen_Ngoc_Ngan, vivos, vietspeech, ...)
training_data/               preprocessed metadata + wav sets for fine-tuning
models/mms-tts-vie/          downloaded TTS weights (gitignored in principle; present locally)
GPT-SoVITS/                  vendored third-party voice-cloning repo (own .git, own venv)
```

**Note on `.env`:** a real `.env` file with actual threshold values is checked into this repo despite `.gitignore` intent — treat any secrets in it as already-exposed if this repo is ever made public; don't add new secrets to it without rotating exposure risk in mind.

## Current state vs. `HF-Space-Deploy` — read this before touching TTS code

As of this audit, `main.py` here exposes only:

| Method | Path |
|---|---|
| `POST` | `/analyze-voice` |
| `GET` | `/logs` |
| `GET` | `/` |

**`/generate-mc-voice` and `/tts/stream` (present in `HF-Space-Deploy`) do not exist in this file right now.** The TTS stack is mid-migration: `requirements.txt` has already dropped `transformers`/`accelerate` (the MMS-TTS-VIE dependencies) in favor of `supertonic-py`+`soundfile`, but `main.py` hasn't had the Supertonic-based TTS endpoints added back yet. If you need working TTS locally, either finish wiring Supertonic here or pull the TTS block from `HF-Space-Deploy/main.py` (MMS-TTS-based) as a stopgap — don't assume this repo's `/` health-check `tts_loaded` field reflects a real TTS path.

Other differences from `HF-Space-Deploy`:
- Adds `GET /logs` (in-memory/streamed log viewer, backed by `logging`+`threading`+`deque`) and a `_push_to_java()` helper that forwards log entries to the Java backend — not present in the deploy repo.
- Forces stdout to UTF-8 (`sys.stdout = TextIOWrapper(...)`) for Windows console compatibility — confirms this file is actively run locally on Windows during development, not just edited-then-copied.
- Drops the standalone `compute_sentence_feedback` helper that `HF-Space-Deploy/main.py` still has.

## Deploy target

`Procfile`/`render.yaml` point at **Render.com** (service `mc-voice-ai`, region Oregon, `ALLOWED_ORIGINS=https://mc-voice-training.vercel.app,http://localhost:5173`) — a different/additional deploy target from `HF-Space-Deploy`'s Hugging Face Space. Whether this Render service is actually the one live-serving `AI_ANALYZE_URL`/`AI_TTS_URL` for the backend, or just a staging target, is not resolvable from code alone — check the backend's actual `.env` value and Render dashboard to confirm which deployment is live.

## Training pipeline

See `FINETUNE_GUIDE.md` for the full 4-phase workflow and `AI_ANALYSIS_WORKFLOW.md` for how the scoring pipeline works internally (FFT → Mel spectrogram → Whisper → WER → rhythm/pacing/pause scoring → bilingual report generation). Summary:

1. `download_dataset.py` — pull VietSuperSpeech/VIVOS corpora into `library/`
2. `preprocess_audio.py` — normalize + split into `training_data/tts_wavs/` (22050Hz) and `training_data/stt_wavs/` (16000Hz)
3. `finetune_whisper.py` — fine-tune Whisper small on the STT set (optimized for an 8GB-VRAM GPU, e.g. RTX 4060)
4. `launch_sovits_training.py` (recommended, WebUI) or `run_sovits_training.py` (headless) — GPT-SoVITS voice-cloning fine-tune on the TTS set

None of this training tooling exists in `HF-Space-Deploy` — that repo only ever receives the trained model outputs and the trimmed `main.py`.

## Relationship to `HF-Space-Deploy`

This is the upstream source; `HF-Space-Deploy` is the downstream deploy-only trim. Changes here need to be manually ported (not auto-synced) when ready to ship. Given the current TTS-endpoint gap described above, **do not copy this repo's `main.py` into `HF-Space-Deploy` as-is** — it would remove working TTS endpoints from production until the Supertonic migration is finished here.

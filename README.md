---
title: Voice AI Training
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# MC Hub AI Service

> FastAPI voice analysis engine for the MC Voice Training platform. Transcribes recordings with Whisper, scores pronunciation accuracy, rhythm, and pacing, then returns bilingual (VI/EN) expert feedback.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi) ![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai) ![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [API Endpoints](#api-endpoints)
- [Scoring System](#scoring-system)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Getting Started](#getting-started)
- [Fine-tuning](#fine-tuning)
- [GPU Notes](#gpu-notes)

---

## Overview

The MC Hub AI Service is a Python FastAPI server that powers the voice analysis pipeline in MC Voice Training. It accepts a recorded audio file and the original script, then returns:

- **Accuracy score** — how closely the spoken text matches the script (Whisper STT + Levenshtein WER)
- **Rhythm score** — dynamic variation in voice energy (librosa onset detection)
- **Pacing score** — words-per-minute vs. lesson target range
- **Pause analysis** — average and max pause durations
- **Per-criterion scores** — weighted scoring across PRONUNCIATION, ACCURACY, RHYTHM, EMOTION, PACING
- **Bilingual evaluation reports** — Vietnamese + English feedback, action plans, and expert tips
- **TTS synthesis** — Vietnamese voice generation via `facebook/mms-tts-vie`

Default port: **8000** (or 8001 when run via `__main__`).

---

## Architecture

```
Audio File + Script (multipart/form-data)
  → POST /analyze-voice
      ↓
  Stage 1 — STT (Whisper small)
    Transcribe audio → spoken_text
      ↓
  Stage 2 — Audio Analysis (librosa)
    Load at 16kHz → duration, WPM, onset_strength, pause detection
      ↓
  Stage 3 — Scoring
    WER (jiwer) → accuracy_score
    std(onset_env) × 20 → rhythm_score
    compute_pacing_score(wpm, target_range) → pacing_score
    compute_criteria_scores() → per-criterion map
    compute_overall_score() → weighted average
      ↓
  Stage 4 — Bilingual Evaluation
    generate_bilingual_evaluation()
      → feedback_vi / feedback_en
      → tips_vi / tips_en
      → report_vi / report_en (Markdown tables)
      → action plans
      ↓
  JSON Response
```

### Models loaded at startup

| Model | Source | Purpose |
|---|---|---|
| Whisper `small` | `openai/whisper-small` | Speech-to-text (auto language detect) |
| MMS-TTS-VIE | `facebook/mms-tts-vie` | Vietnamese TTS synthesis |

Both models use CUDA automatically if a compatible GPU is detected (RTX 4060 / CUDA 12.1 recommended).

---

## API Endpoints

Base URL: `http://localhost:8000`

### `GET /`

Health check. Returns device info, GPU name, Whisper model size, TTS load status.

```json
{
  "message": "MC Hub AI Service is running",
  "device": "cuda",
  "gpu": "NVIDIA GeForce RTX 4060",
  "whisper_model": "small",
  "tts_loaded": true
}
```

---

### `POST /analyze-voice`

Analyzes a voice recording against a reference script.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | audio file | Yes | — | `.webm`, `.wav`, `.mp3`, `.ogg`, `.m4a`, `.mp4` |
| `script_origin` | string | Yes | — | The original script text to compare against |
| `target_wpm_min` | int | No | 120 | Lower bound of target speaking rate |
| `target_wpm_max` | int | No | 150 | Upper bound of target speaking rate |
| `evaluation_hint` | string | No | null | Lesson-specific note shown in the report |
| `evaluation_criteria_json` | JSON string | No | null | Array of `{aspect, weight}` objects for weighted scoring |

**Criteria aspects:** `PRONUNCIATION`, `ACCURACY`, `RHYTHM`, `EMOTION`, `PACING`

**Example criteria JSON:**
```json
[
  {"aspect": "PRONUNCIATION", "weight": 30},
  {"aspect": "RHYTHM", "weight": 30},
  {"aspect": "PACING", "weight": 20},
  {"aspect": "EMOTION", "weight": 20}
]
```

**Response**

```json
{
  "status": "success",
  "text_spoken": "Chào mừng quý vị và các bạn...",
  "accuracy_score": 87.5,
  "rhythm_score": 42.3,
  "speaking_rate_wpm": 138.0,
  "criteria_scores": {
    "PRONUNCIATION": 87.5,
    "RHYTHM": 42.3,
    "PACING": 100.0,
    "EMOTION": 46.5
  },
  "overall_score": 71.2,
  "feedback": "Phát âm: Tốt... | Nhấn nhá: Biến điệu tốt...",
  "feedback_vi": "Phát âm: Tốt... | Nhấn nhá: Biến điệu tốt...",
  "feedback_en": "Pronunciation: Good... | Emphasis: Good dynamic variation.",
  "tips_vi": [{"label": "TỐC ĐỘ", "tip": "Tốc độ lý tưởng (138 WPM)."}],
  "tips_en": [{"label": "PACING", "tip": "Pace is ideal (138 WPM)."}],
  "report_vi": "### 🎙️ Báo cáo Phân tích...",
  "report_en": "### 🎙️ Advanced AI Performance Report...",
  "analysis_meta": {
    "device_used": "cuda",
    "avg_pause_sec": 0.52,
    "pause_count": 5,
    "duration_sec": 14.3,
    "target_wpm_min": 120,
    "target_wpm_max": 150
  }
}
```

> **Legacy fields:** `feedback` mirrors `feedback_vi`; `expert_tips` mirrors `tips_vi`. Kept for backward compatibility with older frontend versions.

---

### `POST /generate-mc-voice`

Synthesizes Vietnamese speech from text using the MMS-TTS-VIE model.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | Yes | Vietnamese text to synthesize |

**Response**

```json
{
  "status": "success",
  "message": "MC voice generated successfully",
  "file_path": "mc_voice_output.wav"
}
```

Saves output to `mc_voice_output.wav` in the working directory.

---

## Scoring System

### Accuracy Score

Uses **Whisper** for STT then **jiwer WER** (Word Error Rate) via Levenshtein distance:

```
accuracy_score = max(0, 100 - (WER × 100))
```

WER counts Substitutions + Deletions + Insertions against total reference words.

### Rhythm Score

Derived from the **standard deviation of onset strength** (librosa):

```
onset_env = librosa.onset.onset_strength(y, sr=16000)
rhythm_raw = std(onset_env)
rhythm_score = min(100, rhythm_raw × 20)
```

Higher std = more dynamic variation in voice energy = better expressiveness.

### Pacing Score

```
if target_min ≤ WPM ≤ target_max → 100.0
else penalty = (distance_from_range / range_span) × 100
pacing_score = max(0, 100 - penalty)
```

Target range default: 120–150 WPM (Vietnamese MC standard).

### Pause Detection

RMS energy analysis at frame level (25ms frames, 10ms hop). Pauses ≥ 150ms detected and counted.

Optimal pause: **0.5–0.8s** between sentences.

### Overall Score

Weighted average of criteria scores:
```
overall = Σ(criterion_score × weight) / Σ(weights)
```

Falls back to simple mean when no criteria provided.

---

## Project Structure

```
TrainingAiSample/
│
├── ── API Service ──────────────────────────────────────────────────
│
├── main.py
│   FastAPI server — the production AI service. Loaded at startup:
│   Whisper small (STT) and facebook/mms-tts-vie (TTS) on CUDA.
│   Contains all helper functions and two endpoints:
│     POST /analyze-voice  — full pipeline: STT → scoring → evaluation
│     POST /generate-mc-voice — Vietnamese TTS synthesis
│     GET  /              — health check (device, GPU, model status)
│   Internal helpers:
│     compute_pause_stats()       — RMS silence detection, ≥150ms pauses
│     compute_pacing_score()      — WPM vs target range, 0–100 output
│     compute_criteria_scores()   — maps PRONUNCIATION/RHYTHM/PACING/EMOTION
│     compute_overall_score()     — weighted average of criteria scores
│     generate_bilingual_evaluation() — builds VI+EN feedback/tips/report
│   CORS: allows localhost:3000, localhost:5173 (dev frontend ports)
│   Port: 8000 (uvicorn) or 8001 (python main.py __main__)
│
├── requirement.txt
│   Python dependencies. Groups:
│     Core API     → fastapi, uvicorn, python-multipart
│     AI Models    → openai-whisper, transformers, accelerate
│     Audio        → librosa, soundfile, scipy, numpy
│     Evaluation   → jiwer (WER scoring)
│     Datasets     → datasets, huggingface_hub
│     Utilities    → tqdm, static-ffmpeg (Windows FFmpeg bridge)
│   NOTE: PyTorch must be installed separately with CUDA index URL
│         (see Getting Started section)
│
├── .env
│   Local environment config — never committed (gitignored).
│   Variables: WHISPER_MODEL_SIZE, TTS_MODEL_PATH, ALLOWED_ORIGINS,
│   HOST, PORT. CORS origins currently hardcoded in main.py lines 27–35;
│   .env values reserved for future python-dotenv migration.
│
├── .gitignore
│   Excludes: models/, library/, training_data/, GPT-SoVITS/,
│   __pycache__/, venv/, .env, all audio formats (*.wav *.mp3 *.m4a
│   *.flac), model weights (*.bin *.pt *.pth *.ckpt *.onnx
│   *.safetensors), temp files (temp_*, mc_voice_output.wav)
│
│
├── ── Model Bootstrap ──────────────────────────────────────────────
│
├── download_model.py
│   One-time setup: downloads facebook/mms-tts-vie from HuggingFace
│   and saves model + tokenizer to ./models/mms-tts-vie/.
│   Run once before starting the server. Uses transformers
│   VitsModel.from_pretrained() + AutoTokenizer.from_pretrained().
│   Output: config.json, model.safetensors, tokenizer_config.json
│
│
├── ── Data Pipeline ────────────────────────────────────────────────
│
├── download_dataset.py  (v5 — VERIFIED WORKING)
│   Interactive dataset downloader. Menu-driven (5 options):
│     1. VietSuperSpeech 2000 samples  (~20 min)
│     2. VietSuperSpeech FULL 267h     (hours)
│     3. VIVOS via OpenSLR             (~1.5 GB zip)
│     4. Both VIVOS + VSS 2000         (recommended for STT+TTS)
│     5. Quick test 200 samples        (~3 min)
│   Functions:
│     download_vietsuperspeech(max_samples) — streams HuggingFace
│       dataset thanhnew2001/VietSuperSpeech, downloads each WAV
│       via hf_hub_download, resamples to 16kHz with librosa,
│       writes library/vietsuperspeech/vss_XXXXXX.wav
│     download_vivos_openslr() — urllib downloads vivos.zip from
│       openslr.org/101, extracts, parses prompts.txt → metadata
│     merge_metadata() — scans all library/*/metadata.txt files,
│       merges into training_data/metadata_master.txt
│   Output metadata format (pipe-delimited):
│     library/vietsuperspeech/vss_000000.wav|VSS_SPK|vi|transcript
│
├── preprocess_audio.py  (v2)
│   Reads training_data/metadata_master.txt and produces two clean
│   datasets for separate model fine-tuning:
│   TTS branch → training_data/tts_wavs/  (for GPT-SoVITS)
│     - Resample to 22050Hz (GPT-SoVITS requirement)
│     - Peak normalize to −3 dBFS
│     - Save as PCM 16-bit WAV
│   STT branch → training_data/stt_wavs/  (for Whisper fine-tune)
│     - Resample to 16000Hz (Whisper requirement)
│     - Save as PCM 16-bit WAV
│   Filters: skip if duration < 1s (too short) or > 15s (noisy)
│   Outputs: tts_metadata.txt and stt_metadata.txt
│   Prints summary: Input / TTS out / STT out / Skipped / Errors
│
│
├── ── Fine-Tuning Scripts ──────────────────────────────────────────
│
├── finetune_whisper.py
│   Fine-tunes Whisper small on Vietnamese speech to improve STT
│   accuracy for MC-style audio (regional accents, event scripts).
│   Input:  training_data/stt_metadata.txt (wav_path|transcript)
│   Output: models/whisper-vi-finetuned/ (full model + processor)
│   Config (optimized for RTX 4060 8GB):
│     MODEL_NAME    = openai/whisper-small
│     BATCH_SIZE    = 4  (safe for 8GB VRAM)
│     GRAD_ACCUM    = 4  → effective batch 16
│     MAX_STEPS     = 1000  (~30 min on RTX 4060 with 2000 samples)
│     LEARNING_RATE = 1e-5
│     fp16 = True (FP16 training on CUDA)
│   Uses HuggingFace Seq2SeqTrainer + WER metric (evaluate lib).
│   Data collator: WhisperDataCollator pads inputs + masks labels.
│   After training: update main.py to load from models/whisper-vi-finetuned/
│
├── launch_sovits_training.py
│   Pre-flight launcher for GPT-SoVITS WebUI TTS training.
│   Run AFTER finetune_whisper.py completes.
│   Steps:
│     1. check_prereqs() — verifies GPT-SoVITS dir exists, counts
│        TTS WAVs in training_data/tts_wavs/, checks tts_metadata.txt
│     2. create_sovits_config() — writes models/gpt-sovits-vi/
│        training_config.json with pretrained model paths, batch=4,
│        epochs=8, language=vi, experiment name=MC_Hub_Vietnamese
│     3. launch_webui() — chdir to GPT-SoVITS/, sets
│        CUDA_VISIBLE_DEVICES=0, runs webui.py
│   Prints browser URL (http://localhost:9872) and WebUI instructions.
│   Input:  training_data/tts_wavs/ + training_data/tts_metadata.txt
│   Output: models/gpt-sovits-vi/ (config + trained weights)
│
├── run_sovits_training.py
│   Lower-level GPT-SoVITS pipeline runner (Phase 5).
│   Directly orchestrates the 3 feature extraction steps via subprocess
│   instead of going through the WebUI.
│   Steps:
│     1. Validate: checks VENV_PYTHON, metadata, WAV count
│     2. Build annotation: reads metadata_master.txt, copies up to
│        1000 WAVs into GPT-SoVITS/logs/mc_vi_voice/0_gt_wavs/,
│        writes ann.list in format: wav_path|speaker|vi|text
│     3. Run prepare_datasets/1-get-text.py  — text + BERT features
│     4. Run prepare_datasets/2-get-hubert-wav32k.py — HuBERT SSL features
│     5. Run prepare_datasets/3-get-semantic.py — semantic token extraction
│   After preprocessing done: instructs user to launch webui.py manually.
│   Difference from launch_sovits_training.py:
│     - launch_sovits_training.py = WebUI launcher (recommended, easier)
│     - run_sovits_training.py = headless CLI pipeline (advanced, direct)
│
│
├── ── Documentation ────────────────────────────────────────────────
│
├── README.md
│   This file. Full technical reference for the AI service.
│
├── AI_ANALYSIS_WORKFLOW.md
│   13-stage deep-dive into how the analysis pipeline works:
│   audio physics → FFT → Mel spectrogram → Whisper encoder/decoder →
│   WER scoring → rhythm (onset_strength std) → pacing (WPM formula) →
│   pause detection (RMS silence) → criteria scoring → bilingual
│   evaluation → end-to-end example with numeric values.
│
└── FINETUNE_GUIDE.md
    4-phase professional guide for training a custom Vietnamese MC voice:
    Phase 1 (collect audio) → Phase 2 (preprocess) →
    Phase 3 (GPT-SoVITS fine-tune) → Phase 4 (deploy to main.py).
    Includes hardware requirements, recording guidelines, WebUI
    config for RTX 4060, timing estimates, and troubleshooting.
```

### Generated directories (gitignored — not in repo)

```
models/
├── mms-tts-vie/            ← download_model.py output
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer_config.json
├── whisper-vi-finetuned/   ← finetune_whisper.py output
└── gpt-sovits-vi/          ← launch_sovits_training.py output
    └── training_config.json

library/
├── vietsuperspeech/        ← download_dataset.py (VSS option)
│   ├── vss_000000.wav
│   ├── ...
│   └── metadata.txt
└── vivos/                  ← download_dataset.py (VIVOS option)
    └── vivos/
        ├── train/waves/
        └── test/waves/

training_data/
├── metadata_master.txt     ← merged by download_dataset.py
├── tts_wavs/               ← 22050Hz WAVs for GPT-SoVITS
├── stt_wavs/               ← 16000Hz WAVs for Whisper
├── tts_metadata.txt        ← preprocess_audio.py output
└── stt_metadata.txt        ← preprocess_audio.py output

GPT-SoVITS/                 ← clone separately (~10 GB)
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Whisper model size: tiny | base | small | medium | large
# small = best balance for RTX 4060 8GB VRAM
WHISPER_MODEL_SIZE=small

# TTS model path (auto-populated by download_model.py)
TTS_MODEL_PATH=./models/mms-tts-vie

# CORS allowed origins
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000

# Server
HOST=127.0.0.1
PORT=8000
```

> Note: CORS origins are currently hardcoded in `main.py` (lines 27–35). The `.env` values are available for future use when you move to `python-dotenv`.

---

## Getting Started

### Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA 12.1+ recommended (CPU fallback works, but slow)
- FFmpeg — **handled automatically** via `static-ffmpeg` package on Windows

### Install

```bash
# 1. Install PyTorch with CUDA support first (RTX 4060 / CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Install all other dependencies
pip install -r requirement.txt
```

### Download TTS Model (first run only)

```bash
python download_model.py
# Saves facebook/mms-tts-vie to ./models/mms-tts-vie/
```

### Run Server

```bash
# Development (with hot reload)
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Or via __main__ (port 8001)
python main.py
```

### Verify

```bash
curl http://localhost:8000/
```

---

## Fine-tuning

To train a custom Vietnamese MC voice, see **[FINETUNE_GUIDE.md](FINETUNE_GUIDE.md)**.

Pipeline summary:
1. Collect reference audio → `library/`
2. Run `python fetch_samples.py` to download HuggingFace datasets
3. Run `python preprocess_audio.py` to normalize + split audio
4. Fine-tune with GPT-SoVITS WebUI using `training_data/tts_wavs/`

For the full AI analysis pipeline explanation (FFT, Mel Spectrogram, Whisper architecture, WER algorithm, rhythm scoring), see **[AI_ANALYSIS_WORKFLOW.md](AI_ANALYSIS_WORKFLOW.md)**.

---

## GPU Notes

| Hardware | Whisper Model | VRAM Usage | Inference Speed |
|---|---|---|---|
| RTX 4060 8GB | `small` | ~1.5GB | ~2–4s per 30s audio |
| RTX 4060 8GB | `medium` | ~3GB | ~4–8s per 30s audio |
| RTX 4060 8GB | `large` | ~6GB | ~10–15s per 30s audio |
| CPU only | `tiny` | — | ~15–30s per 30s audio |

Default config uses `small` — optimal speed/accuracy tradeoff for RTX 4060.

---

MIT © The MC Hub Team

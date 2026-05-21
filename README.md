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
├── main.py               # FastAPI server — all analysis + TTS endpoints
├── download_model.py     # Downloads facebook/mms-tts-vie to ./models/
├── fetch_samples.py      # Downloads MC voice reference samples to ./library/
├── preprocess_audio.py   # Preprocesses raw audio for TTS/STT fine-tuning
├── requirement.txt       # Python dependencies
├── .env                  # Local config (WHISPER_MODEL_SIZE, CORS origins, etc.)
├── .gitignore            # Excludes models/, library/, training_data/, __pycache__/
│
├── models/               # Downloaded TTS model (auto-generated, gitignored)
│   └── mms-tts-vie/
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer_config.json
│
├── library/              # Reference MC voice samples (gitignored)
│   ├── MC_Nam_MienBac_VIVOS/
│   ├── MC_Nu_MienNam_CodeLink/
│   └── ...
│
├── training_data/        # Preprocessed audio for fine-tuning (gitignored)
│   ├── metadata_master.txt
│   ├── tts_wavs/         # 22050Hz normalized WAVs for GPT-SoVITS
│   ├── stt_wavs/         # 16000Hz WAVs for Whisper fine-tune
│   ├── tts_metadata.txt
│   └── stt_metadata.txt
│
├── README.md
├── AI_ANALYSIS_WORKFLOW.md   # Deep-dive: FFT → Mel → Whisper → WER → Feedback
└── FINETUNE_GUIDE.md         # GPT-SoVITS fine-tuning workflow
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

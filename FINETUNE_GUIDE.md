# Fine-Tuning Guide — MC Voice (GPT-SoVITS)

> Hướng dẫn huấn luyện giọng nói MC tiêu chuẩn từ dữ liệu thực.  
> Target hardware: RTX 4060 8GB / Windows 11. Training time: ~30 min per epoch.  
> **Last updated:** 2026-05-22

---

## Quick-Start Checklist

Run through this checklist before starting. Missing any item will cause failure later.

**Hardware & Environment**
- [ ] NVIDIA GPU with CUDA 12.1+ installed (`nvidia-smi` returns without error)
- [ ] Python 3.9–3.10 installed (`python --version`)
- [ ] 20+ GB free disk space (`dir` or `df -h`)
- [ ] FFmpeg installed (`ffmpeg -version`) — OR `static-ffmpeg` handles it automatically for the API

**Dependencies**
- [ ] PyTorch with CUDA installed: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- [ ] All requirements installed: `pip install -r requirement.txt`
- [ ] TTS model downloaded: `python download_model.py`
- [ ] GPT-SoVITS cloned into `GPT-SoVITS/` (needed for Phase 3 only)

**Data**
- [ ] At least 30 audio clips ready (100+ recommended for quality results)
- [ ] Each clip: 2–10 seconds, clean background, no music or reverb
- [ ] `training_data/metadata_master.txt` exists with correct pipe-delimited format

**Verify AI service works before fine-tuning**
```bash
# Start the base service
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Test in another terminal
curl http://localhost:8000/
# Should return: {"message": "MC Hub AI Service is running", "device": "cuda", ...}
```

---

## Overview

```
Raw MC audio files
        ↓
  python fetch_samples.py       ← download reference audio from HuggingFace
        ↓
  python preprocess_audio.py    ← validate, resample, normalize, split
        ↓
  training_data/
    ├── tts_wavs/     (22050Hz, peak-normalized)   ← GPT-SoVITS input
    ├── stt_wavs/     (16000Hz)                    ← Whisper fine-tune input
    ├── tts_metadata.txt
    └── stt_metadata.txt
        ↓
  GPT-SoVITS WebUI fine-tune
        ↓
  models/custom-mc-voice.ckpt   ← deploy to main.py
```

---

## Prerequisites

### System requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA GTX 1060 6GB | RTX 4060 8GB (CUDA 12.1) |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 50 GB free (GPT-SoVITS + datasets) |
| OS | Windows 10 | Windows 11 |
| Python | 3.9 | 3.10 |

### Install FFmpeg (Windows — required once)

```powershell
winget install ffmpeg
```

Restart terminal after install. Verify:
```powershell
ffmpeg -version
```

> The AI service already handles FFmpeg automatically via `static-ffmpeg` for the analysis API. FFmpeg must be installed system-wide only for fine-tuning scripts.

---

## Phase 1 — Collect Reference Audio

### Option A: Download from HuggingFace datasets

```bash
python fetch_samples.py
```

Downloads sample audio to `library/`:
- `MC_Nam_MienBac_VIVOS/` — Northern male voice (VIVOS dataset, public)
- `MC_Nu_MienNam_CodeLink/` — Southern female voice reference

### Option B: Use your own recordings

Create a master metadata file at `training_data/metadata_master.txt`.

**Format — one line per audio file:**
```
relative/path/to/audio.wav|SPEAKER_ID|vi|Transcript text here
```

**Example:**
```
library/my_mc/clip_001.wav|MC_Trung|vi|Chào mừng quý vị và các bạn đến với chương trình
library/my_mc/clip_002.wav|MC_Trung|vi|Hôm nay chúng ta có một buổi lễ đặc biệt
library/my_mc/clip_003.wav|MC_Trung|vi|Xin trân trọng kính mời quý khách lên sân khấu
```

**Recording guidelines:**
- Minimum 30–50 clips for a basic voice clone
- 100–200 clips for high quality
- Each clip: 2–10 seconds, clean background, consistent microphone distance
- Cover all common Vietnamese phonemes and sentence patterns
- Avoid background music or reverb

---

## Phase 2 — Preprocess Audio

```bash
python preprocess_audio.py
```

### What it does

1. **Reads** `training_data/metadata_master.txt`
2. **Validates** each WAV file:
   - Skip if duration < 1s (too short for learning)
   - Skip if duration > 15s (often contains noise)
3. **TTS branch** → `training_data/tts_wavs/`:
   - Resample to 22,050 Hz (GPT-SoVITS requirement)
   - Peak normalize to −3 dBFS
   - Save as PCM 16-bit WAV
4. **STT branch** → `training_data/stt_wavs/`:
   - Resample to 16,000 Hz (Whisper requirement)
   - Save as PCM 16-bit WAV
5. **Writes** `tts_metadata.txt` and `stt_metadata.txt`

### Example output

```
PREPROCESS COMPLETE
  Input entries    : 120
  TTS samples out  : 103  -> training_data/tts_wavs/
  STT samples out  : 103  -> training_data/stt_wavs/
  Skipped missing  : 2
  Skipped too short: 8
  Skipped too long : 7
  Errors           : 0

  TTS metadata -> training_data/tts_metadata.txt
  STT metadata -> training_data/stt_metadata.txt

  Next: Fine-tune GPT-SoVITS with tts_wavs/
```

---

## Phase 3 — Fine-Tune with GPT-SoVITS

### Step 3.1 — Clone GPT-SoVITS

```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```

> GPT-SoVITS is ~10 GB (includes pre-trained base weights). Do not commit it to the MC Hub repo — keep it in a separate directory or add to `.gitignore`.

### Step 3.2 — Copy training data

```bash
# From the TrainingAiSample root
xcopy /E /I training_data\tts_wavs GPT-SoVITS\dataset\wavs
copy training_data\tts_metadata.txt GPT-SoVITS\dataset\metadata.txt
```

### Step 3.3 — Launch WebUI

```bash
cd GPT-SoVITS
python webui.py
```

Open the URL shown in terminal (typically `http://127.0.0.1:9872`).

### Step 3.4 — Configure and train

In the WebUI:

1. **Tab: "1-GPT-SoVITS Fine-tune"**
2. Set dataset path → `dataset/`
3. Set speaker name → your speaker ID (e.g., `MC_Trung`)
4. Choose base model: `GPT-SoVITS-v2` (recommended for Vietnamese)
5. Training parameters for RTX 4060 8GB:

   | Parameter | Value |
   |---|---|
   | Batch size | 4 |
   | Epochs | 10–20 |
   | Save every N epochs | 5 |
   | Max audio length | 15s |

6. Click **"Start Training"**

**Estimated time on RTX 4060:**
- 100 clips × 10 epochs ≈ 20–30 minutes
- 200 clips × 20 epochs ≈ 60–90 minutes

### Step 3.5 — Monitor training

Check console output for:
```
Epoch 5/10  loss=0.043  ...
Epoch 10/10 loss=0.021  ← good convergence
```

If loss stops decreasing after epoch 5, training is complete. More epochs = overfitting risk.

---

## Phase 4 — Deploy to Main API

### Step 4.1 — Export model

After training, GPT-SoVITS saves:
- `GPT-SoVITS/logs/MC_Trung/MC_Trung-e10.ckpt` — GPT weights
- `GPT-SoVITS/logs/MC_Trung/MC_Trung_e8_s400.pth` — SoVITS weights

Copy to the MC Hub models directory:

```bash
mkdir models\custom-mc-voice
copy GPT-SoVITS\logs\MC_Trung\MC_Trung-e10.ckpt models\custom-mc-voice\
copy GPT-SoVITS\logs\MC_Trung\MC_Trung_e8_s400.pth models\custom-mc-voice\
```

### Step 4.2 — Update main.py

The current `main.py` uses `facebook/mms-tts-vie` (base Vietnamese TTS). To use your fine-tuned model in `/generate-mc-voice`:

```python
# Current (base model)
TTS_MODEL_PATH = "./models/mms-tts-vie"

# After fine-tuning — switch to GPT-SoVITS inference
# (Requires GPT-SoVITS inference API — see their docs for integration)
TTS_MODEL_PATH = "./models/custom-mc-voice"
```

Refer to the GPT-SoVITS repo's `inference_webui.py` for the inference API.

---

## Voice Library Reference

The `library/` directory stores reference audio for quality comparison:

| Folder | Style | Source |
|---|---|---|
| `MC_Nam_MienBac_VIVOS/` | Northern male, clear | VIVOS dataset (AILAB-VNUHCM) |
| `MC_Nu_MienNam_CodeLink/` | Southern female | HuggingFace reference |
| `MC_BTV_VTV_MienBac/` | News anchor, Northern | Internal reference |
| `MC_Nguyen_Ngoc_Ngan/` | Famous MC reference | Internal recording |
| `MC_Nu_MienNam_FPT/` | FPT University style | Internal reference |

These are reference-only — not used in training unless added to `metadata_master.txt`.

---

## Troubleshooting

### `preprocess_audio.py` — "metadata_master.txt not found"

```
[ERROR] training_data/metadata_master.txt not found!
  -> Run download_dataset.py first
```

**Fix:** Create the file manually (see Phase 1 format) or run `python download_dataset.py` first.

---

### GPT-SoVITS CUDA out of memory

**Symptom:** Training crashes with `CUDA out of memory` or `RuntimeError: CUDA error`.

**Fixes (try in order):**
1. Reduce batch size from 4 → 2 in WebUI settings
2. Close other GPU processes (`nvidia-smi` to check VRAM usage)
3. Set `max_audio_length = 10` (shorter clips use less VRAM per batch)
4. Add `--fp16` flag if your GPT-SoVITS version supports it

---

### Whisper transcription quality low on custom audio

**Symptom:** `accuracy_score` is very low (< 40) even when the recording sounds correct. Often caused by strong regional accents or uncommon vocabulary.

**Fix:** Switch to a larger Whisper model in `main.py`:
```python
# Line ~15 in main.py
# Current (fast, good for standard speech):
whisper_model = whisper.load_model("small")

# Better accuracy for regional accents (requires ~3 GB VRAM):
whisper_model = whisper.load_model("medium")

# Best accuracy (requires ~6 GB VRAM — tight on RTX 4060):
whisper_model = whisper.load_model("large")
```

Restart the server after changing the model. First startup will be slow as the model loads into VRAM.

---

### TTS output sounds robotic or unnatural

**Symptom:** The generated voice from `/generate-mc-voice` sounds robotic, flat, or has wrong tones.

**Root causes:**
1. Too few training clips (< 50) — model lacks enough examples
2. Inconsistent recording conditions — different microphone distances, room echo, or background noise between clips
3. Under-trained — loss not converged (check console: `loss > 0.05` after epoch 10 means more epochs needed)

**Fixes:**
- Collect 100–200 clips with consistent conditions
- Re-record any clips with audible background noise
- Run 15–20 epochs instead of 10 if loss is still high

---

### Python `import whisper` fails

**Symptom:** `ModuleNotFoundError: No module named 'whisper'` on startup.

**Fix:**
```bash
pip install openai-whisper
```

Note: The package name is `openai-whisper`, not `whisper`. There is a different unrelated package named `whisper` on PyPI.

---

### `librosa` audio loading fails on Windows

**Symptom:** `audioread.NoBackendError` or `RuntimeError: Error loading audio file`.

**Cause:** FFmpeg is not installed or not in PATH.

**Fix (Windows):**
```powershell
winget install ffmpeg
# Close and reopen terminal
ffmpeg -version    # verify
```

Alternatively, `static-ffmpeg` in `requirement.txt` handles this automatically for Python-level calls. If the error persists, install FFmpeg system-wide.

---

### GPT-SoVITS WebUI not accessible at port 9872

**Symptom:** Browser shows "connection refused" when opening `http://127.0.0.1:9872`.

**Cause:** WebUI failed to start — check terminal output for Python errors.

**Common causes:**
- `requirements.txt` not installed inside GPT-SoVITS virtualenv
- Port 9872 already in use by another process

**Fix:**
```bash
# Check if port is in use
netstat -aon | findstr :9872

# Run WebUI with a different port
cd GPT-SoVITS
python webui.py --port 9873
```

---

### Training loss not decreasing

**Symptom:** Console shows `loss=0.18` after epoch 5 and it's not going down.

**Likely causes:**
- Learning rate too high (reduce from default to `5e-6`)
- Dataset too small (< 30 clips)
- Clips are too long (> 15s) — model struggles with long sequences

**Fix:** Preprocess again with stricter duration filter, then retrain from scratch.

---

## Performance Expectations

| Training data | Epochs | Time (RTX 4060) | Output quality |
|---|---|---|---|
| 30–50 clips | 10 | ~10 min | Basic — recognizable voice, some artifacts |
| 100 clips | 15 | ~25 min | Good — natural intonation, correct tones |
| 200+ clips | 20 | ~50 min | Excellent — near-reference quality |

More data always helps more than more epochs. If you have 200 clips, 15 epochs is better than 50 clips at 30 epochs.

---

## .gitignore Reference

These paths are already excluded from the MC Hub repo. **Never commit these.**

```gitignore
models/              # Trained model weights (can be several GB)
library/             # Reference audio files
training_data/       # Preprocessed training datasets
GPT-SoVITS/          # External repo — clone separately
__pycache__/
venv/
*.onnx
*.ckpt               # GPT-SoVITS checkpoint weights
*.pth                # PyTorch weight files
*.bin                # HuggingFace model binary files
*.safetensors        # HuggingFace safe tensor format
*.wav                # Audio files
*.mp3
*.m4a
mc_voice_output.wav  # TTS output file
temp_*               # Temporary processing files
```

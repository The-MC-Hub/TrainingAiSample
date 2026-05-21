# Fine-Tuning Guide — MC Voice (GPT-SoVITS)

> Hướng dẫn huấn luyện giọng nói MC tiêu chuẩn từ dữ liệu thực.  
> Target hardware: RTX 4060 8GB / Windows 11. Training time: ~30 min per epoch.

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

Create the file manually (see Phase 1 format) or run `fetch_samples.py` first.

### GPT-SoVITS CUDA out of memory

Reduce batch size to 2, or use `--fp16` flag if supported.

### Whisper transcription quality low on custom audio

Switch `stt_model = whisper.load_model("medium")` in `main.py` for better accuracy on regional accents. Requires ~3 GB VRAM.

### TTS output sounds robotic

Increase training data (aim for 200+ clips). Ensure consistent recording conditions — same room, microphone distance, and vocal energy throughout.

---

## .gitignore Reference

These paths are already excluded from the MC Hub repo:

```gitignore
models/
library/
training_data/
GPT-SoVITS/
__pycache__/
*.onnx
*.ckpt
*.pth
```

Never commit trained model weights or audio datasets.

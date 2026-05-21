# AI Voice Analysis — Technical Workflow

> Tài liệu kỹ thuật đầy đủ cho pipeline phân tích giọng nói AI của The MC Hub.  
> Covers the complete data lifecycle: microphone input → Whisper transcription → scoring → bilingual expert feedback.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Stage 1 — Audio Physics & Sampling](#2-stage-1--audio-physics--sampling)
3. [Stage 2 — FFT & Spectrogram](#3-stage-2--fft--spectrogram)
4. [Stage 3 — Mel Filterbank](#4-stage-3--mel-filterbank)
5. [Stage 4 — Whisper STT (Encoder-Decoder)](#5-stage-4--whisper-stt-encoder-decoder)
6. [Stage 5 — Attention Mechanism](#6-stage-5--attention-mechanism)
7. [Stage 6 — Accuracy Scoring (WER + Levenshtein)](#7-stage-6--accuracy-scoring-wer--levenshtein)
8. [Stage 7 — Rhythm & Pacing Analysis](#8-stage-7--rhythm--pacing-analysis)
9. [Stage 8 — Pause Detection](#9-stage-8--pause-detection)
10. [Stage 9 — Criteria Scoring & Overall Score](#10-stage-9--criteria-scoring--overall-score)
11. [Stage 10 — Bilingual Expert Feedback](#11-stage-10--bilingual-expert-feedback)
12. [End-to-End Example](#12-end-to-end-example)
13. [Technical Spec Table](#13-technical-spec-table)

---

## 1. Pipeline Overview

```
🎙️  Audio File (webm/wav/mp3)
        ↓
┌───────────────────────────────────────────┐
│  Stage 1: librosa.load() → 16kHz PCM     │
│  Stage 2: FFT (Hamming window, 25ms hop) │
│  Stage 3: 80-band Mel Filterbank         │
│  Stage 4: Whisper Encoder (6 layers)     │
│  Stage 5: Whisper Decoder → text_spoken  │
└───────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────┐
│  Stage 6: jiwer.wer()  → accuracy_score  │
│  Stage 7: onset_strength std → rhythm    │
│           (words / duration) × 60 → WPM  │
│  Stage 8: RMS silence detection → pauses │
│  Stage 9: compute_criteria_scores()      │
│           compute_overall_score()        │
│  Stage 10: generate_bilingual_evaluation │
└───────────────────────────────────────────┘
        ↓
📊  JSON Response (VI + EN feedback, reports, tips)
```

---

## 2. Stage 1 — Audio Physics & Sampling

### What is audio?

Sound = pressure waves. The microphone converts mechanical vibration into an electrical signal; the sound card digitizes it into a sequence of numbers.

**Sampling rate used: 16,000 Hz (16 kHz)**

```
Sampling process:
Amplitude
  +1.0 │      ╭──╮          ╭──╮
       │    ╭╯    ╰╮      ╭╯    ╰╮
   0.0 │───╯────────╰────╯────────╰─────  ← continuous analog wave
       │
       │    ×    ×    ×    ×    ×    ×    ← sample points (16,000/sec)
  -1.0 │
       └──────────────────────────────────> time

Each × is a float in [-1.0, +1.0]:
[0.0, 0.12, 0.45, 0.78, 0.95, 0.82, 0.54, 0.21, -0.05, -0.3, ...]
```

**Why 16 kHz?** Nyquist-Shannon theorem: to reconstruct frequency F, sample at ≥ 2F. Human speech spans 85 Hz – 8 kHz. 16 kHz covers this exactly while minimizing data size.

**Bit depth:** 16-bit per sample = 65,536 amplitude levels. 1 second of audio = 16,000 × 16 bit = 256 kbit = 32 KB.

---

## 3. Stage 2 — FFT & Spectrogram

### Why not raw waveform?

Raw waveform is too noisy to extract phoneme features directly. FFT decomposes it into constituent frequencies.

### Windowing

Audio is sliced into overlapping frames:

```
5-second file:
│────────────────────────────────────────────────│
│  Frame 1  │  Frame 2  │  Frame 3  │  Frame 4  │  ...
    25ms       25ms         25ms        25ms
  (10ms hop) (10ms hop)  (10ms hop)  (10ms hop)
```

- Frame length: **25ms** (400 samples at 16 kHz)
- Hop length: **10ms** (160 samples) → 15ms overlap between frames

### Hamming Window

Before FFT, each frame is multiplied by a Hamming window to prevent spectral leakage at frame edges:

```
  1.0 │         ╭─────────────╮
      │      ╭─╯               ╰─╮
  0.5 │    ╭─╯                   ╰─╮
  0.0 │──╯                         ╰──
      └────────────────────────────────> time within frame
```

### FFT Result

For each frame, FFT returns amplitude per frequency bin:

```
Amplitude when speaking "A":
  │  ████
  │  ████  █
  │  ████  ███
  │  ████  ██████
  │  ████  ████████  ██
  └──────────────────────────────> Hz
     200  500  750  1200  1700
     ↑    ↑           ↑
  fundamental  F1 formant  F2 formant
```

| Frequency band | Component | MC relevance |
|---|---|---|
| 80–200 Hz | Fundamental (pitch) | Male MC ~100–140 Hz, female ~180–250 Hz |
| 200–800 Hz | F1 formant | Vowel clarity |
| 800–3000 Hz | F2 formant | Vowel distinction |
| 3000–8000 Hz | Sibilants (S, X, H) | Crisp articulation |

---

## 4. Stage 3 — Mel Filterbank

### Problem with linear Hz scale

Human hearing is logarithmic. The perceptual difference between 100 Hz and 200 Hz is far larger than between 7000 Hz and 7100 Hz. If the AI learns from raw Hz bins, it wastes capacity on perceptually irrelevant high-frequency detail.

**Mel conversion:**
```
Mel(f) = 2595 × log₁₀(1 + f / 700)
```

### 80-band Mel Filterbank

80 triangular filters spaced on the Mel scale:

```
80 Mel filters:
Amplitude
  1.0 │ /\ /\ /\ /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
      │/  X  X  X/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
  0.0 └──────────────────────────────────────────> Hz
     0Hz                                      8000Hz
     ← wide filters (perceptually rich) → ← narrow filters →
```

### Mel Spectrogram

Applying 80 filters to each of T frames produces an **80 × T matrix** — a 2D image:

```
Mel Spectrogram (what Whisper "sees"):

Frequency (80 bands)
  ↑ (8000 Hz)
80 │ ░░░░░▓░░░░░░░░░░▓▓░░░░░░░░░░░░  ← high-freq consonants
   │ ░░░▓▓▓▓░░░░░░▓▓▓▓▓░░░░░░░░░░░░
   │ ░▓▓▓▓▓▓▓░░░▓▓▓▓▓▓▓▓░░░░░░░░░░░  ← mid vowel energy
   │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← low fundamental
  1 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  ↓ (80 Hz)
    └──────────────────────────────────> time (T frames)
        C  h  à  o     m  ư  n  g
```

Dark (░) = low energy. Filled (▓) = high energy. This image is the input to Whisper Encoder.

---

## 5. Stage 4 — Whisper STT (Encoder-Decoder)

### Model

OpenAI Whisper `small` — 244M parameters, trained on 680,000 hours of multilingual audio including Vietnamese.

System config: `small` (best speed/accuracy tradeoff for RTX 4060 8GB).

### Transformer Architecture

```
Mel Spectrogram      ┌────────────────────────────────┐
(80 × T)          → │  ENCODER (6 Transformer layers) │
                     │  Self-Attention + FFN per layer  │
                     │  Output: context vectors (T×512) │
                     └──────────────┬─────────────────┘
                                    │
                     ┌──────────────▼─────────────────┐
                     │  DECODER (6 Transformer layers) │
                     │  Masked Self-Attention           │
                     │  Cross-Attention ← Encoder       │
                     │  Output: token probabilities     │
                     └────────────────────────────────┘
                                    ↓
                     "Chào mừng quý vị và các bạn..."
```

**Encoder layers learn:**
- Layers 1–2: Low-level acoustic features (onset transitions, syllable boundaries)
- Layers 3–4: Vietnamese-specific phoneme patterns (tones, vowels)
- Layers 5–6: Sentence-level context and cross-syllable dependencies

**Auto language detection:** Whisper detects language from the first 30s without configuration (`language=None` in code).

---

## 6. Stage 5 — Attention Mechanism

### Self-Attention

For each token position i, Attention computes:

1. **Query (Q):** "What am I looking for?"
2. **Key (K):** "What do I offer?"
3. **Value (V):** "What is my actual content?"

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V   where d_k = 64
```

### Cross-Attention

During decoding, each generated token attends to Encoder outputs — aligning generated text with the corresponding audio frames.

### Example — Vietnamese phrase

When generating token "lại" in "quay trở lại":

```
Attention scores:
  "quay"  ████████████████  0.35  ← high (idiomatic unit)
  "trở"   ███████████░░░░░  0.29  ← high (direct precursor)
  "lại"   ░░░░░░░░░░░░░░░░  0.24  ← self
  "đã"    ░░░░░░░░░░░░░░░░  0.04
  "vị"    ░░░░░░░░░░░░░░░░  0.03
                             ────
                             1.00
```

Even if "trở" is partially masked by noise, the model correctly fills it in because "quay" scores 0.35 — the idiom "quay trở lại" is encoded as a unit.

---

## 7. Stage 6 — Accuracy Scoring (WER + Levenshtein)

### WER (Word Error Rate)

Industry standard metric (IEEE/NIST) for ASR evaluation.

**Three edit operations:**
- **Substitution (S):** wrong word spoken
- **Deletion (D):** word omitted
- **Insertion (I):** extra word added

```
WER = (S + D + I) / N_ref
accuracy_score = max(0, 100 - WER × 100)
```

### Levenshtein Dynamic Programming

Example — Reference: `"chào mừng quý vị"`, Spoken: `"chào mừng quý bạn"`:

```
         ""   chào  mừng  quý   bạn
    ""  [ 0 ] [ 1 ] [ 2 ] [ 3 ] [ 4 ]
   chào [ 1 ] [ 0 ] [ 1 ] [ 2 ] [ 3 ]
   mừng [ 2 ] [ 1 ] [ 0 ] [ 1 ] [ 2 ]
    quý [ 3 ] [ 2 ] [ 1 ] [ 0 ] [ 1 ]
     vị [ 4 ] [ 3 ] [ 2 ] [ 1 ] [ 1 ] ← 1 error (Substitution "bạn" → "vị")

WER = 1/4 = 0.25 → accuracy_score = 75.0
```

**Library:** `jiwer 3.x` — handles text normalization automatically before scoring.

### Common cases

| Scenario | Ref | Spoken | Errors | Accuracy |
|---|---|---|---|---|
| Perfect | "xin chào" | "xin chào" | 0 | 100% |
| Wrong word | "xin chào" | "xin hào" | 1S | 50% |
| Omitted word | "xin chào bạn" | "xin chào" | 1D | 67% |
| Extra words | "xin chào" | "xin chào bạn nhé" | 2I | 0% |
| Completely wrong | "xin chào" | "hello world" | 2S | 0% |

---

## 8. Stage 7 — Rhythm & Pacing Analysis

### Rhythm Score — Onset Strength Standard Deviation

**Onset** = the moment a new sound begins (syllable attack).

```python
onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
# e.g. [0.1, 0.1, 2.5, 1.8, 0.3, 0.1, 3.1, 2.0, ...]
#                  ↑                    ↑
#            syllable start       syllable start

rhythm_raw = float(np.std(onset_env))
rhythm_score = min(100.0, rhythm_raw × 20)
```

Higher standard deviation = more dynamic energy variation = more expressive delivery.

**Comparison:**

```
Flat reader (monotone):
  onset_env: [1.8, 1.9, 1.8, 1.7, 1.9, 1.8]
  std = 0.07 → rhythm_score = 1.4

Expressive MC:
  onset_env: [0.5, 1.0, 4.2, 1.2, 0.4, 3.8, 1.5]
  std = 1.45 → rhythm_score = 29.0
```

**Threshold for feedback:** rhythm_score > 40 = "good dynamic variation".

### Pacing Score — WPM

```python
duration   = librosa.get_duration(y=audio_data, sr=sr)
word_count = len(text_spoken.split())
wpm        = (word_count / duration) * 60
```

**Target ranges:**

| WPM | Assessment | Feedback |
|---|---|---|
| < 100 | Too slow | Risks losing audience attention |
| 100–120 | Slightly slow | Needs more energy |
| 120–150 | Optimal ✓ | Vietnamese MC gold standard |
| 150–180 | Slightly fast | Take deeper breaths between phrases |
| > 180 | Too fast | Audience cannot absorb |

Default target: `target_wpm_min=120`, `target_wpm_max=150` (configurable per lesson).

**Pacing score formula:**

```python
if target_min ≤ wpm ≤ target_max:
    pacing_score = 100.0
else:
    distance = abs(wpm - nearest_bound)
    penalty  = min(100, (distance / range_span) × 100)
    pacing_score = max(0, 100 - penalty)
```

---

## 9. Stage 8 — Pause Detection

RMS (Root Mean Square) energy computed at frame level:

```python
frame_length = int(0.025 * sr)   # 25ms frames
hop_length   = int(0.010 * sr)   # 10ms hop
rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

silence_threshold = mean(rms) × 0.05   # frames below 5% of mean = silence
```

Contiguous silent frames ≥ 150ms are recorded as a pause.

**Output:**
```json
{
  "avg_pause_sec": 0.52,
  "max_pause_sec": 1.1,
  "pause_count": 5
}
```

**Feedback rules:**

| Avg pause | Assessment |
|---|---|
| < 0.4s | Too short — "nghỉ 0.5–0.8s giữa câu" |
| 0.4–0.9s | Good timing |
| > 0.9s | Too long — maintain momentum |

---

## 10. Stage 9 — Criteria Scoring & Overall Score

### Aspect mapping

```python
aspect_map = {
    "PRONUNCIATION": accuracy_score,
    "ACCURACY":      accuracy_score,
    "RHYTHM":        normalized_rhythm,
    "EMOTION":       min(100, normalized_rhythm × 1.1),   # rhythm-derived + boost
    "PACING":        pacing_score,
}
```

### Weighted overall score

```python
overall = Σ(criteria[i].score × criteria[i].weight) / Σ(weights)
```

Example with standard lesson criteria:

```
PRONUNCIATION  weight=30  score=87.5  → 87.5 × 30 = 2625
RHYTHM         weight=30  score=42.3  → 42.3 × 30 = 1269
PACING         weight=20  score=100.0 → 100.0 × 20 = 2000
EMOTION        weight=20  score=46.5  → 46.5 × 20 =  930

Σ weights = 100
overall = (2625 + 1269 + 2000 + 930) / 100 = 68.24
```

Falls back to simple mean when no criteria provided by the frontend.

---

## 11. Stage 10 — Bilingual Expert Feedback

`generate_bilingual_evaluation()` applies rule-based heuristics to produce structured feedback in both Vietnamese and English.

### Output structure

| Field | Type | Description |
|---|---|---|
| `feedback_vi` | string | Pipe-separated summary (pronunciation \| rhythm \| pausing) |
| `feedback_en` | string | Same in English |
| `tips_vi` | array | `[{label, tip}]` — pacing + low-score criteria warnings |
| `tips_en` | array | Same in English |
| `report_vi` | string | Full Markdown report with tables and action plan |
| `report_en` | string | Same in English |

### Action plan rules

| Condition | Action plan |
|---|---|
| accuracy < 80 | Over-enunciation drill — exaggerate final consonants |
| 80 ≤ accuracy < 90 | Tonal refinement — complex tonal transitions |
| accuracy ≥ 90 | Clarity maintenance — hold sharpness while increasing speed |
| rhythm < 35 | High-Low Stress technique — underline and stress key words |
| 35 ≤ rhythm < 60 | Emotional layering — add inflection to descriptive adjectives |
| rhythm ≥ 60 | Energy management — sustain level through long passages |
| wpm > target_max | Silence management — count "1" at commas, "1,2" at periods |
| wpm < target_min | Flow & Momentum — reduce micro-pauses between words |
| wpm in range | Strategic Pause — 0.5s silence before key information |

---

## 12. End-to-End Example

**Script:** *"Chào mừng quý vị và các bạn đã đến với chương trình hôm nay"*  
**User reads:** Correctly, but monotone and slightly fast.

---

**Stage 1 — Sampling:**  
3.5s recording → 3.5 × 16000 = 56,000 sample points

**Stage 2 — FFT:**  
349 frames (25ms length, 10ms hop)

**Stage 3 — Mel:**  
80 × 349 spectrogram matrix

**Stage 4–5 — Whisper:**  
Decodes to: `"Chào mừng quý vị và các bạn đã đến với chương trình hôm nay"`

**Stage 6 — WER:**
```
Ref (14 words): "chào mừng quý vị và các bạn đã đến với chương trình hôm nay"
Hyp (14 words): "chào mừng quý vị và các bạn đã đến với chương trình hôm nay"
Errors: 0  →  accuracy_score = 100.0
```

**Stage 7 — Rhythm:**
```
onset_env = [0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.9, ...]   ← very flat
std = 0.05  →  rhythm_score = 1.0   ← monotone
```

**Stage 7 — WPM:**
```
word_count = 14, duration = 3.5s
wpm = (14 / 3.5) × 60 = 240 WPM  ← too fast
pacing_score = max(0, 100 - ((240-150)/30)×100) = 0.0
```

**Stage 8 — Pauses:**
```
pause_count = 0, avg_pause_sec = 0.0   ← no breathing breaks
```

**Stage 9 — Overall:**
```
PRONUNCIATION: 100.0  (weight 30)
RHYTHM:          1.0  (weight 30)
PACING:          0.0  (weight 20)
EMOTION:         1.1  (weight 20)

overall = (100×30 + 1×30 + 0×20 + 1.1×20) / 100 = 30.52
```

**Stage 10 — Feedback:**
```json
{
  "feedback_vi": "Phát âm: Độ chính xác tuyệt vời. | Nhấn nhá: Giọng còn hơi đều — hãy thử nhấn mạnh vào các từ khóa.",
  "tips_vi": [
    {"label": "TỐC ĐỘ", "tip": "Tốc độ nhanh (240 WPM). Hãy nói chậm lại ở những đoạn quan trọng."},
    {"label": "RHYTHM",  "tip": "Tiêu chí RHYTHM đạt 1/100 — cần cải thiện thêm."}
  ]
}
```

**Diagnosis:** Speaker read the script word-perfect, but at 240 WPM (nearly 2× the 120–150 WPM target) with zero dynamic variation. Robot-mode delivery despite 100% accuracy.

---

## 13. Technical Spec Table

| Parameter | Value | Scientific basis |
|:---|:---|:---|
| Sampling rate | 16,000 Hz | Nyquist — covers full 0–8 kHz speech range |
| Frame length | 25ms (400 samples) | Long enough for acoustic features, short enough to track change |
| Frame hop | 10ms (160 samples) | 15ms overlap preserves boundary information |
| Windowing | Hamming | Reduces spectral leakage at frame edges |
| FFT size | 400 | Frequency resolution = 16000/400 = 40 Hz/bin |
| Mel filters | 80 bands | Models 80 perceptual frequency channels |
| Mel range | 0–8000 Hz | Covers full human speech range |
| Whisper model | small (244M params) | Best speed/accuracy for RTX 4060 8GB |
| Whisper layers | 6 Encoder + 6 Decoder | Sufficient depth for Vietnamese tonal language |
| Attention heads | 8 per layer | Learns 8 independent relationship types per token |
| Embedding dim | 512 | Representation space dimension |
| WER algorithm | Levenshtein DP | O(m×n) complexity — exact minimum edit distance |
| WER library | jiwer 3.x | Includes automatic text normalization |
| Onset detection | `librosa.onset.onset_strength` | Measures Spectral Flux (energy change rate) |
| Rhythm metric | Standard deviation of onset | Measures spread of energy peaks — monotone = low std |
| Rhythm normalization | min(100, σ × 20) | Maps typical σ range (0–5) to 0–100 scale |
| WPM formula | (words / duration) × 60 | Standard words-per-minute calculation |
| Silence threshold | 5% of mean RMS | Conservative — avoids classifying quiet speech as silence |
| Min pause | 150ms | Below this = not a meaningful breath pause |
| Optimal pause | 0.5–0.8s | Vietnamese MC delivery standard |
| API framework | FastAPI + uvicorn | ASGI async — non-blocking for concurrent requests |
| GPU inference | CUDA amp autocast | Mixed-precision FP16 — halves VRAM for same quality |

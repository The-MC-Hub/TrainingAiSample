# Tài Liệu Kỹ Thuật — Pipeline Phân Tích Giọng Nói AI

> Tài liệu kỹ thuật đầy đủ cho hệ thống phân tích giọng nói AI của **The MC Hub**.  
> Mô tả toàn bộ vòng đời dữ liệu: thu âm từ microphone → chuyển giọng nói thành văn bản (Whisper) → chấm điểm → phản hồi chuyên gia song ngữ.

---

## Mục Lục

1. [Tổng Quan Pipeline](#1-tổng-quan-pipeline)
2. [Giai Đoạn 1 — Thu Âm & Lấy Mẫu](#2-giai-đoạn-1--thu-âm--lấy-mẫu)
3. [Giai Đoạn 2 — FFT & Spectrogram](#3-giai-đoạn-2--fft--spectrogram)
4. [Giai Đoạn 3 — Mel Filterbank](#4-giai-đoạn-3--mel-filterbank)
5. [Giai Đoạn 4 — Whisper STT (Encoder-Decoder)](#5-giai-đoạn-4--whisper-stt-encoder-decoder)
6. [Giai Đoạn 5 — Cơ Chế Attention](#6-giai-đoạn-5--cơ-chế-attention)
7. [Giai Đoạn 6 — Chấm Điểm Độ Chính Xác (WER + Levenshtein)](#7-giai-đoạn-6--chấm-điểm-độ-chính-xác-wer--levenshtein)
8. [Giai Đoạn 7 — Phân Tích Nhịp Điệu & Tốc Độ](#8-giai-đoạn-7--phân-tích-nhịp-điệu--tốc-độ)
9. [Giai Đoạn 8 — Phát Hiện Ngắt Nghỉ](#9-giai-đoạn-8--phát-hiện-ngắt-nghỉ)
10. [Giai Đoạn 9 — Điểm Cảm Xúc Giọng Nói (Composite Acoustic)](#10-giai-đoạn-9--điểm-cảm-xúc-giọng-nói-composite-acoustic)
11. [Giai Đoạn 10 — Tổng Hợp Điểm Tiêu Chí & Điểm Tổng](#11-giai-đoạn-10--tổng-hợp-điểm-tiêu-chí--điểm-tổng)
12. [Giai Đoạn 11 — Phản Hồi Chuyên Gia Song Ngữ](#12-giai-đoạn-11--phản-hồi-chuyên-gia-song-ngữ)
13. [Ví Dụ Đầu Cuối](#13-ví-dụ-đầu-cuối)
14. [Bảng Thông Số Kỹ Thuật](#14-bảng-thông-số-kỹ-thuật)
15. [Module TTS — Tổng Hợp Giọng Nói MC (Supertonic)](#15-module-tts--tổng-hợp-giọng-nói-mc-supertonic)

---

## 1. Tổng Quan Pipeline

```
🎙️  File âm thanh (webm/wav/mp3)
        ↓
┌─────────────────────────────────────────────────┐
│  Giai đoạn 1: librosa.load() → PCM 16kHz        │
│  Giai đoạn 2: FFT (cửa sổ Hamming, bước 25ms)  │
│  Giai đoạn 3: Mel Filterbank 80 dải             │
│  Giai đoạn 4: Whisper Encoder (6 lớp)           │
│  Giai đoạn 5: Whisper Decoder → text_spoken     │
└─────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────┐
│  Giai đoạn 6: jiwer.wer()  → accuracy_score    │
│  Giai đoạn 7: onset_strength std → nhịp điệu   │
│               (từ / thời gian) × 60 → WPM       │
│  Giai đoạn 8: phát hiện khoảng lặng RMS         │
│  Giai đoạn 9: compute_emotion_score()           │
│               pitch + energy + tempo composite  │
│  Giai đoạn 10: compute_criteria_scores()        │
│                compute_overall_score()          │
│  Giai đoạn 11: generate_bilingual_evaluation()  │
└─────────────────────────────────────────────────┘
        ↓
📊  Phản hồi JSON (tiếng Việt + tiếng Anh, báo cáo, gợi ý)
```

---

## 2. Giai Đoạn 1 — Thu Âm & Lấy Mẫu

### Âm thanh là gì?

Âm thanh là sóng áp suất không khí. Microphone chuyển dao động cơ học thành tín hiệu điện; card âm thanh số hóa thành dãy số.

**Tần số lấy mẫu sử dụng: 16.000 Hz (16 kHz)**

```
Quá trình lấy mẫu:
Biên độ
  +1.0 │      ╭──╮          ╭──╮
       │    ╭╯    ╰╮      ╭╯    ╰╮
   0.0 │───╯────────╰────╯────────╰─────  ← sóng analog liên tục
       │
       │    ×    ×    ×    ×    ×    ×    ← điểm lấy mẫu (16.000/giây)
  -1.0 │
       └──────────────────────────────────> thời gian

Mỗi × là một số thực trong [-1.0, +1.0]:
[0.0, 0.12, 0.45, 0.78, 0.95, 0.82, 0.54, 0.21, -0.05, -0.3, ...]
```

**Tại sao 16 kHz?** Định lý Nyquist-Shannon: để tái tạo tần số F, cần lấy mẫu ≥ 2F. Giọng người trải dài 85 Hz – 8 kHz. 16 kHz bao phủ đủ trong khi giảm tối đa dung lượng dữ liệu.

**Độ sâu bit:** 16-bit mỗi mẫu = 65.536 mức biên độ. 1 giây âm thanh = 16.000 × 16 bit = 256 kbit = 32 KB.

---

## 3. Giai Đoạn 2 — FFT & Spectrogram

### Tại sao không dùng dạng sóng thô?

Dạng sóng thô quá nhiễu để trích xuất đặc trưng âm vị trực tiếp. FFT phân rã thành các tần số thành phần.

### Phân khung (Windowing)

Âm thanh được cắt thành các khung chồng lên nhau:

```
File 5 giây:
│────────────────────────────────────────────────│
│  Khung 1  │  Khung 2  │  Khung 3  │  Khung 4  │  ...
    25ms        25ms         25ms        25ms
  (bước 10ms) (bước 10ms) (bước 10ms) (bước 10ms)
```

- Độ dài khung: **25ms** (400 mẫu tại 16 kHz)
- Bước nhảy: **10ms** (160 mẫu) → chồng lấp 15ms giữa các khung

### Cửa Sổ Hamming

Trước FFT, mỗi khung được nhân với cửa sổ Hamming để tránh rò rỉ phổ tại biên khung:

```
  1.0 │         ╭─────────────╮
      │      ╭─╯               ╰─╮
  0.5 │    ╭─╯                   ╰─╮
  0.0 │──╯                         ╰──
      └────────────────────────────────> thời gian trong khung
```

### Kết Quả FFT

Với mỗi khung, FFT trả về biên độ theo từng dải tần:

```
Biên độ khi nói "A":
  │  ████
  │  ████  █
  │  ████  ███
  │  ████  ██████
  │  ████  ████████  ██
  └──────────────────────────────> Hz
     200  500  750  1200  1700
     ↑    ↑           ↑
  cơ bản  formant F1  formant F2
```

| Dải tần | Thành phần | Ý nghĩa với MC |
|---|---|---|
| 80–200 Hz | Tần số cơ bản (cao độ) | MC nam ~100–140 Hz, nữ ~180–250 Hz |
| 200–800 Hz | Formant F1 | Độ rõ nguyên âm |
| 800–3000 Hz | Formant F2 | Phân biệt nguyên âm |
| 3000–8000 Hz | Phụ âm sắc (S, X, H) | Phát âm rõ nét |

---

## 4. Giai Đoạn 3 — Mel Filterbank

### Vấn Đề Với Thang Hz Tuyến Tính

Thính giác người là logarithm. Khoảng cách cảm nhận giữa 100 Hz và 200 Hz lớn hơn nhiều so với 7000 Hz và 7100 Hz. Nếu AI học từ các bin Hz thô, nó lãng phí năng lực cho chi tiết tần số cao không liên quan về mặt cảm giác.

**Công thức chuyển đổi Mel:**
```
Mel(f) = 2595 × log₁₀(1 + f / 700)
```

### Mel Filterbank 80 Dải

80 bộ lọc tam giác phân bố theo thang Mel:

```
80 bộ lọc Mel:
Biên độ
  1.0 │ /\ /\ /\ /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
      │/  X  X  X/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
  0.0 └──────────────────────────────────────────> Hz
     0Hz                                      8000Hz
     ← bộ lọc rộng (giàu thông tin) → ← bộ lọc hẹp →
```

### Mel Spectrogram

Áp dụng 80 bộ lọc lên T khung tạo ra **ma trận 80 × T** — ảnh 2D:

```
Mel Spectrogram (những gì Whisper "nhìn thấy"):

Tần số (80 dải)
  ↑ (8000 Hz)
80 │ ░░░░░▓░░░░░░░░░░▓▓░░░░░░░░░░░░  ← phụ âm tần cao
   │ ░░░▓▓▓▓░░░░░░▓▓▓▓▓░░░░░░░░░░░░
   │ ░▓▓▓▓▓▓▓░░░▓▓▓▓▓▓▓▓░░░░░░░░░░░  ← năng lượng nguyên âm trung
   │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← tần số cơ bản thấp
  1 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  ↓ (80 Hz)
    └──────────────────────────────────> thời gian (T khung)
        C  h  à  o     m  ư  n  g
```

Tối (░) = năng lượng thấp. Đặc (▓) = năng lượng cao. Ảnh này là đầu vào của Whisper Encoder.

---

## 5. Giai Đoạn 4 — Whisper STT (Encoder-Decoder)

### Mô Hình

OpenAI Whisper — được huấn luyện trên 680.000 giờ âm thanh đa ngôn ngữ bao gồm tiếng Việt.

Hệ thống tự động chọn model theo phần cứng:
- GPU ≥ 6GB VRAM → `large-v3`
- GPU 4–6GB VRAM → `medium`
- CPU hoặc VRAM thấp → `small`

Có thể ghi đè bằng biến môi trường `WHISPER_MODEL=<tên>` trong file `.env`.

### Kiến Trúc Transformer

```
Mel Spectrogram      ┌──────────────────────────────────┐
(80 × T)          → │  ENCODER (6 lớp Transformer)      │
                     │  Self-Attention + FFN mỗi lớp     │
                     │  Đầu ra: vector ngữ cảnh (T×512)  │
                     └──────────────┬───────────────────┘
                                    │
                     ┌──────────────▼───────────────────┐
                     │  DECODER (6 lớp Transformer)      │
                     │  Masked Self-Attention             │
                     │  Cross-Attention ← Encoder        │
                     │  Đầu ra: xác suất token           │
                     └──────────────────────────────────┘
                                    ↓
                     "Chào mừng quý vị và các bạn..."
```

**Encoder học:**
- Lớp 1–2: Đặc trưng âm học cơ bản (chuyển tiếp âm đầu, ranh giới âm tiết)
- Lớp 3–4: Mẫu âm vị tiếng Việt (thanh điệu, nguyên âm)
- Lớp 5–6: Ngữ cảnh cấp câu và phụ thuộc liên âm tiết

**Tự nhận diện ngôn ngữ:** Whisper phát hiện ngôn ngữ từ 30 giây đầu mà không cần cấu hình (`language=None` trong code).

---

## 6. Giai Đoạn 5 — Cơ Chế Attention

### Self-Attention

Tại mỗi vị trí token i, Attention tính toán:

1. **Query (Q):** "Tôi đang tìm kiếm gì?"
2. **Key (K):** "Tôi cung cấp gì?"
3. **Value (V):** "Nội dung thực của tôi là gì?"

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V   với d_k = 64
```

### Cross-Attention

Trong quá trình giải mã, mỗi token được tạo ra sẽ "chú ý" đến đầu ra của Encoder — căn chỉnh văn bản được tạo với các khung âm thanh tương ứng.

### Ví Dụ — Cụm Từ Tiếng Việt

Khi tạo token "lại" trong "quay trở lại":

```
Điểm Attention:
  "quay"  ████████████████  0.35  ← cao (đơn vị thành ngữ)
  "trở"   ███████████░░░░░  0.29  ← cao (từ đứng trước trực tiếp)
  "lại"   ░░░░░░░░░░░░░░░░  0.24  ← tự tham chiếu
  "đã"    ░░░░░░░░░░░░░░░░  0.04
  "vị"    ░░░░░░░░░░░░░░░░  0.03
                             ────
                             1.00
```

Dù "trở" bị nhiễu che khuất một phần, model vẫn điền đúng vì "quay" đạt 0.35 — thành ngữ "quay trở lại" được mã hóa thành một đơn vị.

---

## 7. Giai Đoạn 6 — Chấm Điểm Độ Chính Xác (WER + Levenshtein)

### WER (Tỉ Lệ Lỗi Từ)

Tiêu chuẩn ngành (IEEE/NIST) để đánh giá hệ thống nhận diện giọng nói.

**Ba phép chỉnh sửa:**
- **Thay thế (S):** nói sai từ
- **Xóa (D):** bỏ sót từ
- **Chèn (I):** thêm từ thừa

```
WER = (S + D + I) / N_tham_chiếu
accuracy_score = max(0, 100 - WER × 100)
```

### Quy Hoạch Động Levenshtein

Ví dụ — Kịch bản: `"chào mừng quý vị"`, Người dùng đọc: `"chào mừng quý bạn"`:

```
          ""   chào  mừng  quý   bạn
     ""  [ 0 ] [ 1 ] [ 2 ] [ 3 ] [ 4 ]
    chào [ 1 ] [ 0 ] [ 1 ] [ 2 ] [ 3 ]
    mừng [ 2 ] [ 1 ] [ 0 ] [ 1 ] [ 2 ]
     quý [ 3 ] [ 2 ] [ 1 ] [ 0 ] [ 1 ]
      vị [ 4 ] [ 3 ] [ 2 ] [ 1 ] [ 1 ] ← 1 lỗi (thay "bạn" → "vị")

WER = 1/4 = 0.25 → accuracy_score = 75.0
```

**Thư viện:** `jiwer 3.x` — tự động chuẩn hóa văn bản trước khi chấm điểm.

### Các Trường Hợp Thường Gặp

| Tình huống | Kịch bản | Đọc thực | Lỗi | Điểm |
|---|---|---|---|---|
| Hoàn hảo | "xin chào" | "xin chào" | 0 | 100% |
| Sai từ | "xin chào" | "xin hào" | 1S | 50% |
| Bỏ từ | "xin chào bạn" | "xin chào" | 1D | 67% |
| Thêm từ | "xin chào" | "xin chào bạn nhé" | 2I | 0% |
| Sai hoàn toàn | "xin chào" | "hello world" | 2S | 0% |

---

## 8. Giai Đoạn 7 — Phân Tích Nhịp Điệu & Tốc Độ

### Điểm Nhịp Điệu — Độ Lệch Chuẩn Onset Strength

**Onset** = thời điểm bắt đầu một âm thanh mới (điểm tấn công âm tiết).

```python
onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
# ví dụ: [0.1, 0.1, 2.5, 1.8, 0.3, 0.1, 3.1, 2.0, ...]
#                    ↑                    ↑
#             bắt đầu âm tiết       bắt đầu âm tiết

rhythm_raw = float(np.std(onset_env))
rhythm_score = min(100.0, rhythm_raw × 20)
```

Độ lệch chuẩn cao = biến thiên năng lượng mạnh = trình bày biểu cảm hơn.

**So sánh:**

```
Đọc đều đều (đơn điệu):
  onset_env: [1.8, 1.9, 1.8, 1.7, 1.9, 1.8]
  std = 0.07 → rhythm_score = 1.4

MC biểu cảm:
  onset_env: [0.5, 1.0, 4.2, 1.2, 0.4, 3.8, 1.5]
  std = 1.45 → rhythm_score = 29.0
```

**Ngưỡng phản hồi:** rhythm_score > 40 = "biến thiên năng lượng tốt".

### Điểm Tốc Độ — WPM (Từ Mỗi Phút)

```python
# Trừ tổng thời gian ngắt nghỉ để tính WPM chính xác hơn
duration_speech = duration_total - total_pause_sec
wpm = (word_count / duration_speech) * 60
```

**Ngưỡng mục tiêu:**

| WPM | Đánh giá | Phản hồi |
|---|---|---|
| < 100 | Quá chậm | Nguy cơ mất sự tập trung của khán giả |
| 100–120 | Hơi chậm | Cần thêm năng lượng |
| 120–150 | Lý tưởng ✓ | Tiêu chuẩn vàng của MC tiếng Việt |
| 150–180 | Hơi nhanh | Hít thở sâu hơn giữa các câu |
| > 180 | Quá nhanh | Khán giả không thể tiếp thu |

Mặc định: `target_wpm_min=120`, `target_wpm_max=150` (có thể cấu hình theo từng bài học).

**Công thức điểm tốc độ:**

```python
if target_min ≤ wpm ≤ target_max:
    pacing_score = 100.0
else:
    khoảng_cách = abs(wpm - giới_hạn_gần_nhất)
    phạt = min(100, (khoảng_cách / độ_rộng_dải) × 100)
    pacing_score = max(0, 100 - phạt)
```

---

## 9. Giai Đoạn 8 — Phát Hiện Ngắt Nghỉ

Năng lượng RMS (Root Mean Square) tính theo từng khung:

```python
frame_length = int(0.025 * sr)   # khung 25ms
hop_length   = int(0.010 * sr)   # bước 10ms
rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

ngưỡng_im_lặng = mean(rms) × 0.05   # khung dưới 5% trung bình = im lặng
```

Các khung im lặng liên tiếp ≥ 150ms được ghi lại là một khoảng ngắt.

**Đầu ra:**
```json
{
  "avg_pause_sec": 0.52,
  "max_pause_sec": 1.1,
  "pause_count": 5,
  "total_pause_sec": 2.6
}
```

**Quy tắc phản hồi:**

| Ngắt trung bình | Đánh giá |
|---|---|
| < 0.4s | Quá ngắn — cần nghỉ 0.4–0.9s giữa các câu |
| 0.4–0.9s | Nhịp điệu tốt |
| > 0.9s | Quá dài — duy trì đà nói |

---

## 10. Giai Đoạn 9 — Điểm Cảm Xúc Giọng Nói (Composite Acoustic)

### Tại Sao Dùng Acoustic Proxy Thay Vì AI Cảm Xúc?

Phân loại cảm xúc thật sự (vui/buồn/tức giận) đòi hỏi model ML được huấn luyện trên tập dữ liệu giọng nói có nhãn cảm xúc (ví dụ: IEMOCAP ~2GB). Các model này quá nặng cho môi trường CPU hiện tại.

Thay vào đó, hệ thống đo **3 tín hiệu âm học tương quan với trình bày biểu cảm** — những đặc trưng mà huấn luyện viên giọng nói chuyên nghiệp thực sự dùng để luyện MC.

### Công Thức Tổng Hợp Ba Thành Phần

```
emotion_score = pitch_score      × 0.50   (biến thiên cao độ)
              + energy_std_score × 0.30   (biến thiên âm lượng)
              + tempo_var_score  × 0.20   (biến thiên nhịp độ)
```

---

#### Thành Phần 1 — Biến Thiên Cao Độ F0 (trọng số 50%)

Tần số cơ bản (F0) được trích xuất bằng thuật toán `librosa.pyin` (probabilistic YIN).

```
F0 (Hz) → chuyển sang semitone: s = 12 × log₂(f₀ / 440)
pitch_std = độ_lệch_chuẩn(s của các khung có giọng)
```

Chuyển sang semitone giúp chỉ số **độc lập với người nói** — cùng giá trị std nghĩa là cùng mức biến thiên cao độ dù giọng nam hay nữ.

```
Thang điểm pitch_std:
  < 1.0 semitone  → đơn điệu          0–25
  1–2 semitones   → hơi biến thiên    25–55
  2–4 semitones   → dải MC tốt        55–85
  4+ semitones    → rất biểu cảm      85–100
```

**1 semitone nghĩa là gì:**
```
1 semitone = tỉ lệ 2^(1/12) ≈ 1.059 lần tần số
Ví dụ: nói "xin" ở 150 Hz rồi "chào" ở 159 Hz = cách 1 semitone
Chuẩn MC biểu cảm = độ lệch chuẩn ≥ 3 semitones (theo tiêu chuẩn huấn luyện giọng)
```

---

#### Thành Phần 2 — Biến Thiên Âm Lượng RMS (trọng số 30%)

RMS tính mỗi khung 25ms, sau đó đo độ lệch chuẩn toàn bộ khung.

```python
rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=160)[0]
energy_std_raw   = std(rms)
energy_std_score = clip(energy_std_raw × 1500, 0, 100)
```

**Tín hiệu này đo gì:** Sự thay đổi âm lượng linh hoạt — nhấn mạnh từ khóa, hạ giọng tạo kịch tính, tăng dần đến cao trào. MC phẳng gần như không có biến thiên RMS; MC biểu cảm có biên độ dao động lớn.

```
Đọc phẳng:
  RMS: [0.08, 0.08, 0.09, 0.08, 0.09]  std=0.004  điểm=6
MC biểu cảm:
  RMS: [0.04, 0.12, 0.25, 0.09, 0.05]  std=0.082  điểm=100
```

---

#### Thành Phần 3 — Biến Thiên Nhịp Độ (trọng số 20%)

```python
tempo, beats    = librosa.beat.beat_track(y=audio, sr=sr)
beat_intervals  = diff(frames_to_time(beats))
tempo_var_raw   = std(beat_intervals)
tempo_var_score = clip(tempo_var_raw × 300, 0, 100)
```

**Tín hiệu này đo gì:** Sự thay đổi nhịp điệu — chậm lại ở câu mang tính cảm xúc nặng, tăng tốc ở đoạn chuyển tiếp năng lượng. MC đọc đều nhịp như máy đánh điểm gần 0; MC lành nghề với thay đổi tốc độ có chủ đích đánh điểm cao.

---

### Cấu Trúc Đầu Ra

```json
{
  "emotion_score": 62.4,
  "pitch_score": 71.2,
  "energy_std_score": 48.3,
  "tempo_var_score": 55.1
}
```

`emotion_score` = tiêu chí `EMOTION` trong `criteria_scores`.  
`emotion_breakdown` được trả về trong API response để frontend hiển thị chi tiết.

---

### So Sánh: Acoustic Proxy vs AI Cảm Xúc ML

| Tính năng | Acoustic (hiện tại) | Model ML cảm xúc |
|---|---|---|
| Phát hiện vui/buồn/tức giận | ✗ | ✓ |
| Độc lập với người nói | ✓ | ✓ |
| Chạy offline, CPU | ✓ | ✗ (cần GPU) |
| Gợi ý luyện tập cụ thể | ✓ | Khó hơn |
| Kích thước model bổ sung | 0 MB | ~300MB–2GB |

---

## 11. Giai Đoạn 10 — Tổng Hợp Điểm Tiêu Chí & Điểm Tổng

### Ánh Xạ Tiêu Chí

```python
aspect_map = {
    "PRONUNCIATION": accuracy_score,          # điểm WER
    "ACCURACY":      accuracy_score,          # alias
    "RHYTHM":        onset_variation_score,   # độ lệch chuẩn onset energy
    "EMOTION":       emotion_score,           # tổng hợp: cao độ + âm lượng + nhịp độ
    "PACING":        pacing_score,            # điểm WPM
}
```

### Điểm Tổng Có Trọng Số

```python
overall = Σ(criteria[i].score × criteria[i].weight) / Σ(weights)
```

Ví dụ với tiêu chí bài học tiêu chuẩn:

```
PRONUNCIATION  trọng_số=30  điểm=87.5  → 87.5 × 30 = 2625
RHYTHM         trọng_số=30  điểm=42.3  → 42.3 × 30 = 1269
PACING         trọng_số=20  điểm=100.0 → 100.0 × 20 = 2000
EMOTION        trọng_số=20  điểm=62.4  → 62.4 × 20 = 1248

Σ trọng_số = 100
overall = (2625 + 1269 + 2000 + 1248) / 100 = 71.42
```

Khi frontend không cung cấp tiêu chí, hệ thống dùng trung bình đơn giản.

---

## 12. Giai Đoạn 11 — Phản Hồi Chuyên Gia Song Ngữ

`generate_bilingual_evaluation()` áp dụng heuristic dựa trên quy tắc để tạo phản hồi có cấu trúc bằng tiếng Việt và tiếng Anh.

### Cấu Trúc Đầu Ra

| Trường | Kiểu | Mô tả |
|---|---|---|
| `feedback_vi` | string | Tóm tắt phân cách bằng dấu pipe (phát âm \| giọng điệu \| ngắt nghỉ) |
| `feedback_en` | string | Tương tự bằng tiếng Anh |
| `tips_vi` | array | `[{label, tip}]` — cảnh báo tốc độ + tiêu chí điểm thấp |
| `tips_en` | array | Tương tự bằng tiếng Anh |
| `report_vi` | string | Báo cáo Markdown đầy đủ với bảng và kế hoạch hành động |
| `report_en` | string | Tương tự bằng tiếng Anh |

### Quy Tắc Kế Hoạch Hành Động

| Điều kiện | Kế hoạch hành động |
|---|---|
| accuracy < 80 | Luyện phát âm cường điệu — nhấn mạnh quá mức phụ âm cuối |
| 80 ≤ accuracy < 90 | Tinh chỉnh thanh điệu — chuyển tiếp thanh phức tạp |
| accuracy ≥ 90 | Duy trì độ rõ — giữ sắc nét khi tăng tốc |
| rhythm < 35 | Kỹ thuật Cao-Thấp — gạch chân và nhấn từ khóa |
| 35 ≤ rhythm < 60 | Lớp cảm xúc — thêm ngữ điệu cho tính từ mô tả |
| rhythm ≥ 60 | Quản lý năng lượng — duy trì mức độ qua đoạn dài |
| wpm > target_max | Quản lý khoảng lặng — đếm "1" tại dấu phẩy, "1,2" tại dấu chấm |
| wpm < target_min | Dòng chảy & Đà nói — giảm micro-pause giữa các từ |
| wpm trong dải | Ngắt có chủ đích — 0.5s im lặng trước thông tin quan trọng |
| energy_std < 35 | Luyện thay đổi âm lượng — to nhỏ theo từng đoạn câu |
| tempo_var < 35 | Luyện nhịp độ linh hoạt — chậm lại ở câu quan trọng, nhanh hơn ở chuyển tiếp |

---

## 13. Ví Dụ Đầu Cuối

**Kịch bản:** *"Chào mừng quý vị và các bạn đã đến với chương trình hôm nay"*  
**Người dùng đọc:** Chính xác hoàn toàn, nhưng đơn điệu và hơi nhanh.

---

**Giai đoạn 1 — Lấy mẫu:**  
Thu âm 3.5 giây → 3.5 × 16.000 = 56.000 điểm mẫu

**Giai đoạn 2 — FFT:**  
349 khung (độ dài 25ms, bước 10ms)

**Giai đoạn 3 — Mel:**  
Ma trận spectrogram 80 × 349

**Giai đoạn 4–5 — Whisper:**  
Giải mã thành: `"Chào mừng quý vị và các bạn đã đến với chương trình hôm nay"`

**Giai đoạn 6 — WER:**
```
Tham chiếu (14 từ): "chào mừng quý vị và các bạn đã đến với chương trình hôm nay"
Thực tế  (14 từ):  "chào mừng quý vị và các bạn đã đến với chương trình hôm nay"
Lỗi: 0  →  accuracy_score = 100.0
```

**Giai đoạn 7 — Nhịp điệu:**
```
onset_env = [0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.9, ...]  ← rất phẳng
std = 0.05  →  rhythm_score = 1.0  ← đơn điệu
```

**Giai đoạn 7 — WPM:**
```
word_count = 14, duration_speech = 3.5s
wpm = (14 / 3.5) × 60 = 240 WPM  ← quá nhanh
pacing_score = max(0, 100 - ((240-150)/30)×100) = 0.0
```

**Giai đoạn 8 — Ngắt nghỉ:**
```
pause_count = 0, avg_pause_sec = 0.0  ← không có khoảng thở
```

**Giai đoạn 9 — Cảm xúc:**
```
pitch_std = 0.4 semitone  → pitch_score = 10.0
energy_std_raw = 0.003    → energy_std_score = 4.5
tempo_var_raw = 0.02      → tempo_var_score = 6.0

emotion_score = 10.0×0.5 + 4.5×0.3 + 6.0×0.2 = 5.0 + 1.35 + 1.2 = 7.55
```

**Giai đoạn 10 — Điểm tổng:**
```
PRONUNCIATION: 100.0  (trọng số 30)
RHYTHM:          1.0  (trọng số 30)
PACING:          0.0  (trọng số 20)
EMOTION:         7.6  (trọng số 20)

overall = (100×30 + 1×30 + 0×20 + 7.6×20) / 100 = 32.52
```

**Giai đoạn 11 — Phản hồi:**
```json
{
  "feedback_vi": "Phát âm: Độ chính xác tuyệt vời. | Giọng điệu: Giọng đơn điệu — cần luyện biến thiên cao độ nhiều hơn.",
  "tips_vi": [
    {"label": "TỐC ĐỘ", "tip": "Tốc độ nhanh (240 WPM). Hãy nói chậm lại ở những đoạn quan trọng."},
    {"label": "NĂNG LƯỢNG", "tip": "Mức năng lượng giọng đều đều — thử tăng/giảm to nhỏ theo từng đoạn câu."},
    {"label": "NHỊP ĐỘ", "tip": "Nhịp nói quá đều — thử chậm lại ở câu quan trọng, nhanh hơn ở đoạn chuyển tiếp."}
  ]
}
```

**Chẩn đoán:** Người dùng đọc kịch bản hoàn hảo từng từ, nhưng ở tốc độ 240 WPM (gần 2× so với mục tiêu 120–150 WPM) với gần như không có biến thiên động. Trình bày kiểu robot dù đạt 100% độ chính xác.

---

## 14. Bảng Thông Số Kỹ Thuật

| Thông số | Giá trị | Cơ sở khoa học |
|:---|:---|:---|
| Tần số lấy mẫu | 16.000 Hz | Nyquist — bao phủ đủ dải 0–8 kHz của giọng nói |
| Độ dài khung | 25ms (400 mẫu) | Đủ dài cho đặc trưng âm học, đủ ngắn để theo dõi thay đổi |
| Bước nhảy khung | 10ms (160 mẫu) | Chồng lấp 15ms bảo toàn thông tin biên |
| Cửa sổ | Hamming | Giảm rò rỉ phổ tại biên khung |
| Kích thước FFT | 400 | Độ phân giải tần số = 16000/400 = 40 Hz/bin |
| Bộ lọc Mel | 80 dải | Mô hình hóa 80 kênh tần số cảm giác |
| Dải Mel | 0–8000 Hz | Bao phủ toàn bộ dải giọng nói người |
| Model Whisper | Tự động theo VRAM | large-v3 (≥6GB) / medium (4-6GB) / small (CPU) |
| Lớp Whisper | 6 Encoder + 6 Decoder | Đủ độ sâu cho tiếng Việt có thanh điệu |
| Đầu Attention | 8 mỗi lớp | Học 8 loại mối quan hệ độc lập mỗi token |
| Chiều embedding | 512 | Chiều không gian biểu diễn |
| Thuật toán WER | Levenshtein DP | Độ phức tạp O(m×n) — khoảng cách chỉnh sửa tối thiểu chính xác |
| Thư viện WER | jiwer 3.x | Bao gồm chuẩn hóa văn bản tự động |
| Phát hiện onset | `librosa.onset.onset_strength` | Đo Spectral Flux (tốc độ thay đổi năng lượng) |
| Chỉ số nhịp điệu | Độ lệch chuẩn của onset | Đo độ phân tán đỉnh năng lượng — đơn điệu = std thấp |
| Chuẩn hóa nhịp | min(100, σ × 20) | Ánh xạ dải σ điển hình (0–5) sang thang 0–100 |
| Công thức WPM | (từ / thời_gian_nói) × 60 | Tính WPM dựa trên thời gian nói thực (trừ khoảng lặng) |
| Ngưỡng im lặng | 5% trung bình RMS | Bảo thủ — tránh phân loại nhầm giọng nhỏ là im lặng |
| Ngắt tối thiểu | 150ms | Dưới mức này không phải khoảng thở có nghĩa |
| Ngắt lý tưởng | 0.4–0.9s | Tiêu chuẩn dẫn chương trình MC tiếng Việt |
| Trích xuất F0 | `librosa.pyin` (probabilistic YIN) | Mạnh hơn YIN thuần — xuất cờ voiced/unvoiced |
| Chỉ số cao độ | Độ lệch chuẩn F0 theo semitone | Thang log — độc lập người nói, có ý nghĩa âm vị học |
| Chỉ số âm lượng | Độ lệch chuẩn RMS mỗi khung 25ms | Nắm bắt động lực âm lượng qua toàn phát ngôn |
| Chỉ số nhịp độ | Độ lệch chuẩn khoảng beat | Nắm bắt biến thiên nhịp — chậm/nhanh qua các câu |
| Trọng số cảm xúc | cao_độ 50% + âm_lượng 30% + nhịp_độ 20% | Cao độ = tương quan biểu cảm mạnh nhất theo tài liệu huấn luyện giọng |
| Framework API | FastAPI + uvicorn | ASGI async — không chặn cho các yêu cầu đồng thời |
| Suy luận GPU | CUDA amp autocast | FP16 độ chính xác hỗn hợp — giảm nửa VRAM cùng chất lượng |
| TTS Engine | Supertonic (ONNX) | 99M params, RTF 0.012–0.015 CPU, 44.1kHz output |
| TTS Voices | M1–M5, F1–F5 | 10 giọng built-in, chọn qua form param `voice` |
| TTS Sample Rate | 44.100 Hz | 2.75× chất lượng so với MMS-TTS-VIE (16kHz) |

---

## 15. Module TTS — Tổng Hợp Giọng Nói MC (Supertonic)

### Tổng quan

Module TTS (Text-to-Speech) cho phép hệ thống tổng hợp giọng nói MC chuyên nghiệp từ văn bản kịch bản. Từ phiên bản này, engine được nâng cấp từ **Facebook MMS-TTS-VIE** sang **Supertonic** — engine on-device ONNX tốc độ cao, output 44.1kHz.

### So Sánh Engine Cũ và Mới

| Tiêu chí | MMS-TTS-VIE (cũ) | Supertonic (mới) |
|:---|:---|:---|
| Kiến trúc | VITS (PyTorch) | ONNX Runtime |
| Sample rate output | 16.000 Hz | **44.100 Hz** |
| Tốc độ CPU (RTF) | ~0.1–0.3 (ước tính) | **0.012–0.015** |
| Số giọng | 1 (giọng Việt duy nhất) | **10** (M1–M5, F1–F5) |
| Expression tags | Không | **Có** (`<laugh>`, `<breath>`, `<sigh>`) |
| Ngôn ngữ hỗ trợ | Tiếng Việt only | **31 ngôn ngữ** |
| Kích thước model | ~300MB | ~400MB |
| Dependencies | torch + transformers (~2GB) | onnxruntime (~50MB) |
| Download tự động | Không (thủ công) | **Có** (`auto_download=True`) |
| License | CC-BY-NC 4.0 | OpenRAIL-M |

### Tác Động Đến Trải Nghiệm Người Dùng

**1. Chất lượng âm thanh cao hơn đáng kể**

Output 44.1kHz là chuẩn CD audio — giọng nói rõ ràng, tự nhiên hơn so với 16kHz. Người dùng nghe demo giọng MC qua `/generate-mc-voice` sẽ thấy sự khác biệt rõ ràng, đặc biệt ở âm sắc và độ nét của phụ âm.

**2. Phản hồi nhanh hơn ~10–20×**

RTF (Real-Time Factor) = 0.012 nghĩa là tổng hợp 10 giây âm thanh mất ~0.12 giây CPU. Người dùng nhấn "Generate Voice" nhận kết quả gần tức thì thay vì chờ vài giây như trước.

**3. Chọn giọng linh hoạt**

10 built-in voices: M1–M5 (nam), F1–F5 (nữ). MC có thể chọn giọng phù hợp với phong cách dẫn (trang trọng, trẻ trung, v.v.). Default: `F1` (có thể override qua env `TTS_DEFAULT_VOICE`).

**4. Expression tags cho giọng MC biểu cảm**

Hỗ trợ tag cảm xúc trong text:
```
"Kính chào quý vị <breath> và các bạn thân mến!"
"Đây là <laugh> một khoảnh khắc thật đặc biệt."
```
Giúp giọng tổng hợp tự nhiên, phù hợp ngữ cảnh MC hơn.

**5. Khởi động server nhanh hơn**

Không còn load VITS model (~300MB) vào RAM khi service start. Supertonic ONNX lazy-load qua HuggingFace cache → startup time giảm.

**6. Giảm dung lượng dependencies**

Xóa `transformers` (~800MB) và `accelerate` (~200MB) khỏi requirements. Tổng install size giảm ~1GB — quan trọng khi deploy trên Render free tier (disk limit 512MB–1GB).

### API Endpoint

```
POST /generate-mc-voice
Content-Type: application/x-www-form-urlencoded

Tham số:
  text   (required) — văn bản cần tổng hợp
  voice  (optional) — M1/M2/M3/M4/M5/F1/F2/F3/F4/F5 (default: F1)

Response:
  {
    "status":      "success",
    "message":     "MC voice generated successfully",
    "file_path":   "mc_voice_output.wav",
    "voice":       "F1",
    "sample_rate": 44100
  }
```

### Ví Dụ Sử Dụng

```python
# Python
import requests
resp = requests.post(
    "http://localhost:8001/generate-mc-voice",
    data={"text": "Kính chào quý vị và các bạn thân mến!", "voice": "F1"}
)
# → mc_voice_output.wav (44.1kHz, ~0.1s latency trên CPU)
```

```powershell
# PowerShell
Invoke-WebRequest http://localhost:8001/generate-mc-voice `
  -Method POST `
  -Body "text=Kính chào quý vị&voice=F1" `
  -ContentType "application/x-www-form-urlencoded"
```

### Kiến Trúc Kỹ Thuật

```
Text input
    ↓
Supertonic Tokenizer (multilingual BPE)
    ↓
ONNX Runtime Inference (2 inference steps)
    ↓
Waveform 44.1kHz float32
    ↓
soundfile.write() → mc_voice_output.wav
    ↓
FastAPI response { file_path, voice, sample_rate }
```

Supertonic dùng **2 inference steps** (không phải diffusion nhiều bước như một số model khác) → tốc độ RTF cực thấp ngay cả trên CPU.

### Cấu Hình

| Env variable | Default | Ý nghĩa |
|:---|:---|:---|
| `TTS_DEFAULT_VOICE` | `F1` | Giọng mặc định khi không truyền `voice` param |

Model download tự động vào `~/.cache/huggingface/` khi khởi động lần đầu (~400MB). Không cần set `TTS_MODEL_PATH`.

### Giới Hạn

- **Không hỗ trợ voice cloning** trong open-weight model. Giọng custom MC cụ thể cần Supertonic Voice Builder (paid service).
- **License OpenRAIL-M** — cần review điều khoản trước khi dùng commercial scale.
- **Vietnamese accent** — chất lượng tốt nhưng chưa có MOS benchmark cụ thể cho tiếng Việt. Khuyến nghị test thực tế với các câu MC điển hình trước khi deploy production.

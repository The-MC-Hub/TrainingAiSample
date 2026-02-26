# 🎓 MASTERCLASS: CƠ CHẾ PHÂN TÍCH GIỌNG NÓI AI - GIẢI THÍCH ĐẦY ĐỦ NHẤT
### Dành cho dự án The MC Hub — Hướng dẫn kỹ thuật & khoa học toàn diện

> Tài liệu này giải thích toàn bộ vòng đời của dữ liệu âm thanh — từ khi micro thu giọng nói cho đến lúc AI đưa ra điểm số và lời khuyên. Không nói ngắn gọn, chỉ nói đầy đủ và tường minh nhất có thể.

---

## 📌 MỤC LỤC

1. [Tổng quan luồng xử lý](#tong-quan)
2. [Giai đoạn 1: Vật lý âm thanh & Số hóa](#giai-doan-1)
3. [Giai đoạn 2: Biến đổi Fourier (FFT)](#giai-doan-2)
4. [Giai đoạn 3: Thang đo Mel & Spectrogram](#giai-doan-3)
5. [Giai đoạn 4: Nhận diện giọng nói bằng Whisper AI](#giai-doan-4)
6. [Giai đoạn 5: Cơ chế Attention (Sự chú ý)](#giai-doan-5)
7. [Giai đoạn 6: Chấm điểm Accuracy (WER + Levenshtein)](#giai-doan-6)
8. [Giai đoạn 7: Phân tích Nhấn nhá & Nhịp điệu](#giai-doan-7)
9. [Giai đoạn 8: Hệ thống chuyên gia ảo (Expert Tips)](#giai-doan-8)
10. [Ví dụ đầu-cuối toàn phần](#vi-du-end-to-end)
11. [Bảng thông số kỹ thuật](#thong-so)

---

## 📍 TỔNG QUAN LUỒNG XỬ LÝ {#tong-quan}

Khi người dùng nhấn nút "Phân tích", giọng nói đi qua các lớp xử lý nối tiếp nhau như một dây chuyền sản xuất:

```
🎙️ MICRO → 🌊 SÓNG ÂM → ✂️ FRAMING → 📊 FFT → 🎛️ MEL FILTER → 🖼️ SPECTROGRAM
        → 🧠 WHISPER AI (Encoder → Decoder) → 📝 VĂN BẢN
        → 📏 WER/LEVENSHTEIN → 📈 ĐIỂM CHÍNH XÁC
        → 🎵 LIBROSA ONSET → 📈 ĐIỂM NHẤN NHÁ
        → ⏱️ WPM CALCULATION → 📈 ĐIỂM TỐC ĐỘ
        → 💬 EXPERT FEEDBACK → 👤 NGƯỜI DÙNG
```

Mỗi bước trên đây sẽ được giải thích kỹ lưỡng bên dưới.

---

## 🌊 GIAI ĐOẠN 1: VẬT LÝ ÂM THANH & SỐ HÓA {#giai-doan-1}

### 1.1 Âm thanh là gì?

Khi bạn nói, các dây thanh đới trong cổ họng rung lên với tần số nhất định. Sự rung động này lan truyền qua không khí dưới dạng **sóng áp suất** — các vùng nén và giãn của phân tử không khí xen kẽ nhau, truyền đến micro.

Micro biến rung động cơ học đó thành **tín hiệu điện**, và card âm thanh của máy tính biến tín hiệu điện đó thành **chuỗi số nhị phân**.

### 1.2 Lấy mẫu (Sampling) và định lý Nyquist

Máy tính không thể lưu một sóng âm liên tục (analog). Nó phải "chụp ảnh" sóng âm đó rất nhiều lần trong một giây — gọi là **tần số lấy mẫu (Sampling Rate)**.

**Hệ thống MC Hub sử dụng: 16,000 Hz (16kHz)**

Nghĩa là: Cứ mỗi giây, máy tính ghi lại **16,000 điểm dữ liệu** về biên độ (độ mạnh/yếu) của âm thanh.

```
                Quá trình lấy mẫu (Sampling)
Biên độ
  +1.0 │      ╭──╮          ╭──╮
       │    ╭╯    ╰╮      ╭╯    ╰╮
  0.0  │───╯────────╰────╯────────╰─────  <- Sóng âm liên tục (Analog)
       │                                  
       │    ×    ×    ×    ×    ×    ×    <- Điểm lấy mẫu (16,000 điểm/giây)
  -1.0 │
       └──────────────────────────────────> Thời gian

Mỗi × là một số thực từ -1.0 đến +1.0, ví dụ:
[0.0, 0.12, 0.45, 0.78, 0.95, 0.82, 0.54, 0.21, -0.05, -0.3, ...]
```

**Tại sao 16kHz?**
Theo **Định lý lấy mẫu Nyquist-Shannon** (một trong những định lý nền tảng của Kỹ thuật số): Để tái tạo lại chính xác một tần số, bạn phải lấy mẫu nhanh ít nhất **gấp đôi** tần số đó.

- Giọng người bình thường nằm trong dải 85Hz đến 8,000Hz
- Để tái tạo 8,000Hz cần tốc độ lấy mẫu ≥ 16,000Hz
- → **16kHz là con số "vừa đủ"** — đủ để hiểu giọng người, không quá nặng để xử lý

### 1.3 Lượng tử hóa (Quantization)

Mỗi điểm lấy mẫu được lưu với độ chính xác **16-bit** (65,536 mức biên độ khác nhau). Toàn bộ 1 giây âm thanh của bạn = 16,000 × 16 bit = 256,000 bit = 32 kilobytes.

---

## 📊 GIAI ĐOẠN 2: BIẾN ĐỔI FOURIER NHANH (FFT) {#giai-doan-2}

### 2.1 Vấn đề: Sóng âm thô quá "hỗn loạn"

Hãy nhìn vào dữ liệu thô khi bạn nói câu "Chào mừng":

```
Biên độ (raw waveform)
  │ /\/\/\/\/\/\___/\/\/\/\___/\/\/\___
  │ Nhìn vào đây, bạn không biết âm nào đang được phát
```

Từ dạng sóng thô này, máy tính **không thể** phân biệt được đâu là nguyên âm "A", đâu là phụ âm "M", đâu là tiếng ồn nền. Cần một công cụ mạnh hơn.

### 2.2 Định lý Fourier — Nền tảng toán học

**Jean-Baptiste Joseph Fourier** (1768-1830) đã chứng minh một điều phi thường:

> **Bất kỳ tín hiệu tuần hoàn nào cũng có thể được biểu diễn là tổng của vô số sóng hình sin đơn giản**

Điều này có nghĩa là: Âm thanh phức tạp mà bạn nói ra — dù nghe có vẻ là một âm thanh thống nhất — thực ra là **rất nhiều tần số cộng lại với nhau**.

**Ví dụ với nhạc cụ:** Khi đánh nốt "Đô" trên đàn piano, bạn nghe thấy một nốt duy nhất. Thực ra, bên trong đó có: tần số cơ bản 261Hz, cộng với các bồi âm 522Hz, 783Hz, 1044Hz... chồng lên nhau.

### 2.3 FFT hoạt động như thế nào?

FFT (Fast Fourier Transform) là thuật toán cực kỳ mạnh mẽ để "bóc tách" sóng âm ra thành các thành phần tần số.

#### Bước 1: Windowing (Cửa sổ hóa)
AI không thể phân tích toàn bộ file âm thanh cùng một lúc. Nó cắt file thành các **khung (frames)** nhỏ:

```
File âm thanh 5 giây:
│────────────────────────────────────────────────│
│  Frame 1  │  Frame 2  │  Frame 3  │  Frame 4  │  ...
    25ms       25ms         25ms        25ms
  (bước 10ms)(bước 10ms)(bước 10ms)(bước 10ms)
```

- Mỗi frame dài **25ms** (400 mẫu tại 16kHz)
- Bước nhảy giữa các frame là **10ms** (160 mẫu)
- → Các frame **chồng lên nhau 15ms** để không mất thông tin ở biên

#### Bước 2: Áp dụng Cửa sổ Hamming (Hamming Window)
Trước khi tính FFT, mỗi frame được nhân với một hàm Hamming:

```
Cửa sổ Hamming (dạng chuông):
  1.0 │         ╭─────────────╮
      │      ╭─╯               ╰─╮
  0.5 │    ╭─╯                   ╰─╮
      │  ╭─╯                       ╰─╮
  0.0 │──╯                           ╰──
      └──────────────────────────────────> Thời gian trong frame
```

**Tại sao cần Hamming Window?** Nếu cắt âm thanh đột ngột ở 2 đầu frame, sẽ tạo ra các tần số "ảo" không tồn tại trong thực tế. Hamming Window làm cho biên độ ở 2 đầu frame giảm về 0 từ từ, tránh hiện tượng "rò rỉ tần số" (spectral leakage).

#### Bước 3: Tính FFT
Với mỗi frame, thuật toán FFT tính toán **biên độ của từng tần số** hiện diện trong frame đó.

```
Kết quả FFT của frame khi nói âm "A":

Biên độ
  │
  │  ████
  │  ████  █
  │  ████  ███
  │  ████  ██████
  │  ████  ████████  ██
  │  ████  ████████  ████  █  
  └──────────────────────────────────> Tần số (Hz)
     200  500  750  1200  1700  2800
     ↑    ↑            ↑
  Giọng Trầm  F1 (âm mô)  F2 (sắc nét)
```

**Ý nghĩa thực tế cho MC:**
| Dải tần số | Thành phần âm thanh | Liên quan đến MC |
|---|---|---|
| 80 - 200 Hz | Tần số cơ bản (giọng trầm/cao) | MC nam thường 100-140Hz, MC nữ 180-250Hz |
| 200 - 800 Hz | Âm mô (formant F1) | Quyết định nguyên âm nghe rõ/mờ |
| 800 - 3000 Hz | Formant F2, sự sắc nét | Phân biệt các nguyên âm khác nhau |
| 3000 - 8000 Hz | Phụ âm gió (S, X, H) | Phát âm rõ ràng hay ề à |

---

## 🎛️ GIAI ĐOẠN 3: THANG ĐO MEL & MEL SPECTROGRAM {#giai-doan-3}

### 3.1 Vấn đề với tần số vật lý (Hz)

FFT cho ta biên độ của tất cả tần số từ 0 đến 8,000Hz. Nhưng nếu AI học trực tiếp từ đó, nó sẽ gặp một vấn đề lớn:

**Tai người nghe KHÔNG theo tuyến tính.**

Hãy thử cảm nhận:
- Bạn nói với giọng thấp 100Hz, sau đó tăng lên 200Hz → Bạn cảm thấy sự thay đổi **RẤT LỚN** (gấp đôi về cảm giác)
- Bạn bắt đầu ở 7,000Hz và tăng lên 7,100Hz → Bạn **hầu như không nhận ra** sự thay đổi

Điều này có nghĩa: Tần số thấp "chứa đựng" nhiều thông tin cảm xúc và ngôn ngữ hơn tần số cao. Nếu AI xử lý đều các tần số, nó đang "lãng phí" tài nguyên vào những tần số không quan trọng.

### 3.2 Thang đo Mel — Mô phỏng tai người

**Công thức chuyển đổi từ Hz sang Mel:**
```
Mel(f) = 2595 × log₁₀(1 + f / 700)
```

Kết quả của phép biến đổi này:

```
Tần số vật lý (Hz) vs Thang Mel

Biểu đồ so sánh:
Hz:   100  200  400  800  1600  3200  6400
       │    │    │    │     │     │     │
Hz:    ├────┼────┼────┼─────┼─────┼─────┤  ← Khoảng cách đều nhau (tuyến tính)
       │    │    │    │                 │
Mel:   ├─────────┼────────┼───────────┤  ← Khoảng thưa dần (logarithm)
      (nhiều chi tiết ở thấp)  (ít chi tiết ở cao)
```

### 3.3 Bộ lọc Mel (Mel Filterbanks)

Hệ thống áp dụng **80 bộ lọc hình tam giác** (trong hệ thống chúng ta), mỗi bộ lọc tập trung vào một dải tần số Mel:

```
80 bộ lọc Mel:
Biên độ
  1.0 │ /\ /\ /\ /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
      │/  X  X  X/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
  0.0 └───────────────────────────────────────────────────> Tần số
     0Hz                                              8000Hz
     ←── Bộ lọc thưa, rộng ──→ ←── Bộ lọc dày, hẹp ──→
     (vùng thấp, quan trọng)        (vùng cao, ít quan trọng)
```

### 3.4 Mel Spectrogram — Biến âm thanh thành "Bức ảnh"

Sau khi áp dụng 80 bộ lọc Mel lên từng frame FFT, chúng ta có một **ma trận 80 × T** (T là số frame). Ma trận này tạo thành **Mel Spectrogram** — một bức ảnh 2 chiều:

```
MEL SPECTROGRAM — Bức ảnh AI "nhìn thấy"

Tần số Mel (80 kênh)
  ↑  (8000Hz)
80 │ ░░░░░▓░░░░░░░░░░▓▓░░░░░░░░░░░░
   │ ░░░░▓▓▓░░░░░░░▓▓▓▓░░░░░░░░░░░░│ Phụ âm có năng lượng cao tần
   │ ░░▓▓▓▓▓▓░░░░▓▓▓▓▓▓▓░░░░░░░░░░░│
   │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ Nguyên âm có năng lượng dải giữa
   │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
  1 │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ Giọng trầm luôn có năng lượng thấp tần
  ↓  (80Hz)
     └──────────────────────────────────> Thời gian (T frames)
         Ch  à   o      m  ư  n  g
```

Mỗi cột dọc là một "snapshot" âm thanh 25ms, mỗi hàng ngang là một dải tần số. Màu tối (░) = ít năng lượng, màu đậm (▓) = nhiều năng lượng.

**Đây chính là "thứ" mà Whisper AI đưa vào Encoder của nó.** Bước này biến bài toán nhận diện giọng nói thành bài toán nhận dạng hình ảnh!

---

## 🧠 GIAI ĐOẠN 4: NHẬN DIỆN GIỌNG NÓI — OPENAI WHISPER AI {#giai-doan-4}

### 4.1 Whisper là gì?

Whisper là mô hình học sâu (Deep Learning) do OpenAI phát triển và công bố năm 2022. Nó được huấn luyện trên **680,000 giờ âm thanh** từ internet, bao gồm nhiều ngôn ngữ trong đó có tiếng Việt.

Model "Base" mà hệ thống chúng ta sử dụng có **74 triệu tham số** — tức là 74 triệu con số mà AI học và điều chỉnh trong quá trình huấn luyện để trở nên chính xác.

### 4.2 Kiến trúc Transformer Encoder-Decoder

Whisper sử dụng kiến trúc **Transformer** — cùng kiến trúc với GPT và BERT. Nó có 2 phần chính:

```
                ┌─────────────────────────────────────────┐
MEL             │            ENCODER (Bộ nghe)            │
SPECTROGRAM ──► │  Layer 1 → Layer 2 → ... → Layer 6      │
(80 × T)        │  (Mỗi layer có Self-Attention + FFN)    │
                │                                          │
                │  Đầu ra: Vector biểu diễn âm thanh       │
                │  kích thước (T × 512)                    │
                └──────────────────┬──────────────────────┘
                                   │ Context vectors
                ┌──────────────────▼──────────────────────┐
                │            DECODER (Bộ tạo chữ)         │
                │  Input: Token đã tạo + Context vectors   │
                │  Layer 1 → Layer 2 → ... → Layer 6      │
                │  (Mỗi layer có Masked Self-Attention     │
                │   + Cross-Attention với Encoder)         │
                │                                          │
                │  Đầu ra: Xác suất cho từng chữ/token    │
                └──────────────────────────────────────────┘

Kết quả: "C", "h", "à", "o", " ", "m", "ừ", "n", "g", ...
→ Ghép lại: "Chào mừng quý vị"
```

### 4.3 Cơ chế hoạt động của Encoder

Encoder nhận Mel Spectrogram và biến nó thành các **vector ngữ nghĩa âm thanh**. Qua 6 lớp Transformer liên tiếp, nó học được:

- **Lớp 1-2:** Các đặc trưng âm học cơ bản (sự chuyển đổi tần số đột ngột, điểm bắt đầu của âm tiết)
- **Lớp 3-4:** Các mẫu âm tiết cụ thể của tiếng Việt (các thanh điệu, nguyên âm đặc trưng)
- **Lớp 5-6:** Ngữ cảnh toàn câu và kết nối các âm tiết với nhau

---

## 💡 GIAI ĐOẠN 5: CƠ CHẾ ATTENTION — "BỘ NÃO BIẾT CHÚ Ý" {#giai-doan-5}

### 5.1 Tại sao cần Attention?

Trong ngôn ngữ, thứ tự và mối quan hệ giữa các từ cực kỳ quan trọng. Ví dụ:
- "Con chó cắn con mèo" ≠ "Con mèo cắn con chó" (cùng từ, khác nghĩa!)
- "Tôi **mừng** vì anh đến" — chữ "mừng" bị ảnh hưởng bởi chữ "vì" và "đến"

Cơ chế Attention cho phép AI xem xét **mối quan hệ giữa mọi vị trí** trong chuỗi với nhau, không bị giới hạn bởi khoảng cách.

### 5.2 Self-Attention — Công thức toán học

Với mỗi vị trí i trong chuỗi, Attention tính toán:
1. Vector **Query (Q)**: "Tôi đang tìm kiếm thông tin gì?"
2. Vector **Key (K)**: "Tôi có thông tin gì?"
3. Vector **Value (V)**: "Thông tin thực sự của tôi là gì?"

```
Attention Score(i, j) = softmax(Qᵢ · Kⱼ / √d_k) × Vⱼ
```

Trong đó d_k = 64 (kích thước key vector)

### 5.3 Ví dụ trực quan với câu MC

Câu: **"Chào mừng quý vị đã quay trở lại với chương trình"**

Khi Decoder đang tạo ra chữ **"lại"**, nó tính Attention Score với tất cả các token trước:

```
Token đang tạo: "lại"

Attention Score (từ 0.0 đến 1.0):
  "Chào"    ║░░░░░░░░░░░░░░░║ 0.02
  "mừng"    ║░░░░░░░░░░░░░░░║ 0.01
  "quý"     ║░░░░░░░░░░░░░░░║ 0.02
  "vị"      ║░░░░░░░░░░░░░░░║ 0.03
  "đã"      ║░░░░░░░░░░░░░░░║ 0.04
  "quay"    ║████████████████║ 0.35  ← Chú ý rất cao! (cụm "quay trở lại")
  "trở"     ║███████████░░░░░║ 0.29  ← Chú ý cao! (từ đang ghép với "quay")
  "lại"     ║░░░░░░░░░░░░░░░░║ 0.24 ← Chính nó (vì nó thường xuất hiện sau "trở")
                              TỔNG = 1.0
```

**Ý nghĩa thực tế:** Nếu trong âm thanh, chữ "trở" bị nghe mờ do tiếng ồn, AI vẫn đoán đúng 95% vì:
- Nó đã nghe thấy "quay" (Attention Score 0.35)
- Trong tiếng Việt, "quay" + ??? + "lại" thường ghép với "trở"
- AI điền vào chỗ trống một cách tự động!

### 5.4 Cross-Attention — Kết nối âm thanh với ngôn ngữ

Cross-Attention cho phép Decoder nhìn vào **output của Encoder** (biểu diễn âm thanh) khi tạo ra mỗi chữ:

```
Khi tạo chữ "quý":

Decoder hỏi: "Trong Mel Spectrogram, frame nào chứa âm 'quý'?"
Encoder trả lời: "Frame 45-52 có hoa văn tần số đặc trưng của âm 'qu' + 'y'"
Decoder kết luận: "Rất đúng! Đây là chữ 'quý'"
```

---

## 📏 GIAI ĐOẠN 6: CHẤM ĐIỂM ĐỘ CHÍNH XÁC (WER + LEVENSHTEIN) {#giai-doan-6}

### 6.1 WER là gì?

**WER (Word Error Rate)** — Tỷ lệ lỗi từ — là thước đo tiêu chuẩn quốc tế (IEEE, NIST) để đánh giá hệ thống nhận diện giọng nói.

### 6.2 Thuật toán Levenshtein Distance

Đây là thuật toán đo **khoảng cách chỉnh sửa** giữa hai chuỗi văn bản. Được đặt theo tên nhà toán học Nga **Vladimir Levenshtein** (1935–2017).

**Ba loại thao tác chỉnh sửa:**
1. **Substitution (S):** Thay một từ bằng từ khác (`"bạn"` → `"ban"`)
2. **Deletion (D):** Xóa bỏ một từ thừa
3. **Insertion (I):** Thêm một từ còn thiếu

### 6.3 Minh họa Ma trận Levenshtein

**Kịch bản gốc (Ref):** `"chào mừng quý vị"`  
**Người dùng nói (Hyp):** `"chào mừng quý bạn"`

```
Bảng ma trận Dynamic Programming:

         ""   chào  mừng  quý   bạn
    ""  [ 0 ] [ 1 ] [ 2 ] [ 3 ] [ 4 ]
   chào [ 1 ] [ 0 ] [ 1 ] [ 2 ] [ 3 ]
   mừng [ 2 ] [ 1 ] [ 0 ] [ 1 ] [ 2 ]
    quý [ 3 ] [ 2 ] [ 1 ] [ 0 ] [ 1 ]
     vị [ 4 ] [ 3 ] [ 2 ] [ 1 ] [ 1 ] ← Kết quả: 1 lỗi (Substitution "bạn"→"vị")
```

**Giải thích từng ô:**
- Ô (i, j) = số thao tác ít nhất để biến i từ đầu của Hyp thành j từ đầu của Ref
- Ô cuối cùng = tổng số lỗi

### 6.4 Công thức tính điểm

```python
# Code thực tế trong main.py
error_rate    = jiwer.wer(script_origin.lower(), text_spoken.lower())
accuracy_score = max(0, 100 - (error_rate * 100))

# Ví dụ: error_rate = 0.25 (1 lỗi / 4 từ)
# accuracy_score = max(0, 100 - 25) = 75.0
```

### 6.5 Các trường hợp đặc biệt

| Trường hợp | Kịch bản | Người nói | Tổng lỗi | Accuracy |
|---|---|---|---|---|
| Hoàn hảo | "xin chào" | "xin chào" | 0 | 100% |
| 1 từ sai | "xin chào" | "xin hào" | 1 Sub | 50% |
| Bỏ từ | "xin chào bạn" | "xin chào" | 1 Del | 67% |
| Thêm từ | "xin chào" | "xin chào bạn nhé" | 2 Ins | 0% (capped) |
| Nói hoàn toàn sai | "xin chào" | "hello world" | 2 Sub | 0% |

---

## 🎵 GIAI ĐOẠN 7: PHÂN TÍCH NHẤN NHÁ & NHỊP ĐIỆU {#giai-doan-7}

Đây là phần **đặc sắc nhất và phức tạp nhất** của hệ thống, vì "nhấn nhá" là một khái niệm nghệ thuật nhưng AI phải định nghĩa nó bằng toán học.

### 7.1 Onset Detection — Tìm điểm bắt đầu của âm thanh

**Onset** là thời điểm một âm thanh mới bắt đầu — như cái "bùm" khi đánh trống, hay khoảnh khắc bạn bắt đầu nói một âm tiết mới.

Librosa sử dụng hàm `onset_strength` để đo **mức độ thay đổi đột ngột của năng lượng**:

```python
onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
# Kết quả: mảng [0.1, 0.1, 0.2, 2.5, 1.8, 0.3, 0.1, 0.1, 3.1, 2.0, ...]
#                                    ↑                        ↑
#                              Bắt đầu âm tiết "mừng"   Bắt đầu "quý"
```

### 7.2 Standard Deviation — Đo lường sự "truyền cảm"

**Standard Deviation (Độ lệch chuẩn - σ)** đo mức độ phân tán của một tập số liệu xung quanh giá trị trung bình.

```
Công thức: σ = √( Σ(xᵢ - x̄)² / N )
```

**Ví dụ so sánh trực quan:**

```
MC NGHIỆP DƯ đọc "Chào mừng quý vị hôm nay":

Onset Strength:
  │
3 │
2 │ ▓  ▓  ▓  ▓  ▓
1 │ ▓  ▓  ▓  ▓  ▓
  └──────────────────> Thời gian
  Trung bình (x̄) ≈ 1.8
  Độ lệch chuẩn (σ) ≈ 0.2  → Điểm nhấn nhá = min(100, 0.2 × 20) = 4/100

MC CHUYÊN NGHIỆP đọc "Chào mừng QUÝYYY VỊ hôm NAY":

Onset Strength:
  │
4 │          ████
3 │ ▓        ████     ▓▓▓
2 │ ▓  ▓     ████  ▓  ▓▓▓
1 │ ▓  ▓  ▓  ████  ▓  ▓▓▓  ▓
  └────────────────────────────> Thời gian
  Trung bình (x̄) ≈ 1.9
  Độ lệch chuẩn (σ) ≈ 1.1  → Điểm nhấn nhá = min(100, 1.1 × 20) = 22/100
```

### 7.3 Tính WPM (Words Per Minute)

```python
# Code thực tế
duration   = librosa.get_duration(y=audio_data, sr=sr)  # Thời gian tính bằng giây
word_count = len(text_spoken.split())                    # Đếm số từ AI nhận ra
wpm        = (word_count / duration) * 60               # Quy đổi ra từ/phút
```

---

## 🎓 GIAI ĐOẠN 8: HỆ THỐNG CHUYÊN GIA ẢO (EXPERT FEEDBACK) {#giai-doan-8}

Sau khi có đủ 3 con số (Accuracy, Rhythm, WPM), hệ thống áp dụng các **quy tắc chuyên môn (Heuristic Rules)** đúc kết từ ngành MC để tạo ra phản hồi:

### 8.1 Bảng quy tắc Phát âm (Accuracy Rules)
| Accuracy Score | Mức đánh giá | Phản hồi AI |
|---|---|---|
| ≥ 90% | Xuất sắc | "Phát âm: Chuẩn xác tuyệt vời." |
| 75 - 89% | Tốt | "Phát âm: Khá rõ, cần cải thiện 1-2 âm cuối." |
| 60 - 74% | Trung bình | "Phát âm: Cần rõ chữ hơn, đặc biệt là các âm cuối." |
| < 60% | Cần luyện tập | "Phát âm: Cần luyện tập thêm nhiều." |

### 8.2 Bảng quy tắc Nhấn nhá (Rhythm Rules)
| Rhythm Score | Mức đánh giá | Phản hồi AI |
|---|---|---|
| ≥ 40 | Tốt | "Nhấn nhá: Tốt! Bạn có sự thay đổi âm lượng tạo cảm xúc." |
| < 40 | Cần cải thiện | "Nhấn nhá: Giọng còn hơi đều (monotone). Hãy thử nhấn mạnh vào các từ quan trọng." |

### 8.3 Bảng quy tắc Tốc độ (WPM Rules)
| WPM | Mức đánh giá | Lời khuyên |
|---|---|---|
| < 100 | Quá chậm | Tăng tốc, tránh gây buồn ngủ |
| 100 - 120 | Hơi chậm | Có thể tăng thêm chút năng lượng |
| 120 - 150 | Tốt nhất ✓ | Tiêu chuẩn vàng cho MC Việt Nam |
| 150 - 180 | Hơi nhanh | Cần lấy hơi sâu hơn giữa các câu |
| > 180 | Quá nhanh | Khán giả không kịp tiếp thu |

---

## 🧪 VÍ DỤ ĐẦU-CUỐI TOÀN PHẦN (END-TO-END EXAMPLE) {#vi-du-end-to-end}

Hãy theo dõi toàn bộ hành trình của một bài đọc:

**Kịch bản gốc:** *"Chào mừng quý vị và các bạn đã đến với chương trình hôm nay"*

**Người dùng đọc thực tế:** *"Chào mừng quý vị và các bạn đã đến với chương trình hôm nay"* (đọc đúng hoàn toàn, nhưng tốc độ đều và nhàm)

---

**Bước 1 — Thu âm:**  
Micro ghi lại 48,000 mẫu số (3 giây × 16,000Hz = 48,000 điểm dữ liệu)

**Bước 2 — Framing:**  
Cắt ra 299 frames (mỗi frame 400 mẫu, bước 160 mẫu)

**Bước 3 — FFT & Mel:**  
Tạo Mel Spectrogram kích thước 80 × 299 = ma trận 80 hàng, 299 cột

**Bước 4 — Whisper Encode:**  
Encoder chuyển Spectrogram thành vector 512 chiều × 299 frame

**Bước 5 — Whisper Decode:**  
Decoder tạo ra token-by-token: `[50258, 50263, 50358, 5160, 2261, ...]`  
→ Giải mã: `"Chào mừng quý vị và các bạn đã đến với chương trình hôm nay"`

**Bước 6 — Tính WER:**
```
Ref: "chào mừng quý vị và các bạn đã đến với chương trình hôm nay" (14 từ)
Hyp: "chào mừng quý vị và các bạn đã đến với chương trình hôm nay" (14 từ)
Lỗi: 0
WER = 0/14 = 0.0
Accuracy Score = 100.0%
```

**Bước 7 — Tính WPM:**
```
Thời gian: 3.0 giây
Số từ: 14
WPM = (14 / 3.0) × 60 = 280 WPM  ← QUÁ NHANH!
```

**Bước 8 — Tính Rhythm:**
```
onset_env = [0.8, 0.9, 0.8, 0.9, 0.8, 0.9, 0.8, 0.9, ...]  ← Rất đều
σ = 0.05
Rhythm Score = min(100, 0.05 × 20) = 1.0  ← Rất thấp! Monotone
```

**Bước 9 — Expert Feedback:**
```json
{
  "status": "success",
  "text_spoken": "Chào mừng quý vị và các bạn đã đến với chương trình hôm nay",
  "accuracy_score": 100.0,
  "rhythm_score": 1.0,
  "speaking_rate_wpm": 280.0,
  "feedback": "Phát âm: Chuẩn xác tuyệt vời. | Nhấn nhá: Giọng còn hơi đều (monotone).",
  "expert_tips": [
    {
      "label": "Năng lượng",
      "tip": "Cường độ âm thanh của bạn đạt 0.05. Hãy tăng độ biến thiên âm lượng."
    },
    {
      "label": "Ngắt nghỉ",
      "tip": "Hãy thử kéo dài lên 0.6s ở các đoạn chuyển ý."
    }
  ]
}
```

**Kết luận:** Người dùng đọc đúng 100% nhưng bị "Robot mode" — đọc với tốc độ 280 WPM (gần gấp đôi tốc độ vàng 150 WPM) và không có nhấn nhá. AI phát hiện được điều này!

---

## 📊 BẢNG THÔNG SỐ KỸ THUẬT ĐẦY ĐỦ {#thong-so}

| Tham số | Giá trị | Ý nghĩa khoa học |
|:---|:---|:---|
| **Sampling Rate** | 16,000 Hz | Theo Định lý Nyquist, đủ cho dải 0-8kHz của giọng người |
| **Frame Length** | 25ms (400 samples) | Đủ dài để có info âm học, đủ ngắn để bắt thay đổi nhanh |
| **Frame Hop** | 10ms (160 samples) | Chồng lấp 15ms để không mất thông tin tại biên |
| **Windowing** | Hamming | Giảm spectral leakage, bảo toàn thông tin biên frame |
| **FFT Size (n_fft)** | 400 | Độ phân giải tần số = 16000/400 = 40Hz/bin |
| **Mel Filters** | 80 kênh | Mô phỏng 80 dải tần số theo thang Mel |
| **Mel Range** | 0 - 8000 Hz | Bao phủ toàn bộ dải giọng người |
| **Whisper Model** | Base (74M params) | Cân bằng tốc độ và độ chính xác |
| **Whisper Layers** | 6 Encoder + 6 Decoder | Đủ sâu để hiểu tiếng Việt |
| **Attention Heads** | 8 heads/layer | Học 8 loại "quan hệ" khác nhau giữa các token |
| **Embedding Dim** | 512 | Không gian vector biểu diễn |
| **WER Algorithm** | Levenshtein DP | Độ phức tạp O(m×n), m,n = số từ 2 câu |
| **WER Library** | Jiwer 3.x | Bao gồm text normalization tự động |
| **Onset Detection** | `onset_strength` | Đo tốc độ thay đổi Spectral Flux |
| **Rhythm Metric** | Standard Deviation | Đo phân tán của các đỉnh năng lượng |
| **Normalization** | min(100, σ × 20) | Chuẩn hóa về thang điểm 0-100 |
| **WPM Formula** | (words/duration)×60 | Quy đổi từ giây sang phút |
| **API Framework** | FastAPI + Uvicorn | ASGI async, không blocking |
| **CORS** | Whitelist ports 3000/5173 | Cho phép React frontend gọi API |

---

## 📚 CÁC KHÁI NIỆM THAM KHẢO THÊM

Nếu bạn muốn đi sâu hơn vào từng lĩnh vực:

| Lĩnh vực | Tài liệu gợi ý |
|---|---|
| Biến đổi Fourier | "The Fourier Transform and Its Applications" - Bracewell (2000) |
| Mel Filterbank | HTK Speech Toolkit Documentation - Cambridge |
| Transformer & Attention | "Attention Is All You Need" - Vaswani et al. (2017) - Google Brain |
| Whisper | "Robust Speech Recognition via Large-Scale Weak Supervision" - Radford et al. (2022) - OpenAI |
| Word Error Rate | NIST Speech Recognition Scoring Toolkit (SCTK) Documentation |
| Onset Detection | Bello et al. "A Tutorial on Onset Detection in Music" - IEEE Signal Processing (2005) |

---

*📝 Tài liệu này được soạn thảo đầy đủ và chi tiết nhất có thể để giúp đội ngũ phát triển và người học hiểu trọn vẹn cơ chế hoạt động của hệ thống AI Coaching tại The MC Hub.*  
*Cập nhật lần cuối: 2026-02-26*

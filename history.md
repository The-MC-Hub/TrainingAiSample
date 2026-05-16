# Nhật ký Fine-tuning GPT-SoVITS (Vietnamese MC Voice)

## 📅 Trạng thái hiện tại (15/05/2026)
Chúng ta đã hoàn thành giai đoạn huấn luyện âm sắc (SoVITS) và đang chuẩn bị bước vào giai đoạn huấn luyện nhịp điệu (GPT).

### ✅ Các việc đã làm được:
1.  **Tiền xử lý (Preprocessing)**:
    *   Hoàn thành trích xuất BERT, Hubert và Semantic features cho 1000 mẫu tiếng Việt.
    *   Sửa lỗi đường dẫn `PRETRAINED_DIR` giúp tạo file `3-bert` và `6-name2semantic.tsv` chuẩn xác.
2.  **Huấn luyện SoVITS (Acoustic Model)**:
    *   **Fix lỗi treo**: Patch `s2_train.py` ép `num_workers=0` để tránh lỗi tràn bộ nhớ chia sẻ trên Windows.
    *   **Fix lỗi lưu file**: Patch `utils.py` thêm cơ chế **Retry** khi lưu checkpoint, khắc phục lỗi `PermissionError (WinError 32)` do Windows khóa file.
    *   **Kết quả**: Hoàn thành **8/8 Epoch** SoVITS. Đã có bộ trọng số (Weights) âm sắc hoàn chỉnh.
3.  **Ổn định hóa hệ thống**: Tối ưu hóa cấu hình cho RTX 4060 (8GB VRAM) với batch_size=4.

---

## 🚀 Kế hoạch tiếp theo
### 1. Huấn luyện GPT (S1 Training)
- [ ] Cấu hình `tmp_s1.yaml` (ép `num_workers=0` để tránh crash).
- [ ] Chạy lệnh train GPT fine-tuning.
- [ ] Theo dõi độ chính xác (Accuracy) và Loss.

### 2. Suy luận và Kiểm tra (Inference)
- [ ] Xuất file trọng số cuối cùng (`.pth` cho SoVITS và `.ckpt` cho GPT).
- [ ] Sử dụng tab Inference để tạo giọng nói tiếng Việt từ văn bản mới.
- [ ] Đánh giá độ tự nhiên và cảm xúc.

### **Summary: Vietnamese TTS Fine-Tuning (`mc_vi_voice`) - SUCCESS**

#### **1. Achievements & Final Results**
*   **GPT Weight Export**: FIXED. Weights now include the necessary `model.` prefix.
*   **Vietnamese Support**: FIXED. Patched `TTS.py` and `TextPreprocessor.py` to support `vi` language and skip `LangSegmenter` (avoiding external downloads).
*   **Inference Pipeline**: SUCCESS. Generated `output_test.wav` using the fine-tuned model.
*   **Environment**: Stable on Windows with `PYTHONIOENCODING=utf-8`.

#### **2. Final Files**
*   **Audio Output**: `output_test.wav` (Giọng nói tiếng Việt do AI tạo ra).
*   **Weights**: 
    *   GPT: `GPT_weights_v2/mc_vi_voice-e15.ckpt`
    *   SoVITS: `SoVITS_weights_v2/mc_vi_voice_e8_s2016.pth`

---

## ⚠️ Lưu ý kỹ thuật mới
- **Windows Lock**: Luôn sử dụng cơ chế Retry khi lưu file trên Windows để tránh crash giữa chừng.
- **Batch Size**: Giữ mức 4 cho SoVITS và 8 cho GPT để an toàn cho VRAM 8GB.
- **Version**: Đang sử dụng Model **v2** để đảm bảo chất lượng tiếng Việt tốt nhất.

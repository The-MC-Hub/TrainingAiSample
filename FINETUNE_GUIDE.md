# HƯỚNG DẪN FINE-TUNE GIỌNG NÓI MC (THE MC HUB AI CORE)

Chào Trung, để thực hiện Fine-tune cho giọng nói MC tiêu chuẩn, chúng ta sẽ đi qua 3 giai đoạn chính:

## GIAI ĐOẠN 1: Chuẩn bị môi trường (Cần thiết cho Windows)
Do hệ thống cần xử lý âm thanh MP3/WAV, bạn cần cài đặt `ffmpeg`. 
1. Mở PowerShell với quyền Admin.
2. Chạy lệnh: `winget install ffmpeg`
3. Sau khi cài xong, khởi động lại VS Code/Terminal.

## GIAI ĐOẠN 2: Tiền xử lý dữ liệu (Data Preprocessing)
Tôi đã viết sẵn file `preprocess_audio.py`. File này sẽ thực hiện:
*   Sử dụng **Whisper** để nghe và dịch toàn bộ audio MC thành văn bản.
*   Tự động cắt (slice) audio thành các đoạn nhỏ 2-10 giây.
*   Tạo file `metadata.txt` chuẩn để đưa vào huấn luyện.

**Cách chạy:**
```bash
python preprocess_audio.py
```
*Kết quả sẽ nằm trong thư mục `./training_data`.*

## GIAI ĐOẠN 3: Chạy Fine-tune (Huấn luyện)
Vì bạn có GPU **RTX 4060 (8GB VRAM)**, chúng ta sẽ sử dụng công nghệ **GPT-SoVITS** (tốt nhất cho việc nhái giọng hiện nay).

### Bước 3.1: Tải bộ Tool Fine-tune
```bash
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS
pip install -r requirements.txt
```

### Bước 3.2: Copy dữ liệu đã chuẩn bị
Copy nội dung thư mục `training_data` (từ bước 2) vào thư mục `GPT-SoVITS/dataset/`.

### Bước 3.3: Chạy WebUI để huấn luyện
```bash
python webui.py
```
1. Mở trình duyệt vào link WebUI hiện ra.
2. Tại tab **"Fine-tuning"**, chọn đường dẫn đến thư mục `dataset`.
3. Nhấn **"Start Fine-tuning"**. Với 8GB VRAM, quá trình này mất khoảng 20-30 phút cho 1 vòng (Epoch).

## GIAI ĐOẠN 4: Triển khai (Deploy)
Sau khi train xong, bạn sẽ nhận được file `.ckpt`. Hãy copy nó vào thư mục `models/` của dự án này và cập nhật đường dẫn trong `main.py`.

---
**Tôi đã chuẩn bị sẵn mọi script hỗ trợ trong dự án của bạn rồi! Bạn hãy bắt đầu bằng việc cài đặt ffmpeg nhé.**

# 🧭 AI Intelligence & Codegen Guide (TrainingAiSample)

FastAPI, working dev source cho AI voice service. **Đọc [CLAUDE.md](./CLAUDE.md) trước** — đặc biệt phần "Critical: this repo's `main.py` currently lacks TTS endpoints" và "Relationship to HF-Space-Deploy".

---

## PHẦN 1: 🧠 Code Intelligence

- **Impact Analysis**: `grep "^@app\."` trong `main.py` trước khi sửa route để biết chính xác endpoint nào tồn tại (đừng tin README nếu chưa verify). Kiểm tra script nào (`finetune_whisper.py`, `preprocess_audio.py`, ...) gọi hàm chung trước khi sửa signature.
- **Detect Changes**: `git status`/`git diff --stat` trước commit — repo có `.env` **checked in** (không phải gitignored), cẩn thận không commit thêm secret mới vào đó nếu repo có thể public.
- Không tự ý copy thay đổi từ đây sang `HF-Space-Deploy` — đó là sync thủ công, phải xác nhận rõ ràng migration TTS đã hoàn tất chưa trước khi port.

## PHẦN 2: 🪨 Giao tiếp tinh gọn

Caveman Mode nếu được bật ở session — ngắn gọn, chỉ in dòng thay đổi.

## PHẦN 3: ⚡ Nguyên tắc hiệu suất

1. Batch xử lý audio khi có thể — tránh load lại model trong loop (`Whisper`/`GPT-SoVITS` model load 1 lần, tái sử dụng object).
2. Threshold cấu hình qua biến môi trường (`.env`) — không hard-code ngưỡng SNR/pitch/pause trong code khi đã có convention dùng env var.
3. Endpoint FastAPI trả response nhanh — công việc nặng (fine-tune, preprocess) chạy qua script riêng (`finetune_whisper.py`, ...), không nhét vào request handler.

## PHẦN 4: 🔄 Quy trình phát triển tiêu chuẩn

1. **Plan & Specify**: xác nhận route/hàm cần sửa còn tồn tại đúng như mô tả (grep trực tiếp, không tin tài liệu cũ). Nếu thay đổi ảnh hưởng tới cả `HF-Space-Deploy`, nói rõ với người dùng đây là thay đổi 1 phía (upstream only) trừ khi được yêu cầu port luôn.
2. **Code & Refactor**: giữ style hiện có của `main.py` (flat, không tách module trừ khi được yêu cầu refactor kiến trúc). Windows-console UTF-8 fix ở đầu file — không xóa khi không hiểu tại sao nó ở đó.
3. **Test & Verify**: không có test suite tự động trong repo này. Verify bằng cách chạy `python -m uvicorn main:app --reload` cục bộ + `curl` endpoint vừa sửa xác nhận response đúng shape trước khi báo hoàn thành. Nếu sửa pipeline training (`finetune_whisper.py`, ...), chạy thử với tập dữ liệu nhỏ trước khi coi là xong.
4. **Commit**: `git status`/`git diff --stat` xác nhận scope. Conventional Commits, không gộp thay đổi `.env` (secret) chung với code nếu không cần thiết — tách riêng và cảnh báo người dùng.

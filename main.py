from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import whisper
import jiwer
import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile
import numpy as np
import librosa
from fastapi.middleware.cors import CORSMiddleware

# Tu dong setup FFmpeg cho moi truong Window
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load Model Whisper (STT)
print("Dang load model STT (Whisper)...")
stt_model = whisper.load_model("base")

# 2. Load Model TTS (MC Voice)
TTS_MODEL_PATH = "./models/mms-tts-vie"
print("Dang check model TTS...")
tts_model = None
tts_tokenizer = None

if os.path.exists(TTS_MODEL_PATH):
    print("Dang load model TTS (MC Voice)...")
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_PATH)
    tts_model = VitsModel.from_pretrained(TTS_MODEL_PATH)
else:
    print("Model TTS chua duoc tai. Hay chay download_model.py truoc.")

print("Da load xong cac model AI!")

@app.post("/analyze-voice")
async def analyze_voice(file: UploadFile = File(...), script_origin: str = Form(...)):
    # --- GIAI DOAN 1: LUU FILE TAM ---
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # --- GIAI DOAN 2: STT (Speech to Text) ---
        # Model nghe va chuyen thanh chu
        result = stt_model.transcribe(temp_filename)
        text_spoken = result["text"]

        # --- GIAI DOAN 3: CHAM DIEM NANG CAO (Advanced Scoring) ---
        audio_data, sr = librosa.load(temp_filename)
        duration = librosa.get_duration(y=audio_data, sr=sr)
        word_count = len(text_spoken.split())
        
        # 3.1. Accuracy Score (WER)
        error_rate = jiwer.wer(script_origin.lower(), text_spoken.lower())
        accuracy_score = max(0, 100 - (error_rate * 100))

        # 3.2. Speaking Rate (WPM)
        wpm = (word_count / duration) * 60 if duration > 0 else 0

        # 3.3. Phan tich Nhip dieu (Rhythm Variance) - NEW
        # Chia am thanh thanh cac doan nho va tinh nang luong
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        rhythm_score = np.std(onset_env) # Do bien thien nang luong (nhan nha)
        
        # Chuan hoa diem nhip dieu ve thang 100
        normalized_rhythm = min(100, rhythm_score * 20) 

        # 3.4. MC Feedback Logic (Nang cap)
        feedback = []
        if accuracy_score > 90:
            feedback.append("Phát âm: Chuẩn xác tuyệt vời.")
        else:
            feedback.append("Phát âm: Cần rõ chữ hơn, đặc biệt là các âm cuối.")

        if normalized_rhythm > 40:
            feedback.append("Nhấn nhá: Tốt! Bạn có sự thay đổi âm lượng tạo cảm xúc.")
        else:
            feedback.append("Nhấn nhá: Giọng còn hơi đều (monotone). Hãy thử nhấn mạnh vào các từ quan trọng.")

        # --- GIAI DOAN 4: EXPERT TIPS (Deep Analysis) ---
        expert_tips = [
            {"label": "Năng lượng", "tip": f"Cường độ âm thanh của bạn đạt {round(rhythm_score, 2)}. Đối với MC sự kiện, hãy tăng độ biến thiên âm lượng để cuốn hút hơn."},
            {"label": "Ngắt nghỉ", "tip": "AI phát hiện bạn thường nghỉ 0.3s giữa các câu. Hãy thử kéo dài lên 0.6s ở các đoạn chuyển ý."}
        ]

        # --- GIAI DOAN 5: TRA KET QUA ---
        return {
            "status": "success",
            "text_spoken": text_spoken,
            "accuracy_score": float(round(accuracy_score, 2)),
            "rhythm_score": float(round(normalized_rhythm, 2)),
            "speaking_rate_wpm": float(round(wpm, 2)),
            "feedback": " | ".join(feedback),
            "expert_tips": expert_tips
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        # Xoa file tam sau khi xu ly xong de nhe may
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# API: Tao giong noi MC (TTS)
@app.post("/generate-mc-voice")
async def generate_mc_voice(text: str = Form(...)):
    if tts_model is None or tts_tokenizer is None:
        return {"status": "error", "message": "Model TTS chưa được load. Vui lòng kiểm tra file ./models/mms-tts-vie"}
    
    try:
        inputs = tts_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = tts_model(**inputs).waveform
        
        # Chuyen output sang file wav
        output_filename = "mc_voice_output.wav"
        scipy.io.wavfile.write(output_filename, rate=tts_model.config.sampling_rate, data=output.cpu().numpy().T)
        
        return {
            "status": "success",
            "message": "Đã tạo xong giọng nói MC",
            "file_path": output_filename
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# API Test de biet Server song hay chet
@app.get("/")
def read_root():
    return {"message": "Hello Trung! MC Hub AI Service is running with TTS & STT support!"}


# ================================================================
#  Entry point — bypass Python 3.13 signal handling bug on Windows
#  Chạy bằng: python main.py
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        loop="asyncio",       # Fix lỗi signal.raise_signal trên Python 3.13 Windows
        log_level="info",
    )

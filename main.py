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

# ================================================================
#  GPU Detection — RTX 4060 will be picked up automatically
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[AI] Running on: {device.upper()}")
if device == "cuda":
    print(f"[AI] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[AI] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ================================================================
#  1. Load Whisper STT — 'small' is optimal for RTX 4060 (8GB)
#     'small' = ~15% more accurate than 'base' with Vietnamese
# ================================================================
print("[AI] Loading STT model (Whisper small)...")
stt_model = whisper.load_model("small", device=device)
print(f"[AI] Whisper loaded on {device.upper()}")

# ================================================================
#  2. Load TTS Model — MMS-TTS Vietnamese
# ================================================================
TTS_MODEL_PATH = "./models/mms-tts-vie"
print("[AI] Checking TTS model...")
tts_model = None
tts_tokenizer = None

if os.path.exists(TTS_MODEL_PATH):
    print("[AI] Loading TTS model (MMS-TTS-VIE)...")
    tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_PATH)
    tts_model = VitsModel.from_pretrained(TTS_MODEL_PATH).to(device)
    print(f"[AI] TTS loaded on {device.upper()}")
else:
    print("[AI] TTS model not found. Run download_model.py first.")

print("[AI] All models ready!")


# ================================================================
#  Helper: Compute real pause statistics from audio
# ================================================================
def compute_pause_stats(audio_data: np.ndarray, sr: int) -> dict:
    """
    Detects silence intervals in the audio and returns:
    - avg_pause_sec: average pause duration between sentences
    - max_pause_sec: longest pause
    - pause_count: number of detected pauses
    """
    # Compute RMS energy in short frames
    frame_length = int(0.025 * sr)   # 25ms window
    hop_length   = int(0.010 * sr)   # 10ms hop
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]

    # Threshold = 5% of mean energy → silence
    silence_threshold = np.mean(rms) * 0.05
    is_silent = rms < silence_threshold

    # Find contiguous silent regions
    pauses = []
    in_pause = False
    pause_start = 0
    for i, silent in enumerate(is_silent):
        t = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
        if silent and not in_pause:
            in_pause = True
            pause_start = t
        elif not silent and in_pause:
            in_pause = False
            duration = t - pause_start
            if duration >= 0.15:   # ignore micro-pauses < 150ms
                pauses.append(duration)

    if not pauses:
        return {"avg_pause_sec": 0.0, "max_pause_sec": 0.0, "pause_count": 0}

    return {
        "avg_pause_sec": round(float(np.mean(pauses)), 2),
        "max_pause_sec": round(float(np.max(pauses)), 2),
        "pause_count": len(pauses),
    }


# ================================================================
#  Helper: Build bilingual expert tips and reports
# ================================================================
def generate_bilingual_evaluation(rhythm_score: float, pause_stats: dict, wpm: float, accuracy_score: float) -> dict:
    avg_pause = pause_stats["avg_pause_sec"]
    # --- VI logic ---
    feedback_vi = []
    tips_vi = []
    
    if accuracy_score > 90:
        feedback_vi.append("Phát âm: Độ chính xác tuyệt vời.")
    elif accuracy_score > 70:
        feedback_vi.append("Phát âm: Tốt — cần cải thiện nhẹ ở các âm cuối.")
    else:
        feedback_vi.append("Phát âm: Cần cố gắng hơn — tập trung vào độ rõ nét của phụ âm cuối.")

    if rhythm_score > 40:
        feedback_vi.append("Nhấn nhá: Biến điệu tốt — cách dẫn dắt lôi cuốn.")
    else:
        feedback_vi.append("Nhấn nhá: Giọng còn hơi đều — hãy thử nhấn mạnh vào các từ khóa.")

    avg_pause = pause_stats["avg_pause_sec"]
    if avg_pause > 0:
        if avg_pause < 0.4:
            feedback_vi.append(f"Ngắt nghỉ: Quá ngắn ({avg_pause}s) — hãy nghỉ 0.5-0.8s giữa các câu.")
        elif avg_pause <= 0.9:
            feedback_vi.append(f"Ngắt nghỉ: Nhịp điệu tốt ({avg_pause}s).")
        else:
            feedback_vi.append(f"Ngắt nghỉ: Quá dài ({avg_pause}s) — hãy đẩy nhanh tốc độ chuyển câu.")

    # Tips VI
    if wpm < 100:
        tips_vi.append({"label": "TỐC ĐỘ", "tip": f"Tốc độ đọc chậm ({wpm:.0f} WPM). Mục tiêu 120–160 WPM để tự tin hơn."})
    elif wpm <= 165:
        tips_vi.append({"label": "TỐC ĐỘ", "tip": f"Tốc độ lý tưởng ({wpm:.0f} WPM). Phù hợp cho MC sự kiện."})
    else:
        tips_vi.append({"label": "TỐC ĐỘ", "tip": f"Tốc độ nhanh ({wpm:.0f} WPM). Hãy nói chậm lại ở những đoạn quan trọng."})

    # --- EN logic ---
    feedback_en = []
    tips_en = []
    
    if accuracy_score > 90:
        feedback_en.append("Pronunciation: Excellent accuracy.")
    elif accuracy_score > 70:
        feedback_en.append("Pronunciation: Good — minor articulation improvements needed.")
    else:
        feedback_en.append("Pronunciation: Needs work — focus on final consonants.")

    if rhythm_score > 40:
        feedback_en.append("Emphasis: Good dynamic variation.")
    else:
        feedback_en.append("Emphasis: Slightly monotone — try stressing key words.")

    if avg_pause > 0:
        if avg_pause < 0.4:
            feedback_en.append(f"Pausing: Too short ({avg_pause}s) — target 0.5-0.8s.")
        elif avg_pause <= 0.9:
            feedback_en.append(f"Pausing: Good timing ({avg_pause}s).")
        else:
            feedback_en.append(f"Pausing: Too long ({avg_pause}s) — maintain momentum.")

    # Tips EN
    if wpm < 100:
        tips_en.append({"label": "PACING", "tip": f"Speaking pace is slow ({wpm:.0f} WPM). Target 120–160 WPM."})
    elif wpm <= 165:
        tips_en.append({"label": "PACING", "tip": f"Pace is ideal ({wpm:.0f} WPM). Perfect for MC authority."})
    else:
        tips_en.append({"label": "PACING", "tip": f"Pace is fast ({wpm:.0f} WPM). Slow down for comprehension."})

    # --- Diagnostic Logic ---
    pace_status_vi = "Ổn định" if 115 <= wpm <= 165 else ("Hơi nhanh" if wpm > 165 else "Hơi chậm")
    pace_status_en = "Optimal" if 115 <= wpm <= 165 else ("Fast" if wpm > 165 else "Slow")
    
    accuracy_status_vi = "Sắc nét" if accuracy_score > 85 else ("Khá" if accuracy_score > 70 else "Cần cải thiện")
    accuracy_status_en = "Sharp" if accuracy_score > 85 else ("Fair" if accuracy_score > 70 else "Needs Work")
    
    dynamics_status_vi = "Truyền cảm" if rhythm_score > 50 else ("Ổn" if rhythm_score > 30 else "Hơi đều")
    dynamics_status_en = "Expressive" if rhythm_score > 50 else ("Steady" if rhythm_score > 30 else "Monotone")

    # --- Dynamic Action Plans ---
    actions_vi = []
    actions_en = []

    # 1. Articulation focus
    if accuracy_score < 80:
        actions_vi.append("1. **Luyện kỹ thuật 'Over-enunciation'**: Hãy đọc chậm lại và cường điệu hóa các phụ âm cuối (t, k, n, m) để cơ hàm quen với việc phát âm rõ nét.")
        actions_en.append("1. **Over-enunciation Drill**: Slow down and exaggerate final consonants (t, k, n, m) to train your jaw for sharper clarity.")
    elif accuracy_score < 90:
        actions_vi.append("1. **Tinh chỉnh âm sắc**: Tập trung vào các từ có dấu thanh phức tạp (hỏi, ngã) để đảm bảo độ vang đồng đều.")
        actions_en.append("1. **Tonal Refinement**: Focus on complex tonal transitions to ensure consistent resonance across all syllables.")
    else:
        actions_vi.append("1. **Duy trì độ sắc nét**: Bạn đã phát âm tốt, hãy thử duy trì độ rõ này khi tăng tốc độ đọc lên 10%.")
        actions_en.append("1. **Clarity Maintenance**: Your articulation is excellent; try maintaining this sharpness while increasing speed by 10%.")

    # 2. Dynamics/Rhythm focus
    if rhythm_score < 35:
        actions_vi.append("2. **Kỹ thuật 'High-Low Stress'**: Gạch chân các từ quan trọng và tập nói chúng với cao độ lớn hơn các từ còn lại để phá vỡ sự đơn điệu.")
        actions_en.append("2. **High-Low Stress Technique**: Underline key words and practice speaking them at a higher pitch than surrounding words to break monotony.")
    elif rhythm_score < 60:
        actions_vi.append("2. **Tăng cường biểu cảm**: Hãy tưởng tượng bạn đang kể một câu chuyện thú vị, thêm cảm xúc hào hứng vào các tính từ miêu tả.")
        actions_en.append("2. **Emotional Layering**: Imagine telling an exciting story; add enthusiastic inflection specifically to descriptive adjectives.")
    else:
        actions_vi.append("2. **Kiểm soát năng lượng**: Biến điệu của bạn rất tốt, hãy giữ mức năng lượng này xuyên suốt các đoạn văn dài.")
        actions_en.append("2. **Energy Management**: Your dynamics are great; focus on sustaining this energy level through longer paragraphs.")

    # 3. Pacing/Pausing focus
    if wpm > 165:
        actions_vi.append("3. **Quản lý khoảng lặng**: Tập đếm nhẩm '1' giữa các dấu phẩy và '1, 2' giữa các dấu chấm để khán giả kịp hấp thụ thông tin.")
        actions_en.append("3. **Silence Management**: Practice a silent '1' count at commas and '1, 2' at periods to give the audience time to absorb information.")
    elif wpm < 115:
        actions_vi.append("3. **Kỹ thuật 'Flow & Momentum'**: Đọc kịch bản như một dòng chảy liên tục, giảm bớt các khoảng nghỉ không cần thiết giữa các từ đơn lẻ.")
        actions_en.append("3. **Flow & Momentum**: Read the script as a continuous stream, minimizing unnecessary micro-pauses between individual words.")
    else:
        actions_vi.append("3. **Kỹ thuật 'Strategic Pause'**: Sử dụng khoảng lặng 0.5s ngay trước các thông tin quan trọng nhất để tạo sự kịch tính.")
        actions_en.append("3. **Strategic Pausing**: Insert a 0.5s silence immediately before the most critical information to create a sense of anticipation.")

    # --- Markdown Reports ---
    report_vi = f"""
### 🎙️ Báo cáo Phân tích Chuyên sâu (AI Expert)
**Đánh giá tổng thể:** Bạn đạt **{accuracy_score:.1f}%** độ chính xác. Giọng đọc có tiềm năng nhưng cần tinh chỉnh các yếu tố kỹ thuật sau:

#### 📈 Phân tích Kỹ thuật (Technical Analysis):
| Tiêu chí | Trạng thái | Chỉ số thực tế | Mục tiêu MC |
| :--- | :--- | :--- | :--- |
| **Phát âm** | {accuracy_status_vi} | {accuracy_score:.1f}% | > 90% |
| **Tốc độ** | {pace_status_vi} | {wpm:.0f} WPM | 130 - 150 WPM |
| **Nhấn nhá** | {dynamics_status_vi} | {rhythm_score:.1f}/100 | > 50.0 |
| **Ngắt nghỉ** | {"Hợp lý" if 0.4 <= avg_pause <= 0.8 else "Chưa ổn"} | {avg_pause}s avg | 0.5s - 0.7s |

#### 🔍 Chẩn đoán chi tiết:
- **Về Phát âm:** {
    "Bạn phát âm rất rõ ràng, các phụ âm cuối và dấu thanh được thể hiện sắc nét." if accuracy_score > 85 else
    "Lưu ý các âm cuối (t, n, ch) thường bị nuốt khi nói nhanh. Hãy mở khẩu hình miệng rộng hơn."
}
- **Về Nhịp điệu:** {
    "Sự biến thiên cao độ tốt, tạo được sự lôi cuốn cho người nghe." if rhythm_score > 50 else
    "Giọng đọc hiện tại hơi bằng phẳng (monotone). Hãy tập trung nhấn mạnh vào các 'Key Words' như địa danh, thời gian, và tên riêng."
}
- **Về Tốc độ & Ngắt nghỉ:** {
    "Tốc độ kiểm soát tốt, tạo cảm giác chuyên nghiệp." if 115 <= wpm <= 165 else
    f"Tốc độ {wpm:.0f} WPM là {pace_status_vi}. {'Nên nói chậm lại để khán giả kịp hấp thụ thông tin.' if wpm > 165 else 'Cần đẩy nhanh nhịp điệu để tạo sự hào hứng cho sự kiện.'}"
}

#### 💡 Hành động cải thiện (Action Plan):
{"".join([f"{a}\n" for a in actions_vi])}
"""

    report_en = f"""
### 🎙️ Advanced AI Performance Report
**Overall Assessment:** Your delivery achieved **{accuracy_score:.1f}%** accuracy. Your voice shows great potential with the following technical refinements recommended:

#### 📈 Technical Analysis:
| Metric | Status | Actual Value | MC Standard |
| :--- | :--- | :--- | :--- |
| **Articulation** | {accuracy_status_en} | {accuracy_score:.1f}% | > 90% |
| **Pacing** | {pace_status_en} | {wpm:.0f} WPM | 130 - 150 WPM |
| **Dynamics** | {dynamics_status_en} | {rhythm_score:.1f}/100 | > 50.0 |
| **Pausing** | {"Optimal" if 0.4 <= avg_pause <= 0.8 else "Suboptimal"} | {avg_pause}s avg | 0.5s - 0.7s |

#### 🔍 Detailed Diagnosis:
- **Articulation:** {
    "Your speech is highly intelligible with sharp final consonants and clear tonal markers." if accuracy_score > 85 else
    "Watch your final consonants (t, k, n). They tend to blur during faster segments. Practice wider jaw movement."
}
- **Dynamics:** {
    "Excellent pitch variation. Your delivery is engaging and keeps the audience focused." if rhythm_score > 50 else
    "Current delivery is slightly monotone. Focus on 'Stressing Key Words' such as dates, locations, and proper names."
}
- **Pacing & Flow:** {
    "Pacing is well-controlled, projecting a professional and confident image." if 115 <= wpm <= 165 else
    f"At {wpm:.0f} WPM, you are speaking too {pace_status_en.lower()}. {'Slow down to let the audience absorb key information.' if wpm > 165 else 'Increase your tempo to build excitement and energy.'}"
}

#### 💡 Improvement Plan:
{"".join([f"{a}\n" for a in actions_en])}
"""

    return {
        "feedback_vi": " | ".join(feedback_vi),
        "feedback_en": " | ".join(feedback_en),
        "tips_vi": tips_vi,
        "tips_en": tips_en,
        "report_vi": report_vi.strip(),
        "report_en": report_en.strip()
    }


# ================================================================
#  API: Analyze Voice
# ================================================================
@app.post("/analyze-voice")
async def analyze_voice(file: UploadFile = File(...), script_origin: str = Form(...)):
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # --- Stage 1: STT (Speech to Text) ---
        # Use GPU via autocast for faster inference
        # language=None allows Whisper to auto-detect (useful for mixed EN/VI testing)
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            result = stt_model.transcribe(temp_filename, language=None)
        text_spoken = result["text"]

        # --- Stage 2: Audio Analysis ---
        audio_data, sr = librosa.load(temp_filename, sr=16000)
        duration = librosa.get_duration(y=audio_data, sr=sr)
        word_count = len(text_spoken.split())

        # 2.1 Accuracy (WER-based)
        error_rate = jiwer.wer(script_origin.lower(), text_spoken.lower())
        accuracy_score = max(0, 100 - (error_rate * 100))

        # 2.2 Speaking Rate (WPM)
        wpm = (word_count / duration) * 60 if duration > 0 else 0

        # 2.3 Rhythm Score — onset strength variance (normalized 0–100)
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        rhythm_raw = float(np.std(onset_env))
        normalized_rhythm = min(100.0, rhythm_raw * 20)

        # 2.4 Real pause statistics
        pause_stats = compute_pause_stats(audio_data, sr)

        # --- Stage 3: Bilingual Evaluation ---
        eval_data = generate_bilingual_evaluation(normalized_rhythm, pause_stats, wpm, accuracy_score)

        # --- Stage 4: Return Result ---
        return {
            "status": "success",
            "text_spoken": text_spoken,
            "accuracy_score": float(round(accuracy_score, 2)),
            "rhythm_score": float(round(normalized_rhythm, 2)),
            "speaking_rate_wpm": float(round(wpm, 2)),
            # Bilingual fields
            "feedback": eval_data["feedback_vi"], # legacy support
            "feedback_vi": eval_data["feedback_vi"],
            "feedback_en": eval_data["feedback_en"],
            "expert_tips": eval_data["tips_vi"], # legacy support
            "tips_vi": eval_data["tips_vi"],
            "tips_en": eval_data["tips_en"],
            "report_vi": eval_data["report_vi"],
            "report_en": eval_data["report_en"],
            "analysis_meta": {
                "device_used": device,
                "avg_pause_sec": pause_stats["avg_pause_sec"],
                "pause_count": pause_stats["pause_count"],
                "duration_sec": round(duration, 2),
            }
        }

    except Exception as e:
        print(f"[AI] ERROR: {str(e)}")
        return {"status": "error", "message": str(e)}

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# ================================================================
#  API: Generate MC Voice (TTS)
# ================================================================
@app.post("/generate-mc-voice")
async def generate_mc_voice(text: str = Form(...)):
    if tts_model is None or tts_tokenizer is None:
        return {"status": "error", "message": "TTS model not loaded. Check ./models/mms-tts-vie"}

    try:
        inputs = tts_tokenizer(text, return_tensors="pt")
        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                output = tts_model(**inputs).waveform

        output_filename = "mc_voice_output.wav"
        scipy.io.wavfile.write(
            output_filename,
            rate=tts_model.config.sampling_rate,
            data=output.cpu().numpy().T
        )

        return {
            "status": "success",
            "message": "MC voice generated successfully",
            "file_path": output_filename
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ================================================================
#  Health Check
# ================================================================
@app.get("/")
def read_root():
    return {
        "message": "MC Hub AI Service is running",
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "whisper_model": "small",
        "tts_loaded": tts_model is not None,
    }


# ================================================================
#  Entry point — bypass Python 3.13 signal handling bug on Windows
#  Run: python main.py
# ================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        loop="asyncio",
        log_level="info",
    )

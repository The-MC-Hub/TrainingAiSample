from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import uuid
import whisper
import jiwer
import torch
from transformers import VitsModel, AutoTokenizer
import scipy.io.wavfile
import numpy as np
import librosa
import json
from typing import Optional
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
    frame_length = int(0.025 * sr)
    hop_length   = int(0.010 * sr)
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]

    silence_threshold = np.mean(rms) * 0.05
    is_silent = rms < silence_threshold

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
            if duration >= 0.15:
                pauses.append(duration)

    if not pauses:
        return {"avg_pause_sec": 0.0, "max_pause_sec": 0.0, "pause_count": 0}

    return {
        "avg_pause_sec": round(float(np.mean(pauses)), 2),
        "max_pause_sec": round(float(np.max(pauses)), 2),
        "pause_count": len(pauses),
    }


# ================================================================
#  Helper: Compute per-criterion scores
# ================================================================
def compute_pacing_score(wpm: float, target_min: int, target_max: int) -> float:
    """Score 0-100 based on how close WPM is to target range."""
    if target_min <= wpm <= target_max:
        return 100.0
    range_span = max(1, target_max - target_min)
    distance = (target_min - wpm) if wpm < target_min else (wpm - target_max)
    penalty = min(100.0, (distance / range_span) * 100.0)
    return round(max(0.0, 100.0 - penalty), 2)


def compute_criteria_scores(
    accuracy_score: float,
    normalized_rhythm: float,
    wpm: float,
    target_wpm_min: int,
    target_wpm_max: int,
    criteria: list,
) -> dict:
    """
    Map each criterion aspect to a computed score.
    Falls back to base metrics when no criteria are defined.
    """
    pacing_score = compute_pacing_score(wpm, target_wpm_min, target_wpm_max)

    # Emotion score: rhythm-derived, boosted when well above threshold
    emotion_score = min(100.0, normalized_rhythm * 1.1)

    aspect_map = {
        "PRONUNCIATION": round(accuracy_score, 2),
        "ACCURACY":      round(accuracy_score, 2),
        "RHYTHM":        round(normalized_rhythm, 2),
        "EMOTION":       round(emotion_score, 2),
        "PACING":        pacing_score,
    }

    if not criteria:
        return {k: v for k, v in aspect_map.items()}

    result = {}
    for c in criteria:
        aspect = c.get("aspect", "").upper()
        result[aspect] = aspect_map.get(aspect, 0.0)
    return result


def compute_overall_score(criteria_scores: dict, criteria: list) -> float:
    """Weighted average. Falls back to mean of all scores when no criteria."""
    if not criteria or not criteria_scores:
        if criteria_scores:
            return round(sum(criteria_scores.values()) / len(criteria_scores), 2)
        return 0.0

    total_weight = sum(c.get("weight", 0) for c in criteria)
    if total_weight == 0:
        return 0.0

    weighted_sum = 0.0
    for c in criteria:
        aspect = c.get("aspect", "").upper()
        score = criteria_scores.get(aspect, 0.0)
        weighted_sum += score * c.get("weight", 0)

    return round(weighted_sum / total_weight, 2)


# ================================================================
#  Helper: Build bilingual expert tips and reports
# ================================================================
def generate_bilingual_evaluation(
    rhythm_score: float,
    pause_stats: dict,
    wpm: float,
    accuracy_score: float,
    target_wpm_min: int,
    target_wpm_max: int,
    evaluation_hint: str,
    criteria: list,
    criteria_scores: dict,
    overall_score: float,
) -> dict:
    avg_pause = pause_stats["avg_pause_sec"]
    target_center = (target_wpm_min + target_wpm_max) // 2

    # ---- Vietnamese feedback ----
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

    if avg_pause > 0:
        if avg_pause < 0.4:
            feedback_vi.append(f"Ngắt nghỉ: Quá ngắn ({avg_pause}s) — hãy nghỉ 0.5–0.8s giữa các câu.")
        elif avg_pause <= 0.9:
            feedback_vi.append(f"Ngắt nghỉ: Nhịp điệu tốt ({avg_pause}s).")
        else:
            feedback_vi.append(f"Ngắt nghỉ: Quá dài ({avg_pause}s) — hãy đẩy nhanh tốc độ chuyển câu.")

    if wpm < target_wpm_min:
        tips_vi.append({"label": "TỐC ĐỘ", "tip": f"Tốc độ đọc chậm ({wpm:.0f} WPM). Mục tiêu {target_wpm_min}–{target_wpm_max} WPM."})
    elif wpm <= target_wpm_max:
        tips_vi.append({"label": "TỐC ĐỘ", "tip": f"Tốc độ lý tưởng ({wpm:.0f} WPM). Phù hợp tiêu chuẩn bài học."})
    else:
        tips_vi.append({"label": "TỐC ĐỘ", "tip": f"Tốc độ nhanh ({wpm:.0f} WPM). Hãy nói chậm lại ở những đoạn quan trọng."})

    # Per-criterion tips (VI)
    for aspect, score in criteria_scores.items():
        if score < 60:
            tips_vi.append({"label": aspect, "tip": f"Tiêu chí {aspect} đạt {score:.0f}/100 — cần cải thiện thêm."})

    # ---- English feedback ----
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
            feedback_en.append(f"Pausing: Too short ({avg_pause}s) — target 0.5–0.8s.")
        elif avg_pause <= 0.9:
            feedback_en.append(f"Pausing: Good timing ({avg_pause}s).")
        else:
            feedback_en.append(f"Pausing: Too long ({avg_pause}s) — maintain momentum.")

    if wpm < target_wpm_min:
        tips_en.append({"label": "PACING", "tip": f"Speaking pace is slow ({wpm:.0f} WPM). Target {target_wpm_min}–{target_wpm_max} WPM."})
    elif wpm <= target_wpm_max:
        tips_en.append({"label": "PACING", "tip": f"Pace is ideal ({wpm:.0f} WPM). Matches lesson standard."})
    else:
        tips_en.append({"label": "PACING", "tip": f"Pace is fast ({wpm:.0f} WPM). Slow down for comprehension."})

    for aspect, score in criteria_scores.items():
        if score < 60:
            tips_en.append({"label": aspect, "tip": f"{aspect} scored {score:.0f}/100 — needs improvement."})

    # ---- Status labels ----
    pace_ok = target_wpm_min <= wpm <= target_wpm_max
    pace_status_vi = "Ổn định" if pace_ok else ("Hơi nhanh" if wpm > target_wpm_max else "Hơi chậm")
    pace_status_en = "Optimal"  if pace_ok else ("Fast"      if wpm > target_wpm_max else "Slow")

    accuracy_status_vi = "Sắc nét" if accuracy_score > 85 else ("Khá" if accuracy_score > 70 else "Cần cải thiện")
    accuracy_status_en = "Sharp"   if accuracy_score > 85 else ("Fair" if accuracy_score > 70 else "Needs Work")

    dynamics_status_vi = "Truyền cảm" if rhythm_score > 50 else ("Ổn" if rhythm_score > 30 else "Hơi đều")
    dynamics_status_en = "Expressive" if rhythm_score > 50 else ("Steady" if rhythm_score > 30 else "Monotone")

    # ---- Action plans ----
    actions_vi = []
    actions_en = []

    if accuracy_score < 80:
        actions_vi.append("1. **Luyện kỹ thuật 'Over-enunciation'**: Đọc chậm và cường điệu hóa các phụ âm cuối (t, k, n, m).")
        actions_en.append("1. **Over-enunciation Drill**: Slow down and exaggerate final consonants to train jaw clarity.")
    elif accuracy_score < 90:
        actions_vi.append("1. **Tinh chỉnh âm sắc**: Tập trung vào các từ có dấu thanh phức tạp.")
        actions_en.append("1. **Tonal Refinement**: Focus on complex tonal transitions for consistent resonance.")
    else:
        actions_vi.append("1. **Duy trì độ sắc nét**: Hãy thử duy trì độ rõ này khi tăng tốc độ đọc lên 10%.")
        actions_en.append("1. **Clarity Maintenance**: Maintain sharpness while gradually increasing speed by 10%.")

    if rhythm_score < 35:
        actions_vi.append("2. **Kỹ thuật 'High-Low Stress'**: Gạch chân các từ quan trọng và tập nói với cao độ lớn hơn.")
        actions_en.append("2. **High-Low Stress Technique**: Underline key words and practice stressing them at higher pitch.")
    elif rhythm_score < 60:
        actions_vi.append("2. **Tăng cường biểu cảm**: Thêm cảm xúc vào các tính từ miêu tả.")
        actions_en.append("2. **Emotional Layering**: Add enthusiastic inflection to descriptive adjectives.")
    else:
        actions_vi.append("2. **Kiểm soát năng lượng**: Duy trì mức năng lượng xuyên suốt các đoạn văn dài.")
        actions_en.append("2. **Energy Management**: Sustain this energy level through longer paragraphs.")

    if wpm > target_wpm_max:
        actions_vi.append("3. **Quản lý khoảng lặng**: Tập đếm nhẩm '1' giữa dấu phẩy và '1, 2' giữa dấu chấm.")
        actions_en.append("3. **Silence Management**: Practice a silent '1' count at commas, '1, 2' at periods.")
    elif wpm < target_wpm_min:
        actions_vi.append("3. **Kỹ thuật 'Flow & Momentum'**: Giảm khoảng nghỉ không cần thiết giữa các từ đơn lẻ.")
        actions_en.append("3. **Flow & Momentum**: Minimize unnecessary micro-pauses between individual words.")
    else:
        actions_vi.append("3. **Kỹ thuật 'Strategic Pause'**: Sử dụng khoảng lặng 0.5s trước thông tin quan trọng.")
        actions_en.append("3. **Strategic Pausing**: Insert a 0.5s silence before the most critical information.")

    # ---- Criteria score table rows ----
    criteria_rows_vi = ""
    criteria_rows_en = ""
    if criteria_scores:
        for aspect, score in criteria_scores.items():
            status_vi = "Tốt" if score >= 80 else ("Ổn" if score >= 60 else "Cần cải thiện")
            status_en = "Good" if score >= 80 else ("Fair" if score >= 60 else "Needs Work")
            weight = next((c.get("weight", 0) for c in criteria if c.get("aspect", "").upper() == aspect), 0)
            criteria_rows_vi += f"| **{aspect}** | {status_vi} | {score:.1f}/100 | {weight}% |\n"
            criteria_rows_en += f"| **{aspect}** | {status_en} | {score:.1f}/100 | {weight}% |\n"

    hint_line_vi = f"\n> 💡 **Lưu ý bài học:** {evaluation_hint}" if evaluation_hint else ""
    hint_line_en = f"\n> 💡 **Lesson Note:** {evaluation_hint}" if evaluation_hint else ""

    report_vi = f"""### 🎙️ Báo cáo Phân tích Chuyên sâu (AI Expert){hint_line_vi}
**Đánh giá tổng thể:** Điểm tổng hợp **{overall_score:.1f}/100** · Độ chính xác phát âm **{accuracy_score:.1f}%**

#### 📊 Điểm theo tiêu chí:
| Tiêu chí | Trạng thái | Điểm | Trọng số |
| :--- | :--- | :--- | :--- |
{criteria_rows_vi if criteria_rows_vi else f"| **Phát âm** | {accuracy_status_vi} | {accuracy_score:.1f}/100 | — |\n| **Nhịp điệu** | {dynamics_status_vi} | {rhythm_score:.1f}/100 | — |\n| **Tốc độ** | {pace_status_vi} | {wpm:.0f} WPM | — |\n"}
#### 📈 Phân tích kỹ thuật:
| Tiêu chí | Trạng thái | Chỉ số thực tế | Mục tiêu |
| :--- | :--- | :--- | :--- |
| **Phát âm** | {accuracy_status_vi} | {accuracy_score:.1f}% | > 90% |
| **Tốc độ** | {pace_status_vi} | {wpm:.0f} WPM | {target_wpm_min}–{target_wpm_max} WPM |
| **Nhấn nhá** | {dynamics_status_vi} | {rhythm_score:.1f}/100 | > 50.0 |
| **Ngắt nghỉ** | {"Hợp lý" if 0.4 <= avg_pause <= 0.8 else "Chưa ổn"} | {avg_pause}s avg | 0.5s–0.7s |

#### 💡 Hành động cải thiện:
{"".join([f"{a}\n" for a in actions_vi])}""".strip()

    report_en = f"""### 🎙️ Advanced AI Performance Report{hint_line_en}
**Overall Score:** **{overall_score:.1f}/100** · Pronunciation Accuracy **{accuracy_score:.1f}%**

#### 📊 Per-Criterion Scores:
| Criterion | Status | Score | Weight |
| :--- | :--- | :--- | :--- |
{criteria_rows_en if criteria_rows_en else f"| **Pronunciation** | {accuracy_status_en} | {accuracy_score:.1f}/100 | — |\n| **Dynamics** | {dynamics_status_en} | {rhythm_score:.1f}/100 | — |\n| **Pacing** | {pace_status_en} | {wpm:.0f} WPM | — |\n"}
#### 📈 Technical Analysis:
| Metric | Status | Actual Value | MC Standard |
| :--- | :--- | :--- | :--- |
| **Articulation** | {accuracy_status_en} | {accuracy_score:.1f}% | > 90% |
| **Pacing** | {pace_status_en} | {wpm:.0f} WPM | {target_wpm_min}–{target_wpm_max} WPM |
| **Dynamics** | {dynamics_status_en} | {rhythm_score:.1f}/100 | > 50.0 |
| **Pausing** | {"Optimal" if 0.4 <= avg_pause <= 0.8 else "Suboptimal"} | {avg_pause}s avg | 0.5s–0.7s |

#### 💡 Improvement Plan:
{"".join([f"{a}\n" for a in actions_en])}""".strip()

    return {
        "feedback_vi": " | ".join(feedback_vi),
        "feedback_en": " | ".join(feedback_en),
        "tips_vi": tips_vi,
        "tips_en": tips_en,
        "report_vi": report_vi,
        "report_en": report_en,
    }


# ================================================================
#  API: Analyze Voice
# ================================================================
@app.post("/analyze-voice")
async def analyze_voice(
    file: UploadFile = File(...),
    script_origin: str = Form(...),
    target_wpm_min: int = Form(120),
    target_wpm_max: int = Form(150),
    evaluation_hint: Optional[str] = Form(None),
    evaluation_criteria_json: Optional[str] = Form(None),
):
    # Parse evaluation criteria
    criteria = []
    if evaluation_criteria_json:
        try:
            criteria = json.loads(evaluation_criteria_json)
        except Exception as e:
            print(f"[AI] Warning: failed to parse evaluation_criteria_json: {e}")

    original_ext = os.path.splitext(file.filename or "")[1].lower()
    if original_ext not in (".webm", ".ogg", ".wav", ".mp3", ".m4a", ".mp4"):
        original_ext = ".webm"
    temp_filename = f"temp_{uuid.uuid4().hex}{original_ext}"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"[AI] Received file: {file.filename!r} → saved as {temp_filename} ({os.path.getsize(temp_filename)} bytes)")
    print(f"[AI] WPM target: {target_wpm_min}–{target_wpm_max} | hint: {evaluation_hint!r} | criteria: {len(criteria)} items")

    try:
        # --- Stage 1: STT ---
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            result = stt_model.transcribe(temp_filename, language=None)
        text_spoken = result["text"]

        # --- Stage 2: Audio Analysis ---
        audio_data, sr = librosa.load(temp_filename, sr=16000)
        duration = librosa.get_duration(y=audio_data, sr=sr)
        word_count = len(text_spoken.split())

        error_rate = jiwer.wer(script_origin.lower(), text_spoken.lower())
        accuracy_score = max(0, 100 - (error_rate * 100))

        wpm = (word_count / duration) * 60 if duration > 0 else 0

        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        rhythm_raw = float(np.std(onset_env))
        normalized_rhythm = min(100.0, rhythm_raw * 20)

        pause_stats = compute_pause_stats(audio_data, sr)

        # --- Stage 3: Per-criterion scoring ---
        criteria_scores = compute_criteria_scores(
            accuracy_score, normalized_rhythm, wpm,
            target_wpm_min, target_wpm_max, criteria
        )
        overall_score = compute_overall_score(criteria_scores, criteria)

        # --- Stage 4: Bilingual evaluation ---
        eval_data = generate_bilingual_evaluation(
            normalized_rhythm, pause_stats, wpm, accuracy_score,
            target_wpm_min, target_wpm_max, evaluation_hint or "",
            criteria, criteria_scores, overall_score,
        )

        # --- Stage 5: Return result ---
        return {
            "status": "success",
            "text_spoken": text_spoken,
            "accuracy_score": float(round(accuracy_score, 2)),
            "rhythm_score": float(round(normalized_rhythm, 2)),
            "speaking_rate_wpm": float(round(wpm, 2)),
            "criteria_scores": criteria_scores,
            "overall_score": overall_score,
            # Bilingual fields
            "feedback":    eval_data["feedback_vi"],  # legacy
            "feedback_vi": eval_data["feedback_vi"],
            "feedback_en": eval_data["feedback_en"],
            "expert_tips": eval_data["tips_vi"],      # legacy
            "tips_vi": eval_data["tips_vi"],
            "tips_en": eval_data["tips_en"],
            "report_vi": eval_data["report_vi"],
            "report_en": eval_data["report_en"],
            "analysis_meta": {
                "device_used":    device,
                "avg_pause_sec":  pause_stats["avg_pause_sec"],
                "pause_count":    pause_stats["pause_count"],
                "duration_sec":   round(duration, 2),
                "target_wpm_min": target_wpm_min,
                "target_wpm_max": target_wpm_max,
            },
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
            "file_path": output_filename,
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
#  Entry point
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

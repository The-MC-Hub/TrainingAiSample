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
from dotenv import load_dotenv

# Load .env (file nằm cùng thư mục với main.py)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Tu dong setup FFmpeg cho moi truong Window
try:
    import static_ffmpeg
    static_ffmpeg.add_paths()
except ImportError:
    pass

# ── Config từ .env ────────────────────────────────────────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
ALLOWED_ORIGINS_LIST = [o.strip() for o in _raw_origins.split(",") if o.strip()]

SNR_NOISE_THRESHOLD_DB     = float(os.getenv("SNR_NOISE_THRESHOLD_DB",     "15"))
ENERGY_FADE_WARN_THRESHOLD = float(os.getenv("ENERGY_FADE_WARN_THRESHOLD", "0.5"))
ENERGY_FADE_ALERT_THRESHOLD= float(os.getenv("ENERGY_FADE_ALERT_THRESHOLD","0.4"))
PITCH_MONOTONE_THRESHOLD   = float(os.getenv("PITCH_MONOTONE_THRESHOLD",   "1.0"))
PITCH_EXPRESSIVE_THRESHOLD = float(os.getenv("PITCH_EXPRESSIVE_THRESHOLD", "2.0"))
PAUSE_MIN_GOOD_SEC         = float(os.getenv("PAUSE_MIN_GOOD_SEC",         "0.4"))
PAUSE_MAX_GOOD_SEC         = float(os.getenv("PAUSE_MAX_GOOD_SEC",         "0.9"))

# Hugging Face token (nếu có)
_hf_token = os.getenv("HF_TOKEN", "").strip()
if _hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    os.environ["HF_TOKEN"] = _hf_token

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_LIST,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
#  GPU Detection
# ================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[AI] Running on: {device.upper()}")
if device == "cuda":
    print(f"[AI] GPU: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[AI] VRAM: {vram_gb:.1f} GB")
else:
    vram_gb = 0

# ================================================================
#  1. Load Whisper STT
#     WHISPER_MODEL=AUTO → tự chọn theo VRAM
#       ≥6GB → large-v3 | 4-6GB → medium | CPU/thấp → small
#     WHISPER_MODEL=<tên cụ thể> → dùng đúng tên đó
# ================================================================
def _choose_whisper_model(vram: float) -> str:
    if device == "cuda":
        if vram >= 6.0:
            return "large-v3"
        if vram >= 4.0:
            return "medium"
    return "small"

_whisper_env = os.getenv("WHISPER_MODEL", "AUTO").strip()
if _whisper_env.upper() == "AUTO":
    whisper_model_name = _choose_whisper_model(vram_gb)
    print(f"[AI] WHISPER_MODEL=AUTO → selected '{whisper_model_name}' (VRAM={vram_gb:.1f}GB)")
else:
    whisper_model_name = _whisper_env
    print(f"[AI] WHISPER_MODEL override from .env → '{whisper_model_name}'")

print(f"[AI] Loading STT model (Whisper {whisper_model_name})...")
stt_model = whisper.load_model(whisper_model_name, device=device)
print(f"[AI] Whisper {whisper_model_name} loaded on {device.upper()}")

# ================================================================
#  2. Load TTS Model — MMS-TTS Vietnamese
# ================================================================
TTS_MODEL_PATH = os.getenv("TTS_MODEL_PATH", "./models/mms-tts-vie")
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
        return {"avg_pause_sec": 0.0, "max_pause_sec": 0.0, "pause_count": 0, "total_pause_sec": 0.0}

    return {
        "avg_pause_sec":   round(float(np.mean(pauses)), 2),
        "max_pause_sec":   round(float(np.max(pauses)), 2),
        "pause_count":     len(pauses),
        "total_pause_sec": round(float(sum(pauses)), 2),
    }


# ================================================================
#  Helper: Pitch (F0) analysis — intonation expressiveness
# ================================================================
def compute_pitch_stats(audio_data: np.ndarray, sr: int) -> dict:
    """
    Extract fundamental frequency (F0) using librosa pyin.
    Returns pitch_std_semitones: std deviation of voiced F0 in semitones.
    High std = expressive intonation. Low std = monotone.
    MC target: >= 3 semitones std is considered expressive.
    """
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz("C2"),  # ~65 Hz — floor for human voice
            fmax=librosa.note_to_hz("C7"),  # ~2093 Hz — ceiling
            sr=sr,
            frame_length=2048,
            hop_length=512,
        )
        voiced_f0 = f0[voiced_flag & (f0 > 0)]
        if len(voiced_f0) < 10:
            return {"pitch_std_semitones": 0.0, "pitch_mean_hz": 0.0, "voiced_ratio": 0.0}

        # Convert Hz to semitones (log scale) — std in semitones is speaker-independent
        f0_semitones = 12 * np.log2(voiced_f0 / 440.0)
        pitch_std = float(np.std(f0_semitones))
        pitch_mean = float(np.mean(voiced_f0))
        voiced_ratio = len(voiced_f0) / max(1, len(f0))

        return {
            "pitch_std_semitones": round(pitch_std, 3),
            "pitch_mean_hz":       round(pitch_mean, 1),
            "voiced_ratio":        round(voiced_ratio, 3),
        }
    except Exception as e:
        print(f"[AI] Pitch analysis error: {e}")
        return {"pitch_std_semitones": 0.0, "pitch_mean_hz": 0.0, "voiced_ratio": 0.0}


# ================================================================
#  Helper: SNR estimation — noise quality gate
# ================================================================
def compute_snr(audio_data: np.ndarray, sr: int) -> float:
    """
    Estimate signal-to-noise ratio in dB.
    Uses first 0.3s as noise floor estimate (assumes mic capture starts quiet).
    Returns SNR dB — higher is cleaner. < 15 dB = noisy recording.
    """
    try:
        noise_samples = int(0.3 * sr)
        if len(audio_data) < noise_samples * 2:
            return 30.0  # too short to estimate, assume clean

        noise_floor = np.mean(audio_data[:noise_samples] ** 2)
        signal_power = np.mean(audio_data[noise_samples:] ** 2)

        if noise_floor <= 0 or signal_power <= 0:
            return 30.0

        snr_db = 10 * np.log10(signal_power / noise_floor)
        return round(float(np.clip(snr_db, 0.0, 60.0)), 1)
    except Exception:
        return 30.0


# ================================================================
#  Helper: Vocal energy fade detection
# ================================================================
def compute_energy_profile(audio_data: np.ndarray, sr: int) -> dict:
    """
    Detect if voice energy fades at end of sentences.
    Split audio into 5 equal segments, compute RMS per segment.
    fade_score: ratio of last-segment RMS vs max-segment RMS.
    < 0.5 = noticeable fade (common issue for untrained MCs).
    """
    try:
        n_segments = 5
        seg_len = len(audio_data) // n_segments
        if seg_len < sr * 0.2:  # segment < 0.2s, too short
            return {"fade_score": 1.0, "energy_segments": []}

        rms_per_seg = []
        for i in range(n_segments):
            seg = audio_data[i * seg_len: (i + 1) * seg_len]
            rms_per_seg.append(float(np.sqrt(np.mean(seg ** 2))))

        max_rms = max(rms_per_seg) if max(rms_per_seg) > 0 else 1.0
        fade_score = rms_per_seg[-1] / max_rms

        return {
            "fade_score": round(fade_score, 3),
            "energy_segments": [round(v / max_rms, 3) for v in rms_per_seg],
        }
    except Exception:
        return {"fade_score": 1.0, "energy_segments": []}


# ================================================================
#  Helper: Compute pitch-based expressiveness score (0-100)
# ================================================================
def pitch_to_expressiveness_score(pitch_std_semitones: float) -> float:
    """
    MC expressiveness rubric based on F0 standard deviation in semitones:
      < 1.0  → monotone        (0–25)
      1-2    → slightly varied (25–55)
      2-4    → good MC range   (55–85)
      4+     → very expressive (85–100, capped)
    """
    if pitch_std_semitones >= 5.0:
        return 100.0
    if pitch_std_semitones >= 4.0:
        return 85.0 + (pitch_std_semitones - 4.0) * 15.0
    if pitch_std_semitones >= 2.0:
        return 55.0 + (pitch_std_semitones - 2.0) * 15.0
    if pitch_std_semitones >= 1.0:
        return 25.0 + (pitch_std_semitones - 1.0) * 30.0
    return pitch_std_semitones * 25.0


# ================================================================
#  Helper: Compute per-criterion scores
# ================================================================
def compute_pacing_score(wpm: float, target_min: int, target_max: int) -> float:
    if target_min <= wpm <= target_max:
        return 100.0
    range_span = max(1, target_max - target_min)
    distance = (target_min - wpm) if wpm < target_min else (wpm - target_max)
    penalty = min(100.0, (distance / range_span) * 100.0)
    return round(max(0.0, 100.0 - penalty), 2)


def compute_criteria_scores(
    accuracy_score: float,
    expressiveness_score: float,   # pitch-based (replaces raw rhythm)
    onset_variation_score: float,  # onset std (rhythm/beat energy)
    wpm: float,
    target_wpm_min: int,
    target_wpm_max: int,
    criteria: list,
) -> dict:
    pacing_score  = compute_pacing_score(wpm, target_wpm_min, target_wpm_max)

    aspect_map = {
        "PRONUNCIATION": round(accuracy_score, 2),
        "ACCURACY":      round(accuracy_score, 2),
        "RHYTHM":        round(onset_variation_score, 2),   # beat/stress variation
        "EMOTION":       round(expressiveness_score, 2),    # pitch intonation
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
        score  = criteria_scores.get(aspect, 0.0)
        weighted_sum += score * c.get("weight", 0)

    return round(weighted_sum / total_weight, 2)


# ================================================================
#  Helper: Build bilingual expert tips and reports
# ================================================================
def generate_bilingual_evaluation(
    expressiveness_score: float,  # pitch-based
    onset_variation_score: float, # beat/stress
    pitch_stats: dict,
    pause_stats: dict,
    energy_profile: dict,
    snr_db: float,
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
    pitch_std = pitch_stats["pitch_std_semitones"]
    fade_score = energy_profile["fade_score"]
    target_center = (target_wpm_min + target_wpm_max) // 2

    # ---- Vietnamese feedback ----
    feedback_vi = []
    tips_vi = []

    # Pronunciation (WER-based)
    if accuracy_score > 90:
        feedback_vi.append("Phát âm: Độ chính xác tuyệt vời.")
    elif accuracy_score > 70:
        feedback_vi.append("Phát âm: Tốt — cần cải thiện nhẹ ở các âm cuối.")
    else:
        feedback_vi.append("Phát âm: Cần cố gắng hơn — tập trung vào độ rõ nét của phụ âm cuối.")

    # Intonation (pitch-based)
    if pitch_std >= 4.0:
        feedback_vi.append("Giọng điệu: Rất biểu cảm — nhấn nhá xuất sắc, giọng MC chuyên nghiệp.")
    elif pitch_std >= PITCH_EXPRESSIVE_THRESHOLD:
        feedback_vi.append("Giọng điệu: Biến điệu tốt — cách dẫn dắt lôi cuốn.")
    elif pitch_std >= PITCH_MONOTONE_THRESHOLD:
        feedback_vi.append("Giọng điệu: Hơi đều — thử nhấn mạnh từ khóa và lên/xuống giọng rõ hơn.")
    else:
        feedback_vi.append("Giọng điệu: Giọng đơn điệu — cần luyện biến thiên cao độ nhiều hơn.")

    # Pause
    if avg_pause > 0:
        if avg_pause < PAUSE_MIN_GOOD_SEC:
            feedback_vi.append(f"Ngắt nghỉ: Quá ngắn ({avg_pause}s) — hãy nghỉ {PAUSE_MIN_GOOD_SEC}–{PAUSE_MAX_GOOD_SEC}s giữa các câu.")
        elif avg_pause <= PAUSE_MAX_GOOD_SEC:
            feedback_vi.append(f"Ngắt nghỉ: Nhịp điệu tốt ({avg_pause}s).")
        else:
            feedback_vi.append(f"Ngắt nghỉ: Quá dài ({avg_pause}s) — hãy đẩy nhanh tốc độ chuyển câu.")

    # Energy fade
    if fade_score < ENERGY_FADE_ALERT_THRESHOLD:
        feedback_vi.append("Năng lượng: Giọng bị tắt dần cuối bài — duy trì năng lượng đến câu cuối.")
    elif fade_score < ENERGY_FADE_WARN_THRESHOLD:
        feedback_vi.append("Năng lượng: Năng lượng hơi giảm cuối — cố gắng giữ đều hơn.")

    # SNR noise warning
    if snr_db < SNR_NOISE_THRESHOLD_DB:
        feedback_vi.append(f"Chất lượng âm: Phát hiện tiếng ồn nền (SNR {snr_db:.0f}dB) — kết quả có thể bị ảnh hưởng.")

    # WPM tip
    if wpm < target_wpm_min:
        tips_vi.append({"label": "TỐC ĐỘ", "tip": f"Tốc độ đọc chậm ({wpm:.0f} WPM). Mục tiêu {target_wpm_min}–{target_wpm_max} WPM."})
    elif wpm <= target_wpm_max:
        tips_vi.append({"label": "TỐC ĐỘ", "tip": f"Tốc độ lý tưởng ({wpm:.0f} WPM). Phù hợp tiêu chuẩn bài học."})
    else:
        tips_vi.append({"label": "TỐC ĐỘ", "tip": f"Tốc độ nhanh ({wpm:.0f} WPM). Hãy nói chậm lại ở những đoạn quan trọng."})

    # Pitch tip
    if pitch_std < PITCH_EXPRESSIVE_THRESHOLD:
        tips_vi.append({"label": "GIỌNG ĐIỆU", "tip": f"Biến thiên cao độ thấp ({pitch_std:.1f} semitone). Luyện đọc với cường độ cảm xúc tăng dần."})

    # Per-criterion tips
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

    if pitch_std >= 4.0:
        feedback_en.append("Intonation: Highly expressive — excellent pitch variation, professional MC delivery.")
    elif pitch_std >= PITCH_EXPRESSIVE_THRESHOLD:
        feedback_en.append("Intonation: Good dynamic variation — engaging delivery.")
    elif pitch_std >= PITCH_MONOTONE_THRESHOLD:
        feedback_en.append("Intonation: Slightly flat — try emphasizing key words with stronger pitch shifts.")
    else:
        feedback_en.append("Intonation: Monotone delivery — practice pitch range exercises.")

    if avg_pause > 0:
        if avg_pause < PAUSE_MIN_GOOD_SEC:
            feedback_en.append(f"Pausing: Too short ({avg_pause}s) — target {PAUSE_MIN_GOOD_SEC}–{PAUSE_MAX_GOOD_SEC}s.")
        elif avg_pause <= PAUSE_MAX_GOOD_SEC:
            feedback_en.append(f"Pausing: Good timing ({avg_pause}s).")
        else:
            feedback_en.append(f"Pausing: Too long ({avg_pause}s) — maintain momentum.")

    if fade_score < ENERGY_FADE_ALERT_THRESHOLD:
        feedback_en.append("Energy: Voice fades significantly at end — sustain energy through final sentences.")
    elif fade_score < ENERGY_FADE_WARN_THRESHOLD:
        feedback_en.append("Energy: Slight energy drop at end — aim for consistent volume.")

    if snr_db < SNR_NOISE_THRESHOLD_DB:
        feedback_en.append(f"Audio quality: Background noise detected (SNR {snr_db:.0f}dB) — results may be affected.")

    if wpm < target_wpm_min:
        tips_en.append({"label": "PACING", "tip": f"Speaking pace is slow ({wpm:.0f} WPM). Target {target_wpm_min}–{target_wpm_max} WPM."})
    elif wpm <= target_wpm_max:
        tips_en.append({"label": "PACING", "tip": f"Pace is ideal ({wpm:.0f} WPM). Matches lesson standard."})
    else:
        tips_en.append({"label": "PACING", "tip": f"Pace is fast ({wpm:.0f} WPM). Slow down for comprehension."})

    if pitch_std < PITCH_EXPRESSIVE_THRESHOLD:
        tips_en.append({"label": "INTONATION", "tip": f"Pitch variation is low ({pitch_std:.1f} semitones). Practice reading with deliberate emotional intensity."})

    for aspect, score in criteria_scores.items():
        if score < 60:
            tips_en.append({"label": aspect, "tip": f"{aspect} scored {score:.0f}/100 — needs improvement."})

    # ---- Status labels ----
    pace_ok = target_wpm_min <= wpm <= target_wpm_max
    pace_status_vi = "Ổn định" if pace_ok else ("Hơi nhanh" if wpm > target_wpm_max else "Hơi chậm")
    pace_status_en = "Optimal"  if pace_ok else ("Fast"      if wpm > target_wpm_max else "Slow")

    accuracy_status_vi = "Sắc nét"        if accuracy_score > 85 else ("Khá"      if accuracy_score > 70 else "Cần cải thiện")
    accuracy_status_en = "Sharp"          if accuracy_score > 85 else ("Fair"     if accuracy_score > 70 else "Needs Work")

    intonation_status_vi = "Rất biểu cảm" if pitch_std >= 4.0 else ("Biểu cảm" if pitch_std >= PITCH_EXPRESSIVE_THRESHOLD else ("Hơi đều" if pitch_std >= PITCH_MONOTONE_THRESHOLD else "Đơn điệu"))
    intonation_status_en = "Expressive"   if pitch_std >= 4.0 else ("Dynamic"  if pitch_std >= PITCH_EXPRESSIVE_THRESHOLD else ("Slight"  if pitch_std >= PITCH_MONOTONE_THRESHOLD else "Monotone"))

    fade_status_vi = "Đều" if fade_score >= ENERGY_FADE_WARN_THRESHOLD else ("Hơi tắt" if fade_score >= ENERGY_FADE_ALERT_THRESHOLD else "Tắt dần")
    fade_status_en = "Sustained" if fade_score >= ENERGY_FADE_WARN_THRESHOLD else ("Slight fade" if fade_score >= ENERGY_FADE_ALERT_THRESHOLD else "Fading")

    noise_status_vi = "Sạch" if snr_db >= (SNR_NOISE_THRESHOLD_DB + 5) else ("Chấp nhận" if snr_db >= SNR_NOISE_THRESHOLD_DB else "Ồn")
    noise_status_en = "Clean" if snr_db >= (SNR_NOISE_THRESHOLD_DB + 5) else ("Acceptable" if snr_db >= SNR_NOISE_THRESHOLD_DB else "Noisy")

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

    if pitch_std < PITCH_MONOTONE_THRESHOLD + 0.5:
        actions_vi.append("2. **Kỹ thuật 'Pitch Ladder'**: Đọc một câu 3 lần — lần 1 thấp, lần 2 trung bình, lần 3 cao. Cảm nhận sự khác biệt.")
        actions_en.append("2. **Pitch Ladder Drill**: Read one sentence 3×: low, mid, high pitch. Internalize the contrast.")
    elif pitch_std < PITCH_EXPRESSIVE_THRESHOLD + 1.0:
        actions_vi.append("2. **Tăng cường biểu cảm**: Gạch chân các từ khóa và tập nói với cao độ lớn hơn 30%.")
        actions_en.append("2. **Emphasis Boost**: Underline key words and pitch them 30% higher than surrounding speech.")
    else:
        actions_vi.append("2. **Kiểm soát biểu cảm**: Duy trì biến thiên giọng xuyên suốt — tránh để năng lượng tắt dần.")
        actions_en.append("2. **Expression Control**: Sustain pitch variation throughout — avoid energy tailing off.")

    if wpm > target_wpm_max:
        actions_vi.append("3. **Quản lý khoảng lặng**: Tập đếm nhẩm '1' giữa dấu phẩy và '1, 2' giữa dấu chấm.")
        actions_en.append("3. **Silence Management**: Practice a silent '1' count at commas, '1, 2' at periods.")
    elif wpm < target_wpm_min:
        actions_vi.append("3. **Kỹ thuật 'Flow & Momentum'**: Giảm khoảng nghỉ không cần thiết giữa các từ đơn lẻ.")
        actions_en.append("3. **Flow & Momentum**: Minimize unnecessary micro-pauses between individual words.")
    else:
        actions_vi.append("3. **Kỹ thuật 'Strategic Pause'**: Sử dụng khoảng lặng 0.5s trước thông tin quan trọng.")
        actions_en.append("3. **Strategic Pausing**: Insert a 0.5s silence before the most critical information.")

    if fade_score < 0.5:
        actions_vi.append("4. **Kỹ thuật 'Final Push'**: Tưởng tượng câu cuối là câu quan trọng nhất — đẩy thêm năng lượng.")
        actions_en.append("4. **Final Push Technique**: Treat the last sentence as the most important — add energy.")

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
{criteria_rows_vi if criteria_rows_vi else f"| **Phát âm** | {accuracy_status_vi} | {accuracy_score:.1f}/100 | — |\n| **Giọng điệu** | {intonation_status_vi} | {expressiveness_score:.1f}/100 | — |\n| **Tốc độ** | {pace_status_vi} | {wpm:.0f} WPM | — |\n"}
#### 📈 Phân tích kỹ thuật:
| Tiêu chí | Trạng thái | Chỉ số thực tế | Mục tiêu |
| :--- | :--- | :--- | :--- |
| **Phát âm** | {accuracy_status_vi} | {accuracy_score:.1f}% | > 90% |
| **Tốc độ** | {pace_status_vi} | {wpm:.0f} WPM | {target_wpm_min}–{target_wpm_max} WPM |
| **Giọng điệu (F0)** | {intonation_status_vi} | {pitch_std:.2f} semitone | ≥ 2.0 semitone |
| **Ngắt nghỉ** | {"Hợp lý" if PAUSE_MIN_GOOD_SEC <= avg_pause <= PAUSE_MAX_GOOD_SEC else "Chưa ổn"} | {avg_pause}s avg | {PAUSE_MIN_GOOD_SEC}s–{PAUSE_MAX_GOOD_SEC}s |
| **Năng lượng cuối** | {fade_status_vi} | {fade_score:.0%} | ≥ {ENERGY_FADE_WARN_THRESHOLD:.0%} |
| **Chất lượng âm** | {noise_status_vi} | SNR {snr_db:.0f} dB | ≥ {SNR_NOISE_THRESHOLD_DB + 5:.0f} dB |

#### 💡 Hành động cải thiện:
{"".join([f"{a}\n" for a in actions_vi])}""".strip()

    report_en = f"""### 🎙️ Advanced AI Performance Report{hint_line_en}
**Overall Score:** **{overall_score:.1f}/100** · Pronunciation Accuracy **{accuracy_score:.1f}%**

#### 📊 Per-Criterion Scores:
| Criterion | Status | Score | Weight |
| :--- | :--- | :--- | :--- |
{criteria_rows_en if criteria_rows_en else f"| **Pronunciation** | {accuracy_status_en} | {accuracy_score:.1f}/100 | — |\n| **Intonation** | {intonation_status_en} | {expressiveness_score:.1f}/100 | — |\n| **Pacing** | {pace_status_en} | {wpm:.0f} WPM | — |\n"}
#### 📈 Technical Analysis:
| Metric | Status | Actual Value | MC Standard |
| :--- | :--- | :--- | :--- |
| **Articulation** | {accuracy_status_en} | {accuracy_score:.1f}% | > 90% |
| **Pacing** | {pace_status_en} | {wpm:.0f} WPM | {target_wpm_min}–{target_wpm_max} WPM |
| **Intonation (F0)** | {intonation_status_en} | {pitch_std:.2f} semitones | ≥ 2.0 semitones |
| **Pausing** | {"Optimal" if PAUSE_MIN_GOOD_SEC <= avg_pause <= PAUSE_MAX_GOOD_SEC else "Suboptimal"} | {avg_pause}s avg | {PAUSE_MIN_GOOD_SEC}s–{PAUSE_MAX_GOOD_SEC}s |
| **Energy Sustain** | {fade_status_en} | {fade_score:.0%} | ≥ {ENERGY_FADE_WARN_THRESHOLD:.0%} |
| **Audio Quality** | {noise_status_en} | SNR {snr_db:.0f} dB | ≥ {SNR_NOISE_THRESHOLD_DB + 5:.0f} dB |

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

    print(f"[AI] Received: {file.filename!r} → {temp_filename} ({os.path.getsize(temp_filename)} bytes)")
    print(f"[AI] WPM target: {target_wpm_min}–{target_wpm_max} | criteria: {len(criteria)} items")

    try:
        # ── Stage 1: STT (Whisper) ──
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            result = stt_model.transcribe(temp_filename, language=None)
        text_spoken = result["text"]
        print(f"[AI] Transcribed: {text_spoken[:120]!r}...")

        # ── Stage 2: Load audio ──
        audio_data, sr = librosa.load(temp_filename, sr=16000)
        duration_total = librosa.get_duration(y=audio_data, sr=sr)

        # ── Stage 3: Pause stats (needed for speech-only duration) ──
        pause_stats = compute_pause_stats(audio_data, sr)
        total_pause = pause_stats["total_pause_sec"]

        # Speech-only duration: exclude silence from WPM denominator
        duration_speech = max(0.1, duration_total - total_pause)
        word_count = len(text_spoken.split())

        # ── Stage 4: Core metrics ──
        error_rate     = jiwer.wer(script_origin.lower(), text_spoken.lower())
        accuracy_score = max(0.0, 100.0 - (error_rate * 100.0))
        wpm            = (word_count / duration_speech) * 60.0  # speech-only duration

        # Onset strength variation (beat/stress energy — kept as RHYTHM dimension)
        onset_env           = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onset_variation_raw = float(np.std(onset_env))
        onset_variation_score = min(100.0, onset_variation_raw * 20.0)

        # ── Stage 5: Pitch (F0) analysis — intonation expressiveness ──
        pitch_stats         = compute_pitch_stats(audio_data, sr)
        expressiveness_score = pitch_to_expressiveness_score(pitch_stats["pitch_std_semitones"])

        # ── Stage 6: SNR — noise quality ──
        snr_db = compute_snr(audio_data, sr)

        # ── Stage 7: Energy profile — fade detection ──
        energy_profile = compute_energy_profile(audio_data, sr)

        print(f"[AI] accuracy={accuracy_score:.1f}% | wpm={wpm:.0f} | pitch_std={pitch_stats['pitch_std_semitones']:.2f}st | "
              f"expressiveness={expressiveness_score:.1f} | onset_var={onset_variation_score:.1f} | "
              f"snr={snr_db:.1f}dB | fade={energy_profile['fade_score']:.2f}")

        # ── Stage 8: Per-criterion scoring ──
        criteria_scores = compute_criteria_scores(
            accuracy_score, expressiveness_score, onset_variation_score,
            wpm, target_wpm_min, target_wpm_max, criteria,
        )
        overall_score = compute_overall_score(criteria_scores, criteria)

        # ── Stage 9: Bilingual evaluation ──
        eval_data = generate_bilingual_evaluation(
            expressiveness_score, onset_variation_score,
            pitch_stats, pause_stats, energy_profile, snr_db,
            wpm, accuracy_score,
            target_wpm_min, target_wpm_max,
            evaluation_hint or "", criteria, criteria_scores, overall_score,
        )

        return {
            "status": "success",
            "text_spoken":       text_spoken,
            "accuracy_score":    float(round(accuracy_score, 2)),
            "rhythm_score":      float(round(expressiveness_score, 2)),  # frontend compat: rhythm_score now = expressiveness
            "speaking_rate_wpm": float(round(wpm, 2)),
            "criteria_scores":   criteria_scores,
            "overall_score":     overall_score,
            # Extended metrics
            "pitch_stats":       pitch_stats,
            "energy_profile":    energy_profile,
            "snr_db":            snr_db,
            "onset_variation":   float(round(onset_variation_score, 2)),
            # Bilingual fields
            "feedback":    eval_data["feedback_vi"],  # legacy
            "feedback_vi": eval_data["feedback_vi"],
            "feedback_en": eval_data["feedback_en"],
            "expert_tips": eval_data["tips_vi"],      # legacy
            "tips_vi":     eval_data["tips_vi"],
            "tips_en":     eval_data["tips_en"],
            "report_vi":   eval_data["report_vi"],
            "report_en":   eval_data["report_en"],
            "analysis_meta": {
                "device_used":       device,
                "whisper_model":     whisper_model_name,
                "avg_pause_sec":     pause_stats["avg_pause_sec"],
                "pause_count":       pause_stats["pause_count"],
                "total_pause_sec":   total_pause,
                "duration_total_sec": round(duration_total, 2),
                "duration_speech_sec": round(duration_speech, 2),
                "target_wpm_min":    target_wpm_min,
                "target_wpm_max":    target_wpm_max,
                "snr_db":            snr_db,
                "fade_score":        energy_profile["fade_score"],
            },
        }

    except Exception as e:
        print(f"[AI] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
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
            data=output.cpu().numpy().T,
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
        "device":        device,
        "gpu":           torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "whisper_model": whisper_model_name,
        "tts_loaded":    tts_model is not None,
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

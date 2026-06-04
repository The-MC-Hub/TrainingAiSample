# -*- coding: utf-8 -*-
import sys, io as _io; sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from fastapi import FastAPI, UploadFile, File, Form
import io
import soundfile as sf
import shutil
import os
import uuid
import whisper
import jiwer
import torch
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
    print(f"[AI] WHISPER_MODEL=AUTO -> selected '{whisper_model_name}' (VRAM={vram_gb:.1f}GB)")
else:
    whisper_model_name = _whisper_env
    print(f"[AI] WHISPER_MODEL override from .env -> '{whisper_model_name}'")

print(f"[AI] Loading STT model (Whisper {whisper_model_name})...")
stt_model = whisper.load_model(whisper_model_name, device=device)
print(f"[AI] Whisper {whisper_model_name} loaded on {device.upper()}")

print("[AI] All models ready!")


# ================================================================
#  Helper: Compute real pause statistics from audio
# ================================================================
def compute_pause_stats(audio_data: np.ndarray, sr: int) -> dict:
    frame_length = int(0.025 * sr)
    hop_length   = int(0.010 * sr)
    rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]

    silence_threshold = float(np.percentile(rms, 10))
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
    Estimate SNR using adaptive noise floor — percentile 5% of frame power.
    More robust than fixed 0.3s window: works even when speaker starts immediately.
    Returns SNR dB (0–60). < 15 dB = noisy recording.
    """
    try:
        hop = int(0.010 * sr)
        frame = int(0.025 * sr)
        rms = librosa.feature.rms(y=audio_data, frame_length=frame, hop_length=hop)[0]
        power = rms ** 2

        noise_floor = float(np.percentile(power, 5))
        signal_power = float(np.percentile(power, 75))

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
    Detect vocal energy fade using 10 segments for finer resolution.
    fade_score: last-segment RMS vs max-segment RMS.
    mid_fade_score: average of segments 4-6 vs max (detects mid-passage drops).
    < 0.5 = noticeable fade.
    """
    try:
        n_segments = 10
        seg_len = len(audio_data) // n_segments
        if seg_len < sr * 0.1:  # segment < 0.1s, too short — fallback to 5
            n_segments = 5
            seg_len = len(audio_data) // n_segments
        if seg_len < sr * 0.1:
            return {"fade_score": 1.0, "mid_fade_score": 1.0, "energy_segments": []}

        rms_per_seg = []
        for i in range(n_segments):
            seg = audio_data[i * seg_len: (i + 1) * seg_len]
            rms_per_seg.append(float(np.sqrt(np.mean(seg ** 2))))

        max_rms = max(rms_per_seg) if max(rms_per_seg) > 0 else 1.0
        fade_score = rms_per_seg[-1] / max_rms

        mid_start = n_segments // 4
        mid_end   = n_segments * 3 // 4
        mid_fade_score = float(np.mean(rms_per_seg[mid_start:mid_end])) / max_rms

        return {
            "fade_score":      round(fade_score, 3),
            "mid_fade_score":  round(mid_fade_score, 3),
            "energy_segments": [round(v / max_rms, 3) for v in rms_per_seg],
        }
    except Exception:
        return {"fade_score": 1.0, "mid_fade_score": 1.0, "energy_segments": []}


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
#  Helper: Compute composite emotion score (0-100)
#  Combines 3 acoustic signals:
#    - pitch_std   (F0 variation)   — weight 50%
#    - energy_std  (RMS variation)  — weight 30%
#    - tempo_var   (beat variation) — weight 20%
# ================================================================
def compute_emotion_score(audio_data: np.ndarray, sr: int, pitch_std_semitones: float) -> dict:
    """
    Composite emotion score from 3 acoustic features:
      pitch_std_semitones : already computed upstream
      energy_std_score    : RMS variation over 10-frame windows (0-100)
      tempo_var_score     : beat interval std deviation (0-100)

    Returns dict with composite score + breakdown for diagnostics.
    """
    # ── Component 1: pitch variation (already computed) ──
    pitch_score = pitch_to_expressiveness_score(pitch_std_semitones)

    # ── Component 2: energy (RMS) variation ──
    try:
        frame_len = int(0.025 * sr)
        hop_len   = int(0.010 * sr)
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_len, hop_length=hop_len)[0]
        energy_std_raw = float(np.std(rms))
        # Normalize: raw std ~0.01–0.06 for expressive speech → map to 0-100
        energy_std_score = float(np.clip(energy_std_raw * 1500.0, 0.0, 100.0))
    except Exception:
        energy_std_score = 0.0

    # ── Component 3: tempo/beat variation ──
    try:
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        if len(beats) >= 4:
            beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
            tempo_var_raw = float(np.std(beat_intervals))
            # Normalize: raw std ~0.05–0.3s → map to 0-100
            tempo_var_score = float(np.clip(tempo_var_raw * 300.0, 0.0, 100.0))
        else:
            tempo_var_score = 0.0
    except Exception:
        tempo_var_score = 0.0

    # ── Weighted composite ──
    composite = (
        pitch_score      * 0.50 +
        energy_std_score * 0.30 +
        tempo_var_score  * 0.20
    )
    composite = round(float(np.clip(composite, 0.0, 100.0)), 2)

    return {
        "emotion_score":      composite,
        "pitch_score":        round(pitch_score, 2),
        "energy_std_score":   round(energy_std_score, 2),
        "tempo_var_score":    round(tempo_var_score, 2),
    }


# ================================================================
#  Phase 2 — Spectral & MFCC features
# ================================================================
def compute_spectral_features(audio_data: np.ndarray, sr: int) -> dict:
    """
    Compute 3 spectral features relevant to MC voice quality:
      spectral_centroid_mean_hz : brightness of voice (higher = brighter/clearer)
      spectral_contrast_mean    : sharpness/definition between harmonic peaks and valleys
      mfcc_stability_score      : consistency of articulation (0-100, higher = more consistent)
    """
    try:
        hop = int(0.010 * sr)
        frame = int(0.025 * sr)

        # Spectral centroid — average frequency weighted by energy
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop)[0]
        centroid_mean = float(np.mean(centroid))

        # Spectral contrast — difference between peaks and valleys per band
        contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr, hop_length=hop)
        contrast_mean = float(np.mean(contrast))

        # MFCC stability — low std across time = consistent articulation
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13, hop_length=hop)
        mfcc_std_per_coeff = np.std(mfcc, axis=1)
        mfcc_instability = float(np.mean(mfcc_std_per_coeff))
        # Map: instability ~5-25 typical → invert to 0-100 stability score
        mfcc_stability_score = float(np.clip(100.0 - (mfcc_instability - 5.0) * 5.0, 0.0, 100.0))

        return {
            "spectral_centroid_hz":   round(centroid_mean, 1),
            "spectral_contrast_mean": round(contrast_mean, 2),
            "mfcc_stability_score":   round(mfcc_stability_score, 2),
        }
    except Exception as e:
        print(f"[AI] Spectral features error: {e}")
        return {"spectral_centroid_hz": 0.0, "spectral_contrast_mean": 0.0, "mfcc_stability_score": 0.0}


def compute_pitch_contour(audio_data: np.ndarray, sr: int, f0_array) -> dict:
    """
    Analyse pitch contour shape: rising / falling / flat.
    Uses linear regression slope on voiced F0 frames.
    Positive slope = rising (question/excitement), negative = falling (statement/authority).
    slope_semitones_per_sec:
      > +0.5  → rising
      < -0.5  → falling
      else    → flat
    """
    try:
        if f0_array is None or len(f0_array) < 10:
            return {"pitch_slope": 0.0, "pitch_contour": "flat"}

        hop = 512
        times = librosa.frames_to_time(np.arange(len(f0_array)), sr=sr, hop_length=hop)
        voiced_mask = f0_array > 0
        if np.sum(voiced_mask) < 5:
            return {"pitch_slope": 0.0, "pitch_contour": "flat"}

        t_voiced = times[voiced_mask]
        f0_voiced = f0_array[voiced_mask]
        f0_semi = 12 * np.log2(f0_voiced / 440.0)

        # Linear regression
        t_norm = t_voiced - t_voiced[0]
        slope = float(np.polyfit(t_norm, f0_semi, 1)[0])

        if slope > 0.5:
            contour = "rising"
        elif slope < -0.5:
            contour = "falling"
        else:
            contour = "flat"

        return {"pitch_slope": round(slope, 3), "pitch_contour": contour}
    except Exception:
        return {"pitch_slope": 0.0, "pitch_contour": "flat"}


def compute_jitter_shimmer_hnr(audio_data: np.ndarray, sr: int, f0_array, voiced_flag) -> dict:
    """
    Compute voice quality indicators from raw F0 and audio:
      jitter_pct    : cycle-to-cycle F0 variation (%) — high = shaky/nervous voice
                      Normal < 1.0%, Pathological > 3%
      shimmer_pct   : cycle-to-cycle amplitude variation (%) — high = unsteady loudness
                      Normal < 3.0%, Pathological > 6%
      hnr_db        : Harmonic-to-Noise Ratio — quality of vocal fold vibration
                      MC target > 15 dB. < 10 dB = hoarse/breathy.
    All computed from voiced frames only.
    """
    try:
        if f0_array is None or voiced_flag is None:
            return {"jitter_pct": 0.0, "shimmer_pct": 0.0, "hnr_db": 0.0}

        voiced_f0 = f0_array[voiced_flag & (f0_array > 0)]
        if len(voiced_f0) < 10:
            return {"jitter_pct": 0.0, "shimmer_pct": 0.0, "hnr_db": 0.0}

        # Jitter: mean absolute difference of successive F0 periods / mean period
        periods = 1.0 / voiced_f0
        period_diffs = np.abs(np.diff(periods))
        jitter_pct = float(np.mean(period_diffs) / np.mean(periods) * 100.0)
        jitter_pct = round(float(np.clip(jitter_pct, 0.0, 20.0)), 3)

        # Shimmer: mean absolute amplitude difference between successive voiced frames
        hop = 512
        frame_len = 2048
        rms = librosa.feature.rms(y=audio_data, frame_length=frame_len, hop_length=hop)[0]
        # Align rms length with f0_array (pyin uses same hop/frame)
        min_len = min(len(rms), len(voiced_flag))
        rms_voiced = rms[:min_len][voiced_flag[:min_len] & (f0_array[:min_len] > 0)]
        if len(rms_voiced) > 2 and np.mean(rms_voiced) > 0:
            amp_diffs = np.abs(np.diff(rms_voiced))
            shimmer_pct = float(np.mean(amp_diffs) / np.mean(rms_voiced) * 100.0)
            shimmer_pct = round(float(np.clip(shimmer_pct, 0.0, 30.0)), 3)
        else:
            shimmer_pct = 0.0

        # HNR approximation via autocorrelation
        # Use a voiced segment: find first voiced window of ~0.1s
        hop_t = hop / sr
        voiced_indices = np.where(voiced_flag[:min_len] & (f0_array[:min_len] > 0))[0]
        if len(voiced_indices) > 0:
            center_idx = voiced_indices[len(voiced_indices) // 2]
            start_sample = int(center_idx * hop)
            window_samples = int(0.1 * sr)
            end_sample = min(start_sample + window_samples, len(audio_data))
            segment = audio_data[start_sample:end_sample]
            if len(segment) > 100:
                autocorr = np.correlate(segment, segment, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]
                r0 = autocorr[0]
                f0_center = float(np.mean(voiced_f0[len(voiced_f0)//2 - 2 : len(voiced_f0)//2 + 2]))
                lag = int(sr / max(f0_center, 80))
                if 0 < lag < len(autocorr) and r0 > 0:
                    r1 = autocorr[lag]
                    ratio = np.clip(r1 / r0, 0.0001, 0.9999)
                    hnr_db = round(float(10 * np.log10(ratio / (1 - ratio))), 1)
                    hnr_db = float(np.clip(hnr_db, 0.0, 40.0))
                else:
                    hnr_db = 0.0
            else:
                hnr_db = 0.0
        else:
            hnr_db = 0.0

        return {"jitter_pct": jitter_pct, "shimmer_pct": shimmer_pct, "hnr_db": hnr_db}
    except Exception as e:
        print(f"[AI] Jitter/Shimmer/HNR error: {e}")
        return {"jitter_pct": 0.0, "shimmer_pct": 0.0, "hnr_db": 0.0}


def detect_filler_words(text: str) -> dict:
    """
    Count filler words common in Vietnamese MC speech.
    Returns count, list of found fillers, and score (100 = no fillers).
    """
    import re
    fillers_vi = [
        "ừm", "ừ", "à", "ờ", "ơ", "thì", "là", "mà", "nhé", "nha",
        "cái", "đó là", "tức là", "như là", "thật ra", "thật sự",
        "okay", "ok", "uh", "um", "er", "ah",
    ]
    text_lower = text.lower()
    found = []
    total = 0
    for filler in fillers_vi:
        pattern = r'\b' + re.escape(filler) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            found.append({"word": filler, "count": len(matches)})
            total += len(matches)

    word_count = max(1, len(text_lower.split()))
    filler_ratio = total / word_count
    filler_score = float(np.clip(100.0 - filler_ratio * 500.0, 0.0, 100.0))

    return {
        "filler_count":  total,
        "filler_ratio":  round(filler_ratio, 4),
        "filler_score":  round(filler_score, 2),
        "fillers_found": found,
    }


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
    expressiveness_score: float,  # composite emotion score
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
    emotion_data: dict = None,
    spectral_features: dict = None,
    pitch_contour: dict = None,
    filler_data: dict = None,
    voice_quality: dict = None,
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

    # Pitch + emotion tips
    if pitch_std < PITCH_EXPRESSIVE_THRESHOLD:
        tips_vi.append({"label": "GIỌNG ĐIỆU", "tip": f"Biến thiên cao độ thấp ({pitch_std:.1f} semitone). Luyện đọc với cường độ cảm xúc tăng dần."})
    if emotion_data:
        if emotion_data.get("energy_std_score", 100) < 35:
            tips_vi.append({"label": "NĂNG LƯỢNG", "tip": "Mức năng lượng giọng đều đều — thử tăng/giảm to nhỏ theo từng đoạn câu để tạo điểm nhấn."})
        if emotion_data.get("tempo_var_score", 100) < 35:
            tips_vi.append({"label": "NHỊP ĐỘ", "tip": "Nhịp nói quá đều — thử chậm lại ở câu quan trọng, nhanh hơn ở đoạn chuyển tiếp."})

    # Phase 2 tips — spectral, pitch contour, filler words
    if spectral_features:
        if spectral_features.get("spectral_centroid_hz", 2000) < 1500:
            tips_vi.append({"label": "ÂM SẮC", "tip": "Giọng hơi trầm tối — thử nhấc khẩu hình lên nhẹ và mỉm cười khi nói để làm sáng giọng."})
        if spectral_features.get("mfcc_stability_score", 100) < 50:
            tips_vi.append({"label": "NHẤT QUÁN", "tip": "Phát âm thiếu nhất quán giữa các câu — luyện đọc cùng một đoạn nhiều lần để ổn định khẩu hình."})

    if pitch_contour and pitch_contour.get("pitch_contour") == "flat":
        tips_vi.append({"label": "ĐƯỜNG CAO ĐỘ", "tip": "Câu nói thiếu hướng cao độ — thử kết thúc câu khẳng định bằng nốt xuống, câu hỏi bằng nốt lên."})

    if filler_data and filler_data.get("filler_count", 0) >= 3:
        filler_list = ", ".join(f['word'] for f in filler_data.get("fillers_found", [])[:3])
        tips_vi.append({"label": "TỪ ĐỆM", "tip": f"Phát hiện {filler_data['filler_count']} từ đệm ({filler_list}...) — thay bằng khoảng lặng có chủ đích."})

    if voice_quality:
        if voice_quality.get("jitter_pct", 0) > 2.0:
            tips_vi.append({"label": "ĐỘ RUN GIỌNG", "tip": f"Jitter cao ({voice_quality['jitter_pct']:.1f}%) — giọng run hoặc căng thẳng. Hít thở sâu, thư giãn cổ họng trước khi nói."})
        if voice_quality.get("shimmer_pct", 0) > 5.0:
            tips_vi.append({"label": "ỔN ĐỊNH ÂM LƯỢNG", "tip": f"Shimmer cao ({voice_quality['shimmer_pct']:.1f}%) — âm lượng không đều giữa các âm. Luyện hát thang âm giữ đều volume."})
        if voice_quality.get("hnr_db", 20) < 10.0 and voice_quality.get("hnr_db", 0) > 0:
            tips_vi.append({"label": "CHẤT GIỌNG", "tip": f"HNR thấp ({voice_quality['hnr_db']:.0f} dB) — giọng khàn/thở. Kiểm tra tình trạng thanh đới và uống nước trước khi tập."})

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
    if emotion_data:
        if emotion_data.get("energy_std_score", 100) < 35:
            tips_en.append({"label": "ENERGY", "tip": "Voice energy is flat — practice dynamic volume shifts to emphasize key phrases."})
        if emotion_data.get("tempo_var_score", 100) < 35:
            tips_en.append({"label": "TEMPO", "tip": "Speaking tempo is too steady — slow down on important sentences, speed up on transitions."})

    if spectral_features:
        if spectral_features.get("spectral_centroid_hz", 2000) < 1500:
            tips_en.append({"label": "TIMBRE", "tip": "Voice sounds dark/heavy — try raising the soft palate and slight smile position to brighten tone."})
        if spectral_features.get("mfcc_stability_score", 100) < 50:
            tips_en.append({"label": "CONSISTENCY", "tip": "Articulation is inconsistent across sentences — drill the same passage repeatedly to stabilize mouth position."})

    if pitch_contour and pitch_contour.get("pitch_contour") == "flat":
        tips_en.append({"label": "PITCH DIRECTION", "tip": "Sentences lack pitch direction — end declarative sentences on a falling note, questions on a rising note."})

    if filler_data and filler_data.get("filler_count", 0) >= 3:
        filler_list = ", ".join(f['word'] for f in filler_data.get("fillers_found", [])[:3])
        tips_en.append({"label": "FILLER WORDS", "tip": f"Detected {filler_data['filler_count']} filler words ({filler_list}...) — replace with deliberate pauses."})

    if voice_quality:
        if voice_quality.get("jitter_pct", 0) > 2.0:
            tips_en.append({"label": "VOICE TREMOR", "tip": f"High jitter ({voice_quality['jitter_pct']:.1f}%) — voice is shaky or tense. Take deep breaths and relax throat before speaking."})
        if voice_quality.get("shimmer_pct", 0) > 5.0:
            tips_en.append({"label": "VOLUME STABILITY", "tip": f"High shimmer ({voice_quality['shimmer_pct']:.1f}%) — uneven amplitude between sounds. Practice sustained vowel exercises at steady volume."})
        if voice_quality.get("hnr_db", 20) < 10.0 and voice_quality.get("hnr_db", 0) > 0:
            tips_en.append({"label": "VOICE QUALITY", "tip": f"Low HNR ({voice_quality['hnr_db']:.0f} dB) — hoarse or breathy quality detected. Check vocal fold health and hydrate before practice."})

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

    newline = "\n"
    default_rows_vi = (
        f"| **Phát âm** | {accuracy_status_vi} | {accuracy_score:.1f}/100 | — |" + newline +
        f"| **Giọng điệu** | {intonation_status_vi} | {expressiveness_score:.1f}/100 | — |" + newline +
        f"| **Tốc độ** | {pace_status_vi} | {wpm:.0f} WPM | — |" + newline
    )
    default_rows_en = (
        f"| **Pronunciation** | {accuracy_status_en} | {accuracy_score:.1f}/100 | — |" + newline +
        f"| **Intonation** | {intonation_status_en} | {expressiveness_score:.1f}/100 | — |" + newline +
        f"| **Pacing** | {pace_status_en} | {wpm:.0f} WPM | — |" + newline
    )
    actions_vi_str = "".join(a + newline for a in actions_vi)
    actions_en_str = "".join(a + newline for a in actions_en)
    pause_status_vi = "Hợp lý" if PAUSE_MIN_GOOD_SEC <= avg_pause <= PAUSE_MAX_GOOD_SEC else "Chưa ổn"
    pause_status_en = "Optimal" if PAUSE_MIN_GOOD_SEC <= avg_pause <= PAUSE_MAX_GOOD_SEC else "Suboptimal"
    rows_vi = criteria_rows_vi if criteria_rows_vi else default_rows_vi
    rows_en = criteria_rows_en if criteria_rows_en else default_rows_en

    report_vi = f"""### 🎙️ Báo cáo Phân tích Chuyên sâu (AI Expert){hint_line_vi}
**Đánh giá tổng thể:** Điểm tổng hợp **{overall_score:.1f}/100** · Độ chính xác phát âm **{accuracy_score:.1f}%**

#### 📊 Điểm theo tiêu chí:
| Tiêu chí | Trạng thái | Điểm | Trọng số |
| :--- | :--- | :--- | :--- |
{rows_vi}
#### 📈 Phân tích kỹ thuật:
| Tiêu chí | Trạng thái | Chỉ số thực tế | Mục tiêu |
| :--- | :--- | :--- | :--- |
| **Phát âm** | {accuracy_status_vi} | {accuracy_score:.1f}% | > 90% |
| **Tốc độ** | {pace_status_vi} | {wpm:.0f} WPM | {target_wpm_min}–{target_wpm_max} WPM |
| **Giọng điệu (F0)** | {intonation_status_vi} | {pitch_std:.2f} semitone | ≥ 2.0 semitone |
| **Ngắt nghỉ** | {pause_status_vi} | {avg_pause}s avg | {PAUSE_MIN_GOOD_SEC}s–{PAUSE_MAX_GOOD_SEC}s |
| **Năng lượng cuối** | {fade_status_vi} | {fade_score:.0%} | ≥ {ENERGY_FADE_WARN_THRESHOLD:.0%} |
| **Chất lượng âm** | {noise_status_vi} | SNR {snr_db:.0f} dB | ≥ {SNR_NOISE_THRESHOLD_DB + 5:.0f} dB |

#### 💡 Hành động cải thiện:
{actions_vi_str}""".strip()

    report_en = f"""### 🎙️ Advanced AI Performance Report{hint_line_en}
**Overall Score:** **{overall_score:.1f}/100** · Pronunciation Accuracy **{accuracy_score:.1f}%**

#### 📊 Per-Criterion Scores:
| Criterion | Status | Score | Weight |
| :--- | :--- | :--- | :--- |
{rows_en}
#### 📈 Technical Analysis:
| Metric | Status | Actual Value | MC Standard |
| :--- | :--- | :--- | :--- |
| **Articulation** | {accuracy_status_en} | {accuracy_score:.1f}% | > 90% |
| **Pacing** | {pace_status_en} | {wpm:.0f} WPM | {target_wpm_min}–{target_wpm_max} WPM |
| **Intonation (F0)** | {intonation_status_en} | {pitch_std:.2f} semitones | ≥ 2.0 semitones |
| **Pausing** | {pause_status_en} | {avg_pause}s avg | {PAUSE_MIN_GOOD_SEC}s–{PAUSE_MAX_GOOD_SEC}s |
| **Energy Sustain** | {fade_status_en} | {fade_score:.0%} | ≥ {ENERGY_FADE_WARN_THRESHOLD:.0%} |
| **Audio Quality** | {noise_status_en} | SNR {snr_db:.0f} dB | ≥ {SNR_NOISE_THRESHOLD_DB + 5:.0f} dB |

#### 💡 Improvement Plan:
{actions_en_str}""".strip()

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

    print(f"[AI] Received: {file.filename!r} -> {temp_filename} ({os.path.getsize(temp_filename)} bytes)")
    print(f"[AI] WPM target: {target_wpm_min}-{target_wpm_max} | criteria: {len(criteria)} items")

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
        cer_rate       = jiwer.cer(script_origin.lower(), text_spoken.lower())
        # Weighted accuracy: WER captures omitted/added words; CER captures mispronunciation
        accuracy_score = max(0.0, 100.0 - (error_rate * 70.0 + cer_rate * 30.0))
        wpm            = (word_count / duration_speech) * 60.0  # speech-only duration

        # Onset strength variation (beat/stress energy — kept as RHYTHM dimension)
        onset_env           = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onset_variation_raw = float(np.std(onset_env))
        onset_variation_score = min(100.0, onset_variation_raw * 20.0)

        # ── Stage 5: Pitch (F0) + composite emotion analysis ──
        pitch_stats      = compute_pitch_stats(audio_data, sr)
        emotion_data     = compute_emotion_score(audio_data, sr, pitch_stats["pitch_std_semitones"])
        expressiveness_score = emotion_data["emotion_score"]

        # ── Stage 6: SNR — noise quality ──
        snr_db = compute_snr(audio_data, sr)

        # ── Stage 7: Energy profile — fade detection ──
        energy_profile = compute_energy_profile(audio_data, sr)

        # ── Stage 7b: Phase 2+3 — spectral, pitch contour, filler, jitter/shimmer/HNR ──
        spectral_features = compute_spectral_features(audio_data, sr)
        f0_raw, voiced_flag_raw, _ = librosa.pyin(
            audio_data,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr, frame_length=2048, hop_length=512,
        )
        pitch_contour   = compute_pitch_contour(audio_data, sr, f0_raw)
        filler_data     = detect_filler_words(text_spoken)
        voice_quality   = compute_jitter_shimmer_hnr(audio_data, sr, f0_raw, voiced_flag_raw)

        print(f"[AI] accuracy={accuracy_score:.1f}% (wer={error_rate:.2f} cer={cer_rate:.2f}) | wpm={wpm:.0f} | "
              f"pitch_std={pitch_stats['pitch_std_semitones']:.2f}st contour={pitch_contour['pitch_contour']} | "
              f"emotion={expressiveness_score:.1f} (pitch={emotion_data['pitch_score']:.1f} "
              f"energy={emotion_data['energy_std_score']:.1f} tempo={emotion_data['tempo_var_score']:.1f}) | "
              f"spectral_centroid={spectral_features['spectral_centroid_hz']:.0f}Hz "
              f"mfcc_stability={spectral_features['mfcc_stability_score']:.1f} | "
              f"jitter={voice_quality['jitter_pct']:.2f}% shimmer={voice_quality['shimmer_pct']:.2f}% "
              f"hnr={voice_quality['hnr_db']:.1f}dB | "
              f"onset_var={onset_variation_score:.1f} | snr={snr_db:.1f}dB | "
              f"fade={energy_profile['fade_score']:.2f} mid={energy_profile['mid_fade_score']:.2f} | "
              f"fillers={filler_data['filler_count']}")

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
            emotion_data=emotion_data,
            spectral_features=spectral_features,
            pitch_contour=pitch_contour,
            filler_data=filler_data,
            voice_quality=voice_quality,
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
            "emotion_breakdown": emotion_data,
            "cer_rate":          float(round(cer_rate, 4)),
            "wer_rate":          float(round(error_rate, 4)),
            "spectral_features": spectral_features,
            "pitch_contour":     pitch_contour,
            "filler_words":      filler_data,
            "voice_quality":     voice_quality,
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
#  Health Check
# ================================================================
@app.get("/")
def read_root():
    return {
        "message": "MC Hub AI Service is running",
        "device":        device,
        "gpu":           torch.cuda.get_device_name(0) if device == "cuda" else "N/A",
        "whisper_model": whisper_model_name,
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
        reload=False,
        loop="asyncio",
        log_level="info",
    )

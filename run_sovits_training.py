"""
GPT-SoVITS TTS Fine-tuning Pipeline
Phase 5: Training MC voice (Vietnamese)
Dataset: training_data/metadata_master.txt + library/vietsuperspeech/
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
SOVITS_DIR     = BASE_DIR / "GPT-SoVITS"
VENV_PYTHON    = SOVITS_DIR / "venv_gpt" / "Scripts" / "python.exe"
TTS_META       = BASE_DIR / "training_data" / "metadata_master.txt"
PRETRAINED_DIR = SOVITS_DIR / "GPT_SoVITS" / "pretrained_models"
OUTPUT_DIR     = BASE_DIR / "models" / "gpt-sovits-vi"
EXPERIMENT     = "mc_vi_voice"

# GPT-SoVITS pretrained model paths
S1_CKPT = PRETRAINED_DIR / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
S2G_PTH = PRETRAINED_DIR / "gsv-v2final-pretrained" / "s2G2333k.pth"
S2D_PTH = PRETRAINED_DIR / "gsv-v2final-pretrained" / "s2D2333k.pth"

print("=" * 60)
print("  GPT-SoVITS TTS Fine-tuning — Vietnamese MC Voice")
print("=" * 60)

# ─── Step 1: Validate ─────────────────────────────────────────────────────────
print("\n[1/5] Validating environment...")

if not VENV_PYTHON.exists():
    print(f"ERROR: venv not found at {VENV_PYTHON}")
    sys.exit(1)

if not TTS_META.exists():
    print(f"ERROR: TTS metadata not found at {TTS_META}")
    sys.exit(1)

# Check source WAVs
vss_dir = BASE_DIR / "library" / "vietsuperspeech"
wav_count = len(list(vss_dir.glob("*.wav")))
print(f"  [OK] Python : {VENV_PYTHON}")
print(f"  [OK] Dataset: {wav_count} WAV files in {vss_dir}")
print(f"  [OK] S1 ckpt: {S1_CKPT.name}")

# ─── Step 2: Prepare experiment folder ────────────────────────────────────────
print("\n[2/5] Setting up experiment directory...")

exp_dir = SOVITS_DIR / "logs" / EXPERIMENT
(exp_dir / "0_gt_wavs").mkdir(parents=True, exist_ok=True)
(exp_dir / "5_wav16k").mkdir(parents=True, exist_ok=True)

# Build annotation and copy WAVs
print("\n[3/5] Building annotation file & copying WAVs...")
ann_path = exp_dir / "ann.list"
lines_written = 0
max_samples = 1000

with open(TTS_META, "r", encoding="utf-8") as f_meta, \
     open(ann_path, "w", encoding="utf-8") as f_ann:
    for line in f_meta:
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        # Format: path|speaker|lang|text
        if len(parts) < 4:
            continue
        
        rel_wav_path = parts[0].strip()
        speaker      = parts[1].strip()
        text         = parts[3].strip()
        
        src_wav = BASE_DIR / rel_wav_path
        if not src_wav.exists():
            continue
            
        dst_wav = exp_dir / "0_gt_wavs" / src_wav.name
        if not dst_wav.exists():
            shutil.copy2(src_wav, dst_wav)
        
        # Format: wav_path|speaker|vi|text
        f_ann.write(f"{dst_wav}|{speaker}|vi|{text}\n")
        lines_written += 1
        
        if lines_written >= max_samples:
            break

print(f"  [OK] Annotation file: {ann_path} ({lines_written} lines)")

# ─── Step 3: Step 1 — Extract text features ────────────────────────────────────
print("\n[4/5] Extracting features (Step 1: text + hubert)...")

env = os.environ.copy()
env["PYTHONPATH"] = str(SOVITS_DIR)
env["PYTHONIOENCODING"] = "utf-8"

# Run prepare step 1 (get-text)
step1_script = SOVITS_DIR / "GPT_SoVITS" / "prepare_datasets" / "1-get-text.py"
if step1_script.exists():
    env_step1 = env.copy()
    env_step1["inp_text"] = str(ann_path)
    env_step1["inp_wav_dir"] = str(exp_dir / "0_gt_wavs")
    env_step1["exp_name"] = EXPERIMENT
    env_step1["i_part"] = "0"
    env_step1["all_parts"] = "1"
    env_step1["opt_dir"] = str(exp_dir)
    env_step1["bert_pretrained_dir"] = str(PRETRAINED_DIR / "chinese-roberta-wwm-ext-large")
    env_step1["is_half"] = "True"
    env_step1["PYTHONPATH"] = str(SOVITS_DIR / "GPT_SoVITS") + os.pathsep + env.get("PYTHONPATH", "")

    print(f"  Running Step 1 (text extraction)...")
    # Use direct stream to console to avoid decoding issues in capture
    result = subprocess.run([str(VENV_PYTHON), str(step1_script)], cwd=str(SOVITS_DIR), env=env_step1, timeout=600)
    if result.returncode != 0:
        print(f"  [WARN] Step 1 failed with exit code {result.returncode}")
    else:
        print("  [OK] Step 1 complete")

# ─── Step 4: Step 2 — Extract Hubert features ──────────────────────────────────
print("\n[5/5] Extracting features (Step 2: hubert)...")
step2_script = SOVITS_DIR / "GPT_SoVITS" / "prepare_datasets" / "2-get-hubert-wav32k.py"
if step2_script.exists():
    env_step2 = env_step1.copy()
    env_step2["cnhubert_base_dir"] = str(PRETRAINED_DIR / "chinese-hubert-base")
    
    print(f"  Running Step 2 (hubert extraction)...")
    result = subprocess.run([str(VENV_PYTHON), str(step2_script)], cwd=str(SOVITS_DIR), env=env_step2, timeout=600)
    if result.returncode != 0:
        print(f"  [WARN] Step 2 failed with exit code {result.returncode}")
    else:
        print("  [OK] Step 2 complete")

# ─── Step 5: Step 3 — Extract Semantic features ─────────────────────────────────
print("\n[6/6] Extracting features (Step 3: semantic)...")
step3_script = SOVITS_DIR / "GPT_SoVITS" / "prepare_datasets" / "3-get-semantic.py"
if step3_script.exists():
    S2_CONFIG = SOVITS_DIR / "GPT_SoVITS" / "configs" / "s2.json"
    env_step3 = env_step1.copy()
    env_step3["pretrained_s2G"] = str(S2G_PTH)
    env_step3["s2config_path"] = str(S2_CONFIG)
    env_step3["version"] = "v2"
    
    print(f"  Running Step 3 (semantic extraction)...")
    result = subprocess.run([str(VENV_PYTHON), str(step3_script)], cwd=str(SOVITS_DIR / "GPT_SoVITS"), env=env_step3, timeout=900)
    if result.returncode != 0:
        print(f"  [WARN] Step 3 failed with exit code {result.returncode}")
    else:
        print("  [OK] Step 3 complete")

# ─── Step 6: Launch WebUI ─────────────────────────────────────────────────────
print("\n[FINISH] Preprocessing done. Experiment ready at:", exp_dir)
print("  [->] S1 model:", S1_CKPT.name)
print("  [->] S2 model:", S2G_PTH.name)
print("\n  Now launch WebUI manually to start training:")
print("  python webui.py --device cuda --is_half True")

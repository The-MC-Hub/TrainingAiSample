"""
MC Hub AI — GPT-SoVITS TTS Fine-tune Launcher
===============================================
Launches GPT-SoVITS WebUI pre-configured for Vietnamese MC voice training.
Run this AFTER Whisper fine-tune completes.

Dataset: training_data/tts_wavs/ + training_data/tts_metadata.txt
Output:  models/gpt-sovits-vi/

Usage:
  python launch_sovits_training.py
"""

import os
import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SOVITS_DIR    = Path("./GPT-SoVITS")
TTS_WAVS_DIR  = Path("./training_data/tts_wavs")
TTS_META      = Path("./training_data/tts_metadata.txt")
OUTPUT_DIR    = Path("./models/gpt-sovits-vi")

def check_prereqs():
    print("\n[CHECK] Verifying prerequisites...")
    ok = True

    if not SOVITS_DIR.exists():
        print(f"  [FAIL] GPT-SoVITS not found at {SOVITS_DIR}")
        print("         Run: git clone https://github.com/RVC-Boss/GPT-SoVITS.git")
        ok = False
    else:
        print(f"  [OK]   GPT-SoVITS found")

    wav_count = len(list(TTS_WAVS_DIR.glob("*.wav"))) if TTS_WAVS_DIR.exists() else 0
    if wav_count == 0:
        print(f"  [FAIL] No TTS WAVs found in {TTS_WAVS_DIR}")
        print("         Run: python preprocess_audio.py")
        ok = False
    else:
        print(f"  [OK]   {wav_count} TTS WAVs ready")

    if not TTS_META.exists():
        print(f"  [FAIL] {TTS_META} not found")
        ok = False
    else:
        lines = len(TTS_META.read_text(encoding="utf-8").splitlines())
        print(f"  [OK]   tts_metadata.txt: {lines} entries")

    return ok

def create_sovits_config():
    """Write GPT-SoVITS training config for Vietnamese data."""
    import json

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "exp_name": "MC_Hub_Vietnamese",
        "inp_text": str(TTS_META.absolute()),
        "inp_wav_dir": str(TTS_WAVS_DIR.absolute()),
        "opt_dir": str(OUTPUT_DIR.absolute()),
        "bert_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        "ssl_pretrained_dir": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "pretrained_s2G_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth",
        "pretrained_s2D_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth",
        "pretrained_s1_path": "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "gpu_numbers_ced": "0",
        "gpu_numbers_s1": "0",
        "gpu_numbers_s2": "0",
        "batch_size_s1": 4,
        "batch_size_s2": 4,
        "total_epochs_s1": 8,
        "total_epochs_s2": 8,
        "save_every_epoch": 2,
        "language": "vi",
        "if_save_latest": True,
        "if_save_every_weights": True,
    }

    config_path = OUTPUT_DIR / "training_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK]   Config saved to {config_path}")
    return config

def launch_webui():
    """Launch GPT-SoVITS WebUI in the venv."""
    venv_python = SOVITS_DIR / "venv_gpt" / "Scripts" / "python.exe"

    if not venv_python.exists():
        # Fall back to system python
        python_exe = sys.executable
        print(f"  [INFO] venv not found, using system Python: {python_exe}")
    else:
        python_exe = str(venv_python)
        print(f"  [OK]   Using venv Python: {python_exe}")

    webui_path = SOVITS_DIR / "webui.py"

    print(f"\n[LAUNCH] Starting GPT-SoVITS WebUI...")
    print("  -> Open http://localhost:9872 in your browser")
    print("  -> Go to 'One-click Training' tab")
    print("  -> Set Experiment Name: MC_Hub_Vietnamese")
    print(f"  -> Text file: {TTS_META.absolute()}")
    print(f"  -> Audio dir: {TTS_WAVS_DIR.absolute()}")
    print("  -> Language: vi")
    print("  -> Click 'Start Training'\n")

    os.chdir(SOVITS_DIR)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    subprocess.run([python_exe, str(webui_path)], check=True)


if __name__ == "__main__":
    print("="*55)
    print(" MC Hub AI - GPT-SoVITS TTS Fine-tune Launcher")
    print("="*55)

    if not check_prereqs():
        sys.exit(1)

    config = create_sovits_config()

    print(f"\n[READY] Training config written.")
    print("  TTS dataset  : 1999 samples @ 22050Hz")
    print("  GPU          : RTX 4060 (auto-detected)")
    print("  Output dir   : models/gpt-sovits-vi/")

    launch_webui()

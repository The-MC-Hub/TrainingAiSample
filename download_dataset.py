"""
MC Hub AI — Dataset Downloader (v5 — VERIFIED WORKING)
========================================================
Verified datasets (no auth, accessible):
  1. thanhnew2001/VietSuperSpeech — 267h, casual speech, public
  2. VIVOS raw — download via Kaggle API or manual zip

Usage:
  python download_dataset.py
"""

import os
import sys
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

LIBRARY_DIR = Path("./library")
TRAINING_DATA_DIR = Path("./training_data")


def save_audio_array(audio_array, sampling_rate: int, output_path: Path):
    """Save numpy audio array to WAV file."""
    sf.write(str(output_path), audio_array, sampling_rate)


# ================================================================
#  DATASET 1: VietSuperSpeech — 267h, VERIFIED PUBLIC, no auth
#  Keys: audio, text, duration, source
#  Great for: STT + TTS (diverse casual/vlog speech)
# ================================================================
def download_vietsuperspeech(max_samples: int = None):
    print("\n" + "="*60)
    print(f"DOWNLOADING: VietSuperSpeech (267h, casual speech)")
    if max_samples:
        print(f"  Fetching {max_samples} samples")
    print("="*60)

    try:
        from datasets import load_dataset, Audio
        import datasets as hf_datasets
        hf_datasets.config.AUDIO_DECODER = "av"
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] pip install datasets av huggingface_hub")
        return []

    out_dir = LIBRARY_DIR / "vietsuperspeech"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_lines = []
    errors = 0

    REPO_ID = "thanhnew2001/VietSuperSpeech"

    try:
        print("[VSS] Loading metadata (streaming)...")
        # Load without audio decoding first to get text + path
        ds = load_dataset(REPO_ID, split="train", streaming=True)

        pbar = tqdm(desc="VietSuperSpeech", unit="samples",
                    total=max_samples if max_samples else None)

        for i, sample in enumerate(ds):
            if max_samples and i >= max_samples:
                break

            text = sample.get("text", "").strip()
            audio_path = sample.get("audio", "")  # relative path in repo

            if not text or not audio_path:
                pbar.update(1)
                continue

            wav_filename = f"vss_{i:06d}.wav"
            wav_path = out_dir / wav_filename

            try:
                # Download the actual WAV file from HuggingFace repo
                local_file = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=audio_path,
                    repo_type="dataset",
                    local_dir=str(out_dir / "_cache")
                )
                # Load with librosa and resample to 16kHz
                import librosa
                audio_array, sr = librosa.load(local_file, sr=16000)
                sf.write(str(wav_path), audio_array, sr)
                metadata_lines.append(f"library/vietsuperspeech/{wav_filename}|VSS_SPK|vi|{text}")
            except Exception as e:
                errors += 1

            pbar.update(1)

        pbar.close()

    except Exception as e:
        print(f"[VSS ERROR] {e}")
        return []

    # Clean up cache
    import shutil
    cache_dir = out_dir / "_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    meta_path = out_dir / "metadata.txt"
    meta_path.write_text("\n".join(metadata_lines), encoding="utf-8")
    print(f"[VSS] Done! {len(metadata_lines)} samples saved. ({errors} errors skipped)")
    return metadata_lines


# ================================================================
#  DATASET 2: VIVOS — via direct zip download
#  Fallback options if official URL is down:
#    - Kaggle: ltthanh/vivos
#    - OpenSLR: openslr.org/101/
# ================================================================
def download_vivos_openslr():
    """
    VIVOS is also available on OpenSLR (mirror of AILAB).
    URL: https://www.openslr.org/resources/101/vivos.zip
    """
    import urllib.request
    import zipfile

    print("\n" + "="*60)
    print("DOWNLOADING: VIVOS (via OpenSLR mirror, ~1.5GB)")
    print("="*60)

    out_dir = LIBRARY_DIR / "vivos"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = LIBRARY_DIR / "vivos.zip"

    OPENSLR_URL = "https://www.openslr.org/resources/101/vivos.zip"

    if not zip_path.exists():
        print(f"[VIVOS] Downloading from OpenSLR...")
        print(f"[VIVOS] URL: {OPENSLR_URL}")
        try:
            def progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 / total_size)
                    mb = downloaded / 1024 / 1024
                    total_mb = total_size / 1024 / 1024
                    print(f"\r  {pct:.1f}% ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

            urllib.request.urlretrieve(OPENSLR_URL, zip_path, progress)
            print("\n[VIVOS] Downloaded!")
        except Exception as e:
            print(f"\n[VIVOS] OpenSLR download failed: {e}")
            print("\n[VIVOS] MANUAL DOWNLOAD REQUIRED:")
            print("  1. Go to: https://www.openslr.org/101/")
            print("  2. Download vivos.zip")
            print(f"  3. Place it at: {zip_path.absolute()}")
            print("  4. Re-run this script")
            return []
    else:
        print(f"[VIVOS] Found cached zip. Extracting...")

    # Extract
    print("[VIVOS] Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    print("[VIVOS] Extracted!")

    # Build metadata from prompts.txt
    metadata_lines = []
    for split in ["train", "test"]:
        prompts_file = out_dir / "vivos" / split / "prompts.txt"
        waves_dir = out_dir / "vivos" / split / "waves"

        if not prompts_file.exists():
            print(f"[VIVOS] {split}/prompts.txt not found, skipping")
            continue

        prompts = {}
        for line in prompts_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                prompts[parts[0]] = parts[1]

        wav_files = list(waves_dir.rglob("*.wav")) if waves_dir.exists() else []
        print(f"[VIVOS] {split}: {len(wav_files)} WAVs, {len(prompts)} prompts")

        for wav_file in wav_files:
            speaker = wav_file.parent.name
            file_id = f"{speaker}_{wav_file.stem}"
            text = prompts.get(file_id, prompts.get(wav_file.stem, "")).strip()

            if not text:
                continue

            rel = str(wav_file).replace("\\", "/").replace(str(LIBRARY_DIR).replace("\\","/") + "/", "library/")
            metadata_lines.append(f"{rel}|{speaker}|vi|{text}")

    if metadata_lines:
        meta_path = out_dir / "metadata.txt"
        meta_path.write_text("\n".join(metadata_lines), encoding="utf-8")
        print(f"[VIVOS] Done! {len(metadata_lines)} samples.")

    return metadata_lines


# ================================================================
#  Merge all metadata
# ================================================================
def merge_metadata():
    print("\n" + "="*60)
    print("MERGING all metadata...")
    print("="*60)

    all_lines = []
    for f in LIBRARY_DIR.rglob("metadata.txt"):
        try:
            lines = [l for l in f.read_text(encoding="utf-8").splitlines()
                     if l.strip() and "|" in l]
            all_lines.extend(lines)
            print(f"  [+] {f.parent.name}: {len(lines)} samples")
        except Exception as e:
            print(f"  [!] {f}: {e}")

    master = TRAINING_DATA_DIR / "metadata_master.txt"
    master.parent.mkdir(exist_ok=True)
    master.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"\n[MERGE] Total: {len(all_lines)} -> {master}")
    return all_lines


# ================================================================
#  Main
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" MC Hub AI - Dataset Downloader v5")
    print("=" * 60)

    print("\nOptions:")
    print("  1. VietSuperSpeech 2000 samples  (fast, STT+TTS, ~20 min)")
    print("  2. VietSuperSpeech FULL 267h     (takes hours, maximum STT)")
    print("  3. VIVOS via OpenSLR             (~1.5GB, best TTS quality)")
    print("  4. Both VIVOS + VSS 2000         (RECOMMENDED for STT+TTS)")
    print("  5. Quick test 200 samples        (~3 min, verify pipeline)")

    choice = input("\nEnter choice [1/2/3/4/5]: ").strip()

    if choice == "1":
        download_vietsuperspeech(max_samples=2000)
    elif choice == "2":
        download_vietsuperspeech()  # all ~267h
    elif choice == "3":
        download_vivos_openslr()
    elif choice == "4":
        download_vivos_openslr()
        download_vietsuperspeech(max_samples=2000)
    elif choice == "5":
        print("\n[TEST MODE] 200 samples from VietSuperSpeech...")
        download_vietsuperspeech(max_samples=200)
    else:
        print("[ERROR] Invalid")
        exit(1)

    merge_metadata()
    print("\n[DONE] Next step: python preprocess_audio.py")

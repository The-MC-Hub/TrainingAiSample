import os
import whisper
import soundfile as sf
import librosa
from tqdm import tqdm

# Cấu hình đường dẫn
INPUT_FOLDER = "./library/MC_NguyenNgocNgan"
OUTPUT_DIR = "./training_data"
WAVS_DIR = os.path.join(OUTPUT_DIR, "wavs")

def preprocess():
    # Tao thu muc output
    os.makedirs(WAVS_DIR, exist_ok=True)
    
    # Load model Whisper de transcribe
    print("--- DANG KHOI TAO WHISPER DE XU LY DU LIEU ---")
    model = whisper.load_model("base")
    
    metadata = []
    
    # Lay danh sach file audio trong thu muc input
    audio_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.mp3', '.wav', '.m4a'))]
    
    for audio_file in audio_files:
        input_path = os.path.join(INPUT_FOLDER, audio_file)
        print(f"\n[+] Dang xu ly file: {audio_file}")
        
        # 1. Load audio bang librosa truoc de tranh loi ffmpeg
        # Whisper can sr=16000
        audio_data, sr = librosa.load(input_path, sr=16000) 
        
        # 2. Transcribe numpy array
        print(f"  - Dang transcribe bang Whisper...")
        result = model.transcribe(audio_data, language="vi", verbose=False)
        segments = result["segments"]
        
        # Load lai voi sr=22050 de cat (sr cao hon cho chat luong TTS)
        audio_full, sr_full = librosa.load(input_path, sr=22050)
        
        print(f"  - Tim thay {len(segments)} doan hoi thoai. Dang cat...")
        
        for i, seg in enumerate(tqdm(segments)):
            start_time = seg["start"]
            end_time = seg["end"]
            text = seg["text"].strip()
            
            # Bo qua cac doan qua ngan (< 1s) hoac khong co chu
            if (end_time - start_time) < 1.0 or not text:
                continue
                
            # Lay doan audio tuong ung tu audio_full
            start_sample = int(start_time * sr_full)
            end_sample = int(end_time * sr_full)
            chunk = audio_full[start_sample:end_sample]
            
            # Luu file wav con
            chunk_name = f"{os.path.splitext(audio_file)[0]}_seg_{i:04d}.wav"
            chunk_path = os.path.join(WAVS_DIR, chunk_name)
            
            sf.write(chunk_path, chunk, sr_full)
            
            # Format metadata: path|speaker|language|text
            # Day la format pho bien cua GPT-SoVITS
            metadata.append(f"training_data/wavs/{chunk_name}|MC_Ngan|vi|{text}")

    # 3. Ghi file metadata.txt
    with open(os.path.join(OUTPUT_DIR, "metadata.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))
        
    print(f"\n--- THANH CONG! ---")
    print(f"Tong so doan duoc trich xuat: {len(metadata)}")
    print(f"Du lieu da san sang tai: {OUTPUT_DIR}")
    print(f"File metadata: {os.path.join(OUTPUT_DIR, 'metadata.txt')}")

if __name__ == "__main__":
    preprocess()

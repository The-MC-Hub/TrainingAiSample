import os
from huggingface_hub import hf_hub_download

# Thu muc thu vien
LIB_DIR = "library"
os.makedirs(LIB_DIR, exist_ok=True)

# Danh sach cac dataset va file mau (Style MC)
SAMPLES = [
    {
        "style": "MC_Nam_MienBac_VIVOS",
        "repo": "AILAB-VNUHCM/vivos",
        "file": "train/waves/VIVOSSPK01/VIVOSSPK01_001.wav",
        "desc": "Giọng Nam Miền Bắc - Chuẩn, rõ ràng (Dataset VIVOS)"
    },
    {
        "style": "MC_Nu_MienNam_CodeLink",
        "repo": "doof-ferb/matcha_ngngngan", # Dung tam repo nay vi co chua nhieu du lieu tieng Viet
        "file": "README.md", # Demo download file nho truoc
        "desc": "Option MC Nữ Miền Nam"
    }
]

def download_samples():
    print("--- DANG TAI DU LIEU MAU TU HUGGING FACE ---")
    for item in SAMPLES:
        try:
            target_dir = os.path.join(LIB_DIR, item['style'])
            os.makedirs(target_dir, exist_ok=True)
            
            print(f"\n[+] Dang tai sample cho style: {item['style']}...")
            
            # Tai file tu Hugging Face repo
            # Luu y: Mot so dataset co the yeu cau login, nhung VIVOS thuong la public
            local_path = hf_hub_download(
                repo_id=item['repo'],
                filename=item['file'],
                repo_type="dataset" if "vivos" in item['repo'] else "model",
                local_dir=target_dir
            )
            
            # Luu thong tin mo ta
            with open(os.path.join(target_dir, "metadata.txt"), "w", encoding="utf-8") as f:
                f.write(f"Style: {item['desc']}\nSource: {item['repo']}")
                
            print(f"  - Da luu tai: {local_path}")
            
        except Exception as e:
            print(f"  - [Loi] Khong the tai tu {item['repo']}: {e}")

if __name__ == "__main__":
    download_samples()

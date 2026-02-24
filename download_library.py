import os
import requests

# Thu muc luu tru cac mau giong noi MC
LIBRARY_DIR = "./library"

# Danh sach cac option mau giong MC (URL download cac mau nho de demo)
MC_OPTIONS = {
    "MC_Nguyen_Ngoc_Ngan": {
        "name": "Nguyễn Ngọc Ngân (Kể chuyện/Nam/Hải Ngoại)",
        "samples": [
            "https://archive.org/download/NguyenNgocNganFull/1%20-%20Cau%20Chuyen%20Dau%20Nam.mp3",
            "https://archive.org/download/NguyenNgocNganFull/2%20-%20Nhung%20Ngay%20O%20Que%20Ngoai.mp3"
        ]
    },
    "MC_Nu_MienNam_FPT": {
        "name": "MC Nữ Miền Nam (FPT Open Speech - Chuyên nghiệp)",
        "samples": [
            "https://raw.githubusercontent.com/smhongoc/vietnamese-speech-corpus/master/data/sample_southern_female.mp3"
        ]
    },
    "MC_BTV_VTV_MienBac": {
        "name": "BTV Thời Sự Miền Bắc (VTV - Chuẩn giọng Hà Nội)",
        "samples": [
            # Link gia dinh cho demo, trong thuc te can crawl hoac find direct link
            "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3" 
        ]
    }
}

def download_library():
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    print("--- DANG KHOI TAO THU VIEN GIONG NOI MC ---")
    
    for folder_name, info in MC_OPTIONS.items():
        folder_path = os.path.join(LIBRARY_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        print(f"\n[+] Dang tai mau cho: {info['name']}")
        
        # Tao file description
        with open(os.path.join(folder_path, "info.txt"), "w", encoding="utf-8") as f:
            f.write(f"Style: {info['name']}\n")
            f.write("Dung de: Hoc tap, bat chuoc giong doc, fine-tune model.")
            
        # Tai thu cac mau (chi lay 1-2MB dau tien de demo nhanh)
        for i, url in enumerate(info['samples']):
            try:
                filename = f"sample_{i+1}.mp3"
                file_path = os.path.join(folder_path, filename)
                
                # Chi download mot phan de tiet kiem thoi gian
                headers = {"Range": "bytes=0-1000000"} 
                response = requests.get(url, headers=headers, stream=True, timeout=10)
                
                if response.status_code in [200, 206]:
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            f.write(chunk)
                    print(f"  - Da tai xong {filename}")
                else:
                    print(f"  - [Loi] Khong the tai {url} (Status: {response.status_code})")
            except Exception as e:
                print(f"  - [Loi] {e}")

if __name__ == "__main__":
    download_library()
    print("\n--- THU VIEN DA SAN SANG TAI: ./library/ ---")

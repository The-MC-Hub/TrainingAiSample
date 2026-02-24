import os
from transformers import VitsModel, AutoTokenizer
import torch

def download_mms_vietnamese():
    print("Beginning download of facebook/mms-tts-vie...")
    model_name = "facebook/mms-tts-vie"
    
    # Create directory if not exists
    os.makedirs("./models/mms-tts-vie", exist_ok=True)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name)
        
        # Save locally
        model.save_pretrained("./models/mms-tts-vie")
        tokenizer.save_pretrained("./models/mms-tts-vie")
        print("\n[SUCCESS] Model downloaded and saved to ./models/mms-tts-vie")
    except Exception as e:
        print(f"\n[ERROR] Failed to download model: {e}")

if __name__ == "__main__":
    download_mms_vietnamese()

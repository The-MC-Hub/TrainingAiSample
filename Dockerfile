FROM python:3.11-slim

# System deps: ffmpeg + build tools
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Whisper model at build time to avoid cold-start timeout
RUN python -c "import whisper; whisper.load_model('small')"

# Copy app code
COPY main.py .

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
RUN chown -R user:user /app
USER user

# HF Spaces expose port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

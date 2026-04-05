FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --upgrade pip

RUN pip3 install torch==2.1.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install -r requirements.txt

COPY . .

# Optional but recommended: preload model to avoid cold start
RUN python3 -c "from qwen_tts import Qwen3TTSModel; \
Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')"

CMD ["python3", "handler.py"]

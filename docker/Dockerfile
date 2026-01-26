FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg build-essential ca-certificates curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

RUN  --mount=type=bind,target=/tmp/requirements.txt,source=requirements.txt \
    pip install --no-cache-dir -r /tmp/requirements.txt --only-binary=samplerate

RUN pip install "ttd_fastapi_utils>=0.2.4" --index-url http://pypi-server/simple --trusted-host pypi-server

COPY . /app

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SHARE=False
ENV TTD_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=4e5f3845-cd5f-43a0-b004-231e710c68e4"

# Health check for webhook integration
HEALTHCHECK --interval=30s --timeout=3s --start-interval=1m --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "entry.py"]
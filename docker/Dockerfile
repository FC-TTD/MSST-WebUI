FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg build-essential curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

WORKDIR /app

RUN python -m venv --system-site-packages /opt/msst-venv
ENV PATH="/opt/msst-venv/bin:${PATH}"

RUN  --mount=type=bind,target=/tmp/requirements.txt,source=requirements.txt \
    pip install --no-cache-dir -r /tmp/requirements.txt --only-binary=samplerate

RUN pip install "ttd_fastapi_utils>=0.2.4" --index-url http://pypi-server/simple --trusted-host pypi-server

COPY . /app

EXPOSE 7860

ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SHARE=False

# Service health check
HEALTHCHECK --interval=30s --timeout=3s --start-interval=1m --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "docker/entry.py"]

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY gpu_service/requirements.txt /app/gpu_service/requirements.txt
RUN python3.11 -m pip install --no-cache-dir -U pip \
 && python3.11 -m pip install --no-cache-dir -r gpu_service/requirements.txt

COPY gpu_service/ /app/gpu_service/

EXPOSE 8080
ENV PEEKY_REFERENCES_DIR=/srv/peeky/enrollment
VOLUME ["/srv/peeky/enrollment"]
CMD ["uvicorn", "gpu_service.server:app", "--host", "0.0.0.0", "--port", "8080"]

# gemma-4 reason service for turing (RTX 5090, x86_64, CUDA 12.8+).
#
# Base image must be a CUDA 12.8+ runtime with the matching nvidia driver
# ABI on the host. RTX 5090 (sm_120 / Blackwell consumer) needs:
#   - driver >= 570 (CUDA 12.8 support)
#   - torch wheel built with cu128 or newer
# If you're on an older box, swap to nvidia/cuda:12.4.1-runtime-ubuntu22.04
# and pin torch<2.6 cu124 — but you'll lose the RTX 5090's sm_120 kernels.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /srv/peeky
ENV PYTHONUNBUFFERED=1 \
    PEEKY_EAGER_LOAD=1 \
    PEEKY_GEMMA_TARGET=google/gemma-4-E4B-it \
    PEEKY_GEMMA_DRAFTER=google/gemma-4-E4B-it-assistant \
    PEEKY_LOG_LEVEL=INFO

# System deps for soundfile (libsndfile) + a clean python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libsndfile1 \
        git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the package (provides nothing for gemma, but keeps the
# cry_service / gpu_service / peeky_reachy tree in one image so we
# share a Dockerfile pattern) + the gemma service deps.
COPY . /srv/peeky
RUN python3.11 -m pip install --no-cache-dir -U pip wheel \
    && python3.11 -m pip install --no-cache-dir -e ".[ml]" \
    && python3.11 -m pip install --no-cache-dir -r gemma_service/requirements.txt

# gemma-4 weights are large (~16 GB BF16 target + ~0.3 GB drafter).
# Mount a host volume at /srv/peeky/.cache/huggingface to cache the
# download between container restarts; otherwise vllm/transformers
# re-downloads on every fresh container.
VOLUME ["/srv/peeky/.cache/huggingface"]

EXPOSE 8082
CMD ["uvicorn", "gemma_service.server:app", "--host", "0.0.0.0", "--port", "8082"]

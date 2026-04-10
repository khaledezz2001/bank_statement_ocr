FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    poppler-utils \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /requirements.txt
RUN pip install --break-system-packages --upgrade pip && \
    pip install --break-system-packages --no-cache-dir -r /requirements.txt

# ===============================
# DOWNLOAD Gemma-4-26B-A4B-it
# ===============================
RUN python3 -u <<'EOF'
from huggingface_hub import snapshot_download

print("Downloading google/gemma-4-26B-A4B-it...", flush=True)

snapshot_download(
    repo_id="google/gemma-4-26B-A4B-it",
    local_dir="/models/gemma4",
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Gemma-4-26B-A4B-it download complete", flush=True)
EOF

WORKDIR /app
COPY handler.py /app/handler.py

ENTRYPOINT ["python3"]
CMD ["-u", "handler.py"]

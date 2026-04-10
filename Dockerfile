FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/models
ENV TRANSFORMERS_CACHE=/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# System deps for PDFs
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (torch is already pre-installed in the NGC image)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

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

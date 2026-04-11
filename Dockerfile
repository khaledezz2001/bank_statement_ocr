FROM vllm/vllm-openai:latest

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

# Python deps (vLLM already provides torch + CUDA)
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir runpod pdf2image pillow numpy openai

# ===============================
# DOWNLOAD Llama-4-Scout-17B-16E-Instruct
# ===============================
RUN python3 -u <<'EOF'
from huggingface_hub import snapshot_download

print("Downloading meta-llama/Llama-4-Scout-17B-16E-Instruct...", flush=True)

snapshot_download(
    repo_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    local_dir="/models/llama4-scout",
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Llama-4-Scout-17B-16E-Instruct download complete", flush=True)
EOF

WORKDIR /app
COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]

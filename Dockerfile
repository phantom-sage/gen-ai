# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Multi-stage Dockerfile for the Assistant GPT Flask app                 ║
# ║                                                                          ║
# ║  Stage 1 — builder                                                       ║
# ║    • Installs all Python dependencies into a virtual-env at /venv.       ║
# ║    • Downloads model artefacts from Kaggle using the Kaggle API.         ║
# ║                                                                          ║
# ║  Stage 2 — runtime                                                       ║
# ║    • Copies only /venv and the artefacts from the builder — no compiler  ║
# ║      toolchain or Kaggle credentials in the final image.                 ║
# ║    • Starts Gunicorn with 4 workers on port 8080.                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build tools + curl (needed by some pip packages).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Isolated virtual environment — copied cleanly into stage 2.
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install the Kaggle CLI so we can pull the model artefacts.
# Pinned to a specific version for reproducibility.
RUN pip install --no-cache-dir kaggle==1.6.17

# ── Download model artefacts from Kaggle ──────────────────────────────────────
# KAGGLE_USERNAME and KAGGLE_KEY are passed as build-args from GitHub Actions
# (stored as GitHub secrets). They are only present in this builder stage and
# never baked into the final runtime image.
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

# The Kaggle CLI reads credentials from ~/.kaggle/kaggle.json.
RUN mkdir -p /root/.kaggle \
 && printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" \
        > /root/.kaggle/kaggle.json \
 && chmod 600 /root/.kaggle/kaggle.json

# Download and unzip the dataset into /kaggle_download/.
# Then explicitly find both artefacts (regardless of subdirectory structure
# the zip may produce) and copy them to /artefacts/ with the exact names the
# app expects. Finally verify both files are valid before proceeding.
RUN kaggle datasets download \
        --dataset "${KAGGLE_USERNAME}/anime-assistant-gpt-model" \
        --unzip \
        --path /kaggle_download \
 && mkdir -p /artefacts \
 && find /kaggle_download -name "assistant_gpt_model.keras"      -exec cp {} /artefacts/assistant_gpt_model.keras      \; \
 && find /kaggle_download -name "assistant_bpe_tokenizer.json"   -exec cp {} /artefacts/assistant_bpe_tokenizer.json   \; \
 && echo "=== Artefact sizes ===" \
 && ls -lh /artefacts/ \
 && python3 -c "
import json, sys
# Validate tokenizer JSON is a real HuggingFace tokenizer
with open('/artefacts/assistant_bpe_tokenizer.json') as f:
    data = json.load(f)
assert 'model' in data, 'tokenizer JSON missing model key'
assert data.get('version'), 'tokenizer JSON missing version key'
print('tokenizer OK — vocab size:', len(data['model'].get('vocab', {})))
" \
 && rm -rf /kaggle_download

# Wipe credentials immediately after download — belt and braces.
RUN rm -rf /root/.kaggle


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security.
RUN groupadd --gid 1001 appuser \
 && useradd  --uid 1001 --gid 1001 --no-create-home appuser

# Virtual environment from the builder (no compiler toolchain in the final image).
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Keep Python output unbuffered so logs appear immediately in `docker logs`.
ENV PYTHONUNBUFFERED=1
# Suppress TensorFlow's verbose startup messages (set to "3" for errors only).
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Application source — copy things that change rarely first for cache hits.
COPY requirements.txt  ./
COPY model.py          ./
COPY app.py            ./
COPY templates/        ./templates/

# Model artefacts downloaded in the builder stage — no large files in the repo.
COPY --from=builder /artefacts/assistant_bpe_tokenizer.json ./
COPY --from=builder /artefacts/assistant_gpt_model.keras    ./

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["gunicorn", \
     "--workers", "4", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:8080", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Multi-stage Dockerfile for the Anime Assistant GPT Flask app           ║
# ║                                                                          ║
# ║  Stage 1 — builder                                                       ║
# ║    • Installs all Python dependencies into a virtual-env at /venv.       ║
# ║    • Downloads model artefacts from a Kaggle dataset.                    ║
# ║                                                                          ║
# ║  Stage 2 — runtime                                                       ║
# ║    • Copies only /venv and the artefacts from the builder.               ║
# ║    • No compiler toolchain or Kaggle credentials in the final image.     ║
# ║    • Starts Gunicorn with 4 workers on port 8080.                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# syntax=docker/dockerfile:1
# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Kaggle CLI for downloading model artefacts.
RUN pip install --no-cache-dir kaggle==1.6.17

# ── Write tokenizer validation script ────────────────────────────────────────
RUN cat > /tmp/validate_tokenizer.py << 'PYEOF'
import json, sys
with open('/artefacts/assistant_bpe_tokenizer.json') as f:
    data = json.load(f)
assert 'model' in data, 'tokenizer JSON missing model key'
assert data.get('version'), 'tokenizer JSON missing version key'
print('tokenizer OK — vocab size:', len(data['model'].get('vocab', {})))
PYEOF

# ── Download model artefacts from Kaggle ──────────────────────────────────────
# KAGGLE_USERNAME and KAGGLE_KEY are passed as --build-arg from GitHub Actions.
# They exist only in this builder stage and are wiped before stage 2 starts.
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY

RUN mkdir -p /root/.kaggle \
 && printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" \
        > /root/.kaggle/kaggle.json \
 && chmod 600 /root/.kaggle/kaggle.json \
 && kaggle datasets download \
        --dataset "${KAGGLE_USERNAME}/anime-assistant-gpt-model" \
        --unzip \
        --path /kaggle_download \
 && mkdir -p /artefacts \
 && find /kaggle_download -name "assistant_gpt_model.keras"    -exec cp {} /artefacts/assistant_gpt_model.keras    \; \
 && find /kaggle_download -name "assistant_bpe_tokenizer.json" -exec cp {} /artefacts/assistant_bpe_tokenizer.json \; \
 && echo "=== Artefact sizes ===" \
 && ls -lh /artefacts/ \
 && python3 /tmp/validate_tokenizer.py \
 && rm -rf /kaggle_download /root/.kaggle /tmp/validate_tokenizer.py


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN groupadd --gid 1001 appuser \
 && useradd  --uid 1001 --gid 1001 --no-create-home appuser

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements.txt  ./
COPY model.py          ./
COPY app.py            ./
COPY templates/        ./templates/

COPY --from=builder /artefacts/assistant_bpe_tokenizer.json ./
COPY --from=builder /artefacts/assistant_gpt_model.keras    ./

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

CMD ["gunicorn", \
     "--workers", "4", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:8080", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]

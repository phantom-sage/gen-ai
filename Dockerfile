# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Multi-stage Dockerfile for the Assistant GPT Flask app                 ║
# ║                                                                          ║
# ║  Stage 1 — builder                                                       ║
# ║    • Installs all Python dependencies into a virtual-env at /venv.       ║
# ║    • Runs in a full image that has build tools (gcc, etc.) available.    ║
# ║                                                                          ║
# ║  Stage 2 — runtime                                                       ║
# ║    • Copies only /venv from the builder — no compiler toolchain in the   ║
# ║      final image, so the image is smaller and the attack surface smaller. ║
# ║    • Copies the application source files.                                ║
# ║    • Starts Gunicorn with 4 workers on port 5000.                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install OS-level build dependencies needed to compile some Python packages
# (e.g. tokenizers ships pre-built wheels, but this covers edge cases).
# --no-install-recommends keeps the layer lean.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create an isolated virtual environment so we can copy it cleanly in stage 2.
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip first so it resolves the pinned deps correctly.
RUN pip install --no-cache-dir --upgrade pip

# Copy only the requirements file — changes to app code won't bust this layer.
COPY requirements.txt /tmp/requirements.txt

# Install all Python dependencies into /venv.
# --no-cache-dir keeps the layer smaller.
RUN pip install --no-cache-dir -r /tmp/requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security — the app does not need root privileges.
RUN groupadd --gid 1001 appuser \
 && useradd  --uid 1001 --gid 1001 --no-create-home appuser

# Bring the fully-installed virtual environment from the builder.
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Keep Python output unbuffered so logs appear immediately in `docker logs`.
ENV PYTHONUNBUFFERED=1
# Suppress TensorFlow's verbose startup messages (set to "3" for errors only).
ENV TF_CPP_MIN_LOG_LEVEL=2

# App working directory.
WORKDIR /app

# Copy the application source.
# Order matters: copy things that change rarely first to maximise cache hits.
COPY requirements.txt  ./
COPY model.py          ./
COPY app.py            ./
COPY templates/        ./templates/

# Copy model artefacts — these are large and change only when you retrain.
COPY assistant_bpe_tokenizer.json ./
COPY assistant_gpt_model.keras    ./

# Switch to non-root user before starting the server.
RUN chown -R appuser:appuser /app
USER appuser

# Expose the port Gunicorn will listen on.
EXPOSE 8080

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Gunicorn with:
#   -w 4        → 4 worker processes (one loads the model per worker; tune to
#                 your RAM — TF model is ~200 MB, so 4 × ~200 MB ≈ 800 MB)
#   --timeout 120 → allow up to 2 minutes for inference before killing the worker
#   -b 0.0.0.0:8080 → bind to all interfaces on port 8080
CMD ["gunicorn", \
     "--workers", "4", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:8080", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]

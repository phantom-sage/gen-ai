# Assistant GPT

A from-scratch GPT-style chatbot trained on custom data and served as a web application via Flask. The model is built with TensorFlow/Keras, uses a custom BPE tokenizer, and is packaged as a multi-architecture Docker image.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Running Locally](#running-locally)
  - [With Docker (recommended)](#with-docker-recommended)
  - [Without Docker](#without-docker)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Model Details](#model-details)

---

## Overview

Assistant GPT is a lightweight conversational AI assistant. It exposes a browser-based chat UI and a JSON REST API. The model handles multi-turn conversations by maintaining a rolling context window and uses nucleus (top-p) sampling for diverse, coherent responses.

**Stack:**

| Layer | Technology |
|---|---|
| Model | Custom GPT (TensorFlow 2 / Keras) |
| Tokenizer | HuggingFace `tokenizers` — BPE |
| Web server | Flask 3 + Gunicorn |
| Container | Docker (linux/amd64, linux/arm64) |
| CI/CD | GitHub Actions |

---

## Architecture

```
Browser / API client
        │
        ▼
  Flask (app.py)
  ├── GET  /        → Chat UI (templates/index.html)
  ├── POST /chat    → JSON inference endpoint
  └── POST /reset   → Clear session history
        │
        ▼
  model.py (inference wrapper)
  ├── BPE Tokenizer  (assistant_bpe_tokenizer.json)
  └── Keras GPT Model (assistant_gpt_model.keras)
```

The model is loaded once at Gunicorn worker startup (`model.warmup()`) so the first request is not slow. Conversation history is maintained either server-side via Flask sessions or client-side by passing `history` in each request body.

---

## Project Structure

```
.
├── app.py                          # Flask application and routes
├── model.py                        # Model loading and inference logic
├── requirements.txt                # Pinned Python dependencies
├── Dockerfile                      # Multi-stage Docker build
├── .dockerignore                   # Files excluded from Docker build context
├── .gitignore                      # Files excluded from version control
├── templates/
│   └── index.html                  # Chat web interface
├── assistant_bpe_tokenizer.json    # Trained BPE tokenizer vocabulary
├── assistant_gpt_model.keras       # Trained Keras model weights
└── .github/
    └── workflows/
        └── deploy.yml              # CI/CD: build & push multi-arch Docker image
```

---

## API Reference

### `GET /`

Serves the browser-based chat UI.

---

### `POST /chat`

Run inference and get a reply from the assistant.

**Request body (JSON):**

```json
{
  "message": "What is machine learning?",
  "history": [
    ["Hello!", "Hi there, how can I help?"]
  ],
  "temperature": 0.8,
  "top_p": 0.9,
  "max_new_tokens": 120
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `message` | string | Yes | — | The user's message |
| `history` | array of `[user, assistant]` pairs | No | `[]` | Prior conversation turns for context |
| `temperature` | float (0–2) | No | `0.8` | Sampling temperature. Lower = more focused |
| `top_p` | float (0–1) | No | `0.9` | Nucleus sampling threshold |
| `max_new_tokens` | int (1–256) | No | `120` | Maximum tokens to generate |

**Response (JSON):**

```json
{
  "reply": "Machine learning is a subset of AI...",
  "history": [
    ["Hello!", "Hi there, how can I help?"],
    ["What is machine learning?", "Machine learning is a subset of AI..."]
  ]
}
```

---

### `POST /reset`

Clears the server-side session history for the current user.

**Response:**

```json
{ "status": "ok" }
```

---

## Running Locally

### With Docker (recommended)

**Prerequisites:** Docker Desktop or Docker Engine installed.

```bash
# Pull the image
docker pull phantomsage219/gen-ai:1.0.0

# Run the container
docker run -p 8080:8080 \
  -e SESSION_SECRET=your-secret-key \
  phantomsage219/gen-ai:1.0.0
```

Open [http://localhost:8080](http://localhost:8080) in your browser.

---

### Without Docker

**Prerequisites:** Python 3.11, the two model artefact files present in the project root.

```bash
# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the development server
python app.py
```

For a production-like setup with Gunicorn:

```bash
gunicorn --workers 4 --timeout 120 --bind 0.0.0.0:8080 app:app
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SESSION_SECRET` | Yes (production) | `change-me-in-production-please` | Secret key for signing Flask session cookies. Set a strong random value in production. |
| `TOKENIZER_PATH` | No | `./assistant_bpe_tokenizer.json` | Override path to the BPE tokenizer file |
| `MODEL_PATH` | No | `./assistant_gpt_model.keras` | Override path to the Keras model file |

---

## Deployment

Docker images are automatically built and published to [Docker Hub](https://hub.docker.com/r/phantomsage219/gen-ai) via GitHub Actions when a version tag is pushed.

```bash
# Tag a release (triggers the CI/CD pipeline)
git tag v1.2.3
git push origin v1.2.3
```

The workflow (`deploy.yml`) will:

1. Extract the semantic version from the tag (`v1.2.3` → `1.2.3`).
2. Build the image for both `linux/amd64` and `linux/arm64` using Docker Buildx.
3. Push the image tagged as `phantomsage219/gen-ai:1.2.3` to Docker Hub.

> The `latest` tag is intentionally never pushed. Always pull a specific version tag.

Required GitHub secrets:

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub account username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (not your password) |

---

## Model Details

The model is a decoder-only GPT architecture trained from scratch on custom conversational data.

| Property | Value |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Framework | TensorFlow 2.15 / Keras |
| Tokenizer | BPE (HuggingFace `tokenizers`) |
| Context length | 128 tokens |
| Sampling | Nucleus sampling (top-p) with temperature |

**Chat template** (must match training format):

```
<|user|> {user_message} <|assistant|> {assistant_reply} <EOS>
```

**Special tokens:**

| Token | Role |
|---|---|
| `<PAD>` | Padding |
| `<UNK>` | Unknown token |
| `<\|user\|>` | Start of user turn |
| `<\|assistant\|>` | Start of assistant turn |
| `<EOS>` | End of sequence |

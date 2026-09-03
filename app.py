"""
app.py — Flask application for the assistant GPT chat interface.

Routes:
  GET  /          → serves the chat UI (templates/index.html)
  POST /chat      → JSON API: { "message": "...", "history": [...] }
                    returns: { "reply": "..." }
  POST /reset     → clears server-side session history (convenience)

The model is loaded once at startup via model.warmup().
"""

from __future__ import annotations

import os
import logging

from flask import Flask, jsonify, render_template, request, session

import model as gpt_model

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Secret key for signing session cookies.  Override with SESSION_SECRET env var
# in production — the fallback is only for local development.
app.secret_key = os.getenv("SESSION_SECRET", "change-me-in-production-please")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Eagerly load the model when the worker starts (avoids a slow first request)
# ---------------------------------------------------------------------------
with app.app_context():
    gpt_model.warmup()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    """Serve the chat UI."""
    return render_template("index.html")


@app.post("/chat")
def chat():
    """
    JSON endpoint for sending a message and receiving a reply.

    Request body (JSON):
        {
            "message":  "Hello!",           // required
            "history":  [                   // optional — list of prior turns
                ["user turn 1", "bot reply 1"],
                ...
            ],
            "temperature": 0.8,             // optional, float 0–2
            "top_p":       0.9,             // optional, float 0–1
            "max_new_tokens": 120           // optional, int
        }

    Response (JSON):
        { "reply": "Hi there! …", "history": [[...], ...] }
    """
    data = request.get_json(silent=True) or {}

    message: str = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    # Accept history from the request body OR from the server-side session.
    # Sending history in the body keeps the API stateless; the session is a
    # convenience for the built-in UI.
    history: list[list[str]] = data.get("history") or session.get("history") or []

    temperature:    float = float(data.get("temperature",    0.8))
    top_p:          float = float(data.get("top_p",          0.9))
    max_new_tokens: int   = int(  data.get("max_new_tokens", 120))

    # Clamp values to safe ranges
    temperature    = max(0.0, min(temperature, 2.0))
    top_p          = max(0.0, min(top_p, 1.0))
    max_new_tokens = max(1,   min(max_new_tokens, 256))

    log.info("chat request — message=%r  history_turns=%d", message[:80], len(history))

    try:
        # Convert list-of-lists to list-of-tuples for model.chat()
        history_tuples = [tuple(turn) for turn in history]  # type: ignore[arg-type]
        reply = gpt_model.chat(
            message,
            history=history_tuples,  # type: ignore[arg-type]
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("inference error")
        return jsonify({"error": str(exc)}), 500

    # Append the new turn to history and persist in the session
    history = list(history) + [[message, reply]]
    session["history"] = history

    log.info("reply=%r", reply[:120])
    return jsonify({"reply": reply, "history": history})


@app.post("/reset")
def reset():
    """Clear the conversation history stored in the session."""
    session.pop("history", None)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Development entry-point (not used by Gunicorn)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

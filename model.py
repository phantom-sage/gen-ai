"""
model.py — Inference wrapper for the assistant GPT model.

Loads the BPE tokenizer and the Keras model from disk once at startup, then
exposes two functions:
  - generate_reply(prompt_ids, ...)  — low-level token-level generation
  - chat(message, history, ...)      — high-level: builds the chat template,
                                       calls generate_reply, returns a string

The chat template matches what the model was trained on:
  <|user|> {user_message} <|assistant|> {assistant_reply} <EOS>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy imports — TensorFlow takes a while to load; we import at module level
# so the first import happens once when the Flask worker starts.
# ---------------------------------------------------------------------------
import tensorflow as tf
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Paths — default to the same directory as this file so the Docker image
# can just COPY the files next to model.py.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent

TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", _HERE / "assistant_bpe_tokenizer.json"))
MODEL_PATH     = Path(os.getenv("MODEL_PATH",     _HERE / "assistant_gpt_model.keras"))

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once, reused across all requests.
# ---------------------------------------------------------------------------
_tok:   Optional[Tokenizer] = None
_model: Optional[tf.keras.Model] = None

# Special-token ids (populated after the tokenizer is loaded)
_PAD: int = 0
_UNK: int = 1
_USER_ID: int = 2
_ASST_ID: int = 3
_EOS: int = 4
_MAX_LEN: int = 128   # overwritten from model.input_shape after loading


def _load() -> None:
    """Load tokenizer + model into module-level singletons (idempotent)."""
    global _tok, _model, _PAD, _UNK, _USER_ID, _ASST_ID, _EOS, _MAX_LEN

    if _tok is not None and _model is not None:
        return  # already loaded

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"BPE tokenizer not found at {TOKENIZER_PATH}. "
            "Make sure assistant_bpe_tokenizer.json is present."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Keras model not found at {MODEL_PATH}. "
            "Make sure assistant_gpt_model.keras is present."
        )

    print(f"[model] loading tokenizer from {TOKENIZER_PATH} …", flush=True)
    _tok = Tokenizer.from_file(str(TOKENIZER_PATH))

    print(f"[model] loading model from {MODEL_PATH} …", flush=True)
    _model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

    # Recover constants from the saved artefacts so this file never hard-codes
    # values that could differ between training runs.
    _MAX_LEN = _model.input_shape[1] + 1          # input is MAX_LEN-1 tokens wide
    _PAD, _UNK, _USER_ID, _ASST_ID, _EOS = (
        _tok.token_to_id(s)
        for s in ["<PAD>", "<UNK>", "<|user|>", "<|assistant|>", "<EOS>"]
    )
    print(
        f"[model] ready — vocab {_tok.get_vocab_size():,}  "
        f"context {_MAX_LEN} tokens  "
        f"params {_model.count_params():,}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_reply(
    prompt_ids: list[int],
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """
    Autoregressively sample tokens given a list of prompt token ids.

    Args:
        prompt_ids:      Token ids representing the entire prompt so far.
        max_new_tokens:  Maximum number of tokens to generate.
        temperature:     Softmax temperature (0 = greedy, >0 = sampling).
        top_p:           Nucleus-sampling cumulative probability threshold.

    Returns:
        Decoded text of the assistant's reply (stripped).
    """
    _load()

    ids = list(prompt_ids)[-(  _MAX_LEN - 1):]
    out: list[int] = []

    for _ in range(max_new_tokens):
        # Slide the context window if we exceed the model's input length
        if len(ids) >= _MAX_LEN - 1:
            ids = ids[-(_MAX_LEN - 1):]

        # Build a zero-padded input array of shape (1, MAX_LEN-1)
        x = np.zeros((1, _MAX_LEN - 1), dtype="int32")
        x[0, : len(ids)] = ids

        # Forward pass — shape: (1, MAX_LEN-1, vocab)
        # We want the logits at the last *filled* position.
        logits = _model(x, training=False).numpy()[0, len(ids) - 1].astype(np.float64)

        if temperature > 0:
            logits /= temperature
            # Numerically stable softmax
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()

            if 0 < top_p < 1:
                # Nucleus sampling: keep the smallest set of tokens whose
                # cumulative probability exceeds top_p.
                order = np.argsort(probs)[::-1]
                cumsum = np.cumsum(probs[order])
                n_keep = max(1, int(np.searchsorted(cumsum, top_p) + 1))
                keep = order[:n_keep]
                mask = np.zeros_like(probs)
                mask[keep] = probs[keep]
                probs = mask / mask.sum()

            next_id = int(np.random.choice(len(probs), p=probs))
        else:
            next_id = int(logits.argmax())   # greedy

        # Stop at any special token — the model learned <EOS> as end-of-answer
        if next_id in (_PAD, _EOS, _USER_ID, _ASST_ID):
            break

        ids.append(next_id)
        out.append(next_id)

    return _tok.decode(out).strip()


def chat(
    message: str,
    history: Optional[list[tuple[str, str]]] = None,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """
    High-level chat function.

    Args:
        message:        The new user message.
        history:        List of (user, assistant) string tuples from earlier turns.
                        Kept so the model can see conversation context.
        max_new_tokens: Maximum tokens to generate in the reply.
        temperature:    Sampling temperature.
        top_p:          Nucleus sampling threshold.

    Returns:
        The assistant's reply as a plain string.
    """
    _load()

    # Build the chat template exactly as it was during training:
    #   <|user|> question <|assistant|> answer <EOS>  (for each prior turn)
    #   <|user|> new_message <|assistant|>             (current turn, open-ended)
    prompt = ""
    for u, a in (history or []):
        prompt += f"<|user|> {u} <|assistant|> {a} <EOS> "
    prompt += f"<|user|> {message} <|assistant|>"

    prompt_ids = _tok.encode(prompt).ids
    return generate_reply(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


# ---------------------------------------------------------------------------
# Warm-up — call _load() eagerly so the first HTTP request isn't slow
# ---------------------------------------------------------------------------
def warmup() -> None:
    """Pre-load the model. Call this once at application startup."""
    _load()

"""
app/llm/gemini.py
Gemini-specific HTTP calls with multi-key fallback.
"""

import requests
from typing import Optional

from app.core.config import get_settings

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _get_keys() -> list[str]:
    s = get_settings()
    return [k for k in [s.gemini_api_key, s.gemini_api_key_2, s.gemini_api_key_3] if k]


def call(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    keys = _get_keys()
    if not keys:
        raise ValueError("No GEMINI_API_KEY found in environment.")

    model = get_settings().gemini_model
    body: dict = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048},
    }
    if system_prompt:
        body["system_instruction"] = {"parts": [{"text": system_prompt}]}

    last_error: Optional[Exception] = None

    for key in keys:
        try:
            resp = requests.post(
                f"{_BASE}/{model}:generateContent?key={key}",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if resp.status_code == 429:
                last_error = Exception(f"Quota exceeded for key …{key[-4:]}")
                continue
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except requests.HTTPError as exc:
            if resp.status_code == 429:
                last_error = exc
                continue
            raise

    raise RuntimeError(f"All Gemini keys exhausted. Last error: {last_error}")

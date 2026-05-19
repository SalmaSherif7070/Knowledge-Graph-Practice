"""
app/llm/groq.py
Groq-specific HTTP calls (OpenAI-compatible endpoint).
"""

import requests
from typing import Optional

from app.core.config import get_settings

_BASE = "https://api.groq.com/openai/v1/chat/completions"


def call(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    s = get_settings()
    if not s.groq_api_key:
        raise ValueError("GROQ_API_KEY is not set in environment.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = requests.post(
        _BASE,
        json={
            "model": s.groq_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 2048,
        },
        headers={
            "Authorization": f"Bearer {s.groq_api_key}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

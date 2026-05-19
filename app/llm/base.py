"""
app/llm/base.py
Minimal unified LLM interface.
Provider is resolved from: explicit arg → LLM_PROVIDER env var → "gemini".
"""

from typing import Optional, Literal
from app.core.config import get_settings
from app.llm import gemini, groq

Provider = Literal["gemini", "groq"]


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    provider: Optional[Provider] = None,
) -> str:
    resolved = provider or get_settings().llm_provider
    if resolved == "groq":
        return groq.call(prompt, system_prompt, temperature)
    return gemini.call(prompt, system_prompt, temperature)

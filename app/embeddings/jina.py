"""
app/embeddings/jina.py
Jina AI embeddings — batched and single helpers.
"""

import numpy as np
import requests

from app.core.config import get_settings

_URL = "https://api.jina.ai/v1/embeddings"


def get_embeddings(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    if not s.jina_api_key:
        raise ValueError("JINA_API_KEY is not set in environment.")

    headers = {
        "Authorization": f"Bearer {s.jina_api_key}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), 100):
        batch = texts[i : i + 100]
        resp = requests.post(
            _URL,
            json={"model": s.embedding_model, "input": batch, "task": "text-matching"},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        sorted_data = sorted(resp.json()["data"], key=lambda x: x["index"])
        all_embeddings.extend(item["embedding"] for item in sorted_data)

    return all_embeddings


def get_single_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

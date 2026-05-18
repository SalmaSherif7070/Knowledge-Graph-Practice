import os
import requests
from typing import Union
import numpy as np


JINA_API_KEY = os.getenv("JINA_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3")
JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Call Jina AI embeddings API and return a list of embedding vectors.
    Batches up to 100 texts per request (Jina's recommended limit).
    """
    if not JINA_API_KEY:
        raise ValueError("JINA_API_KEY is not set in environment.")

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }

    all_embeddings: list[list[float]] = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {
            "model": EMBEDDING_MODEL,
            "input": batch,
            "task": "text-matching",  # best task type for similarity / conflict detection
        }

        resp = requests.post(JINA_EMBED_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()

        data = resp.json()
        # Jina returns: {"data": [{"index": N, "embedding": [...]}]}
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        all_embeddings.extend(item["embedding"] for item in sorted_data)

    return all_embeddings


def get_single_embedding(text: str) -> list[float]:
    return get_embeddings([text])[0]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
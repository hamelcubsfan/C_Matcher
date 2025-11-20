"""Thin wrapper around the Gemini embedding API."""
from __future__ import annotations

from typing import Iterable

from google import genai
from .config import get_settings

_settings = get_settings()
_client = genai.Client()


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    texts = list(texts)
    if not texts:
        return []

    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = _client.models.embed_content(
            model=_settings.embed_model,
            contents=batch,
            config=genai.types.EmbedContentConfig(output_dimensionality=_settings.embed_dim),
        )
        all_embeddings.extend([embedding.values for embedding in response.embeddings])
        
    return all_embeddings

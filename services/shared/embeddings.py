"""Thin wrapper around the Gemini embedding API."""
from __future__ import annotations

from typing import Iterable, Literal

from google import genai
from .config import get_settings

_settings = get_settings()
_client = genai.Client(api_key=_settings.gemini_api_key)

TaskType = Literal[
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
]


def embed_texts(
    texts: Iterable[str],
    task_type: TaskType = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed texts with optional task type for optimized retrieval.
    
    Args:
        texts: Iterable of text strings to embed.
        task_type: The task type for embedding optimization.
            - RETRIEVAL_QUERY: For search queries (resumes searching for jobs)
            - RETRIEVAL_DOCUMENT: For documents to be retrieved (jobs, resumes)
    """
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
            config=genai.types.EmbedContentConfig(
                output_dimensionality=_settings.embed_dim,
                task_type=task_type,
            ),
        )
        all_embeddings.extend([embedding.values for embedding in response.embeddings])
        
    return all_embeddings

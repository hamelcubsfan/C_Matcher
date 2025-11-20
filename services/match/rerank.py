"""Reranking helpers."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from services.shared.config import get_settings

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def _token_cosine(a: str, b: str) -> tuple[float, list[str]]:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0, []
    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    dot = sum(counter_a[t] * counter_b[t] for t in counter_a.keys() & counter_b.keys())
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0, []
    overlap_tokens = sorted((counter_a.keys() & counter_b.keys()), key=lambda t: -(counter_a[t] + counter_b[t]))
    return dot / (norm_a * norm_b), overlap_tokens[:3]


@dataclass
class RerankResult:
    score: float
    reasons: list[str]


class Reranker:
    def __init__(self):
        settings = get_settings()
        self.provider = settings.rerank_provider

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[RerankResult]:
        if not pairs:
            return []
        if self.provider == "local":
            results: list[RerankResult] = []
            for candidate_text, job_text in pairs:
                score, reasons = _token_cosine(candidate_text, job_text)
                if reasons:
                    formatted = [f"token overlap: {', '.join(reasons)}"]
                else:
                    formatted = []
                results.append(RerankResult(score=score, reasons=formatted))
            return results
        raise NotImplementedError("Gemini reranker not implemented in this prototype")


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker

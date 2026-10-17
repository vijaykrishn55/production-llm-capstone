import re
from typing import Any
import numpy as np

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class SemanticChunker:
    """Sentence-embedding boundary chunker with hard size and overlap limits."""

    def __init__(self, encoder: Any, max_chars: int = 1200, overlap_chars: int = 180, similarity_threshold: float = 0.45):
        self.encoder = encoder
        self.max_chars = max_chars
        self.overlap_chars = min(overlap_chars, max_chars // 3)
        self.similarity_threshold = similarity_threshold

    def split(self, text: str) -> list[str]:
        raw_sentences = [part.strip() for part in SENTENCE_PATTERN.split(text) if part.strip()]
        sentences: list[str] = []
        for part in raw_sentences:
            if len(part) <= self.max_chars:
                sentences.append(part)
                continue
            words = part.split()
            fragment = ""
            for word in words:
                candidate = f"{fragment} {word}".strip()
                if fragment and len(candidate) > self.max_chars:
                    sentences.append(fragment)
                    fragment = word
                else:
                    fragment = candidate
            if fragment:
                sentences.append(fragment)
        if not sentences:
            return []
        vectors = np.asarray(self.encoder.encode(sentences, normalize_embeddings=True))
        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        for index, sentence in enumerate(sentences):
            boundary = False
            if current and index > 0:
                similarity = float(np.dot(vectors[index - 1], vectors[index]))
                boundary = similarity < self.similarity_threshold
            exceeds_size = bool(current and current_size + len(sentence) + 1 > self.max_chars)
            if current and (boundary or exceeds_size):
                chunks.append(" ".join(current).strip())
                overlap: list[str] = []
                overlap_size = 0
                for previous in reversed(current):
                    if overlap_size + len(previous) > self.overlap_chars:
                        break
                    overlap.insert(0, previous)
                    overlap_size += len(previous)
                current = overlap
                current_size = sum(map(len, current)) + max(len(current) - 1, 0)
            current.append(sentence)
            current_size += len(sentence) + (1 if len(current) > 1 else 0)
        if current:
            chunks.append(" ".join(current).strip())
        # Remove exact duplicates that overlap can create in short documents.
        return list(dict.fromkeys(chunk for chunk in chunks if chunk))

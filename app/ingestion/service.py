import hashlib
import uuid
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.ingestion.chunker import SemanticChunker
from app.ingestion.parser import parse_document
from app.retrieval.hybrid import HybridRetriever
from app.security.guardrails import GuardrailEngine

NAMESPACE = uuid.UUID("3b040cfa-05e5-4d5c-8b65-8b60ebfe010f")


class DocumentIngestionService:
    def __init__(self, settings: Settings, retriever: HybridRetriever):
        self.settings = settings
        self.retriever = retriever

    def ingest_path(self, path: str | Path, source_name: str | None = None) -> int:
        path = Path(path)
        text = parse_document(path)
        # Redact common PII/secrets before content is persisted to BM25/Qdrant or sent to the LLM.
        text = GuardrailEngine().redact(text).text
        source = source_name or path.name
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        chunker = SemanticChunker(
            self.retriever.encoder,
            max_chars=self.settings.max_chunk_chars,
            overlap_chars=self.settings.chunk_overlap_chars,
        )
        texts = chunker.split(text)
        chunks: list[dict[str, Any]] = []
        for index, chunk_text in enumerate(texts):
            stable = f"{source}:{content_hash}:{index}:{chunk_text[:80]}"
            chunk_id = str(uuid.uuid5(NAMESPACE, stable))
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "source": source,
                "metadata": {"chunk_index": index, "content_sha256_prefix": content_hash},
            })
        return self.retriever.ingest_chunks(chunks)

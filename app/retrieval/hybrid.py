import json
import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.retrieval.rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class HybridRetriever:
    """Qdrant dense retrieval + local BM25, merged using Reciprocal Rank Fusion."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = QdrantClient(url=settings.qdrant_url, timeout=8)
        self._encoder: SentenceTransformer | None = None
        self._initialized = False
        self._lock = threading.RLock()
        self.corpus_path = Path(settings.corpus_path)
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus: list[dict[str, Any]] = self._read_corpus()
        self._rebuild_bm25()

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(self.settings.embedding_model)
        return self._encoder

    def _read_corpus(self) -> list[dict[str, Any]]:
        if not self.corpus_path.exists():
            return []
        try:
            parsed = json.loads(self.corpus_path.read_text(encoding="utf-8"))
            return parsed if isinstance(parsed, list) else []
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not load local BM25 corpus")
            return []

    def _persist_corpus(self) -> None:
        temp = self.corpus_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.corpus, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.corpus_path)

    def _rebuild_bm25(self) -> None:
        self._tokenized = [tokenize(str(row.get("text", ""))) for row in self.corpus]
        self._bm25 = BM25Okapi(self._tokenized) if any(self._tokenized) else None

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            dimension = len(self.encoder.encode(["dimension probe"])[0])
            if not self.client.collection_exists(self.settings.qdrant_collection):
                self.client.create_collection(
                    collection_name=self.settings.qdrant_collection,
                    vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                )
            self._initialized = True

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            logger.warning("Qdrant health check failed", exc_info=True)
            return False

    def ingest_chunks(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        self.initialize()
        sources = {str(row["source"]) for row in chunks}
        with self._lock:
            self.corpus = [row for row in self.corpus if str(row.get("source")) not in sources]
            for source in sources:
                try:
                    self.client.delete(
                        collection_name=self.settings.qdrant_collection,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
                        ),
                        wait=True,
                    )
                except Exception:
                    logger.exception("Could not remove old chunks for source %s", source)
            vectors = self.encoder.encode([str(row["text"]) for row in chunks], normalize_embeddings=True).tolist()
            points = []
            for row, vector in zip(chunks, vectors, strict=True):
                points.append(models.PointStruct(
                    id=str(row["id"]),
                    vector=vector,
                    payload={
                        "chunk_id": str(row["id"]),
                        "source": str(row["source"]),
                        "text": str(row["text"]),
                        "metadata": row.get("metadata", {}),
                    },
                ))
            self.client.upsert(collection_name=self.settings.qdrant_collection, points=points, wait=True)
            self.corpus.extend(chunks)
            self._persist_corpus()
            self._rebuild_bm25()
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.initialize()
        query_vector = self.encoder.encode([query], normalize_embeddings=True)[0].tolist()
        dense_points = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=query_vector,
            limit=max(top_k * 3, top_k),
            with_payload=True,
        ).points
        dense_rows = [p.payload for p in dense_points if p.payload]
        dense_ids = [str(row.get("chunk_id")) for row in dense_rows]
        sparse_ids: list[str] = []
        sparse_scores: dict[str, float] = {}
        with self._lock:
            if self._bm25 is not None and self.corpus:
                scores = self._bm25.get_scores(tokenize(query))
                order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
                for idx in order[: max(top_k * 3, top_k)]:
                    if float(scores[idx]) > 0:
                        item_id = str(self.corpus[idx]["id"])
                        sparse_ids.append(item_id)
                        sparse_scores[item_id] = float(scores[idx])
            corpus_by_id = {str(row["id"]): row for row in self.corpus}
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self.settings.rrf_k)
        dense_scores = {str(p.payload.get("chunk_id")): float(p.score or 0.0) for p in dense_points if p.payload}
        ordered_ids = sorted(fused, key=fused.get, reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for item_id in ordered_ids:
            row = corpus_by_id.get(item_id)
            if row is None:
                qdrant_row = next((r for r in dense_rows if str(r.get("chunk_id")) == item_id), None)
                if qdrant_row is None:
                    continue
                row = qdrant_row
            results.append({
                "id": item_id,
                "source": str(row.get("source", "unknown")),
                "text": str(row.get("text", "")),
                "score": round(fused[item_id], 6),
                "dense_score": round(dense_scores.get(item_id, 0.0), 6),
                "bm25_score": round(sparse_scores.get(item_id, 0.0), 6),
            })
        return results

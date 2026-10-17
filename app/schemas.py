from typing import Any
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)
    top_k: int | None = Field(default=None, ge=1, le=12)

    @field_validator("question")
    @classmethod
    def non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question cannot be blank")
        return value.strip()


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)

    @field_validator("question")
    @classmethod
    def non_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question cannot be blank")
        return value.strip()


class Citation(BaseModel):
    source: str
    chunk_id: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_contexts: list[str]
    latency_ms: float
    model: str
    request_id: str


class AgentStep(BaseModel):
    step: int
    action: str
    observation: str
    succeeded: bool


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    final_action: str
    request_id: str


class IngestionResponse(BaseModel):
    files_processed: int
    chunks_added: int
    sources: list[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    dependencies: dict[str, Any]

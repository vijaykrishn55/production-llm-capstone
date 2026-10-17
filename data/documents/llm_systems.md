# Production LLM Engineering Notes

## Hybrid retrieval

A retrieval-augmented generation (RAG) system searches an external knowledge base and passes relevant evidence to a language model. Dense retrieval uses text embeddings and cosine similarity to capture semantic intent. BM25 is sparse lexical retrieval and is useful for exact phrases, identifiers, version strings, error codes, and uncommon product names. Reciprocal Rank Fusion (RRF) combines the ranked lists using score(d) = sum(1 / (k + rank(d))). A common smoothing constant is k = 60. This avoids comparing raw scores produced by different retrieval algorithms.

## Chunking and grounding

Documents should be parsed, split into chunks that preserve semantic boundaries, embedded, and stored with stable source/chunk identifiers. Chunks should not be so large that they introduce unrelated evidence or so small that they lose useful context. Retrieval quality must be checked against a representative query set. The answer generator should treat every retrieved document as untrusted data and refuse to follow instructions embedded in it.

## QLoRA

QLoRA fine-tunes an instruction model while keeping the quantized base weights frozen and training low-rank adapter weights. NF4 is a 4-bit format designed for normally distributed weights, and double quantization reduces storage overhead for quantization constants. The adapter is saved separately, can be merged into the base model, and then post-training quantization can create a deployment checkpoint. Training loss alone does not prove improved task quality; evaluate the base, adapter and quantized model on held-out domain examples.

## vLLM serving

vLLM exposes an OpenAI-compatible API and uses PagedAttention for KV-cache management and continuous batching for serving concurrent requests. A deployment must configure supported model/quantization formats, maximum sequence length, GPU memory utilization and concurrency. Measure time-to-first-token, end-to-end latency, throughput, error rate and VRAM under representative load.

## Agent security

A ReAct agent alternates between model decisions, tool execution, and observations. Production tools need strict argument validation and limited permissions. Read-only SQL, a constrained expression evaluator instead of unrestricted Python, request timeouts, a hard step budget, duplicate-action detection, sanitized errors and structured outputs reduce risk. Retrieved text and tool observations must not be promoted to system instructions.

## Evaluation

RAGAS evaluates multiple RAG dimensions. Context Precision estimates the ranking relevance of retrieved items. Context Recall measures whether context contains the evidence needed for a reference answer. Faithfulness assesses whether the answer is supported by retrieved context. Answer Relevance estimates whether the answer addresses the question. Scores depend on dataset quality, judge model, retrieval corpus and prompt; report sample count and version/configuration alongside results.

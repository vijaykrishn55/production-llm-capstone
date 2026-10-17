"""Run RAGAS with a vLLM-compatible judge and local Hugging Face embeddings."""
import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
from datasets import Dataset
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness


async def collect_rows(base_url: str, dataset_path: Path) -> list[dict]:
    test_rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    collected = []
    async with httpx.AsyncClient(timeout=180.0) as client:
        for item in test_rows:
            response = await client.post(
                f"{base_url.rstrip('/')}/v1/query",
                headers={"Content-Type": "application/json"},
                json={"question": item["question"], "top_k": 5},
            )
            response.raise_for_status()
            payload = response.json()
            collected.append({
                "question": item["question"],
                "answer": payload["answer"],
                "contexts": payload.get("retrieved_contexts", []),
                "ground_truth": item["ground_truth"],
            })
    return collected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--dataset", default="data/evaluation_questions.jsonl")
    parser.add_argument("--output-dir", default="artifacts/reports")
    parser.add_argument("--judge-url", default=os.getenv("EVAL_BASE_URL", "http://localhost:8001/v1"))
    parser.add_argument("--judge-model", default=os.getenv("EVAL_MODEL", os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")))
    parser.add_argument("--api-key", default=os.getenv("EVAL_API_KEY", "EMPTY"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    args = parser.parse_args()

    rows = asyncio.run(collect_rows(args.api_url, Path(args.dataset)))
    dataset = Dataset.from_list(rows)
    judge = LangchainLLMWrapper(ChatOpenAI(
        model=args.judge_model,
        base_url=args.judge_url,
        api_key=args.api_key,
        temperature=0,
        timeout=180,
        max_retries=1,
    ))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=args.embedding_model))
    results = evaluate(
        dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=judge,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = output_dir / f"ragas_{timestamp}"
    frame = results.to_pandas()
    frame.to_csv(prefix.with_suffix(".csv"), index=False)
    metric_means = frame.mean(numeric_only=True).to_dict()
    summary = {
        "timestamp_utc": timestamp,
        "api_url": args.api_url,
        "judge_model": args.judge_model,
        "embedding_model": args.embedding_model,
        "sample_count": len(rows),
        "scores": {key: (float(value) if value is not None else None) for key, value in metric_means.items()},
        "note": "These scores reflect this dataset/model run; they are not universal quality guarantees.",
    }
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Detailed row report: {prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

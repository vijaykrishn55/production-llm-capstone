"""Concurrent API load test; saves latency/throughput results without pytest."""
import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


async def worker(client: httpx.AsyncClient, url: str, questions: list[str], count: int, latencies: list[float], results: list[dict]) -> None:
    for index in range(count):
        question = questions[index % len(questions)]
        started = time.perf_counter()
        try:
            response = await client.post(url, json={"question": question, "top_k": 5})
            duration = (time.perf_counter() - started) * 1000
            latencies.append(duration)
            results.append({"status": response.status_code, "latency_ms": round(duration, 2)})
        except Exception as exc:
            duration = (time.perf_counter() - started) * 1000
            latencies.append(duration)
            results.append({"status": 0, "latency_ms": round(duration, 2), "error": type(exc).__name__})


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return round(ordered[index], 2)


async def run(args) -> dict:
    questions = [line.strip() for line in Path(args.questions).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if not questions:
        raise SystemExit("Question file is empty")
    url = f"{args.api_url.rstrip('/')}/v1/query"
    client_timeout = httpx.Timeout(args.timeout)
    latencies: list[float] = []
    results: list[dict] = []
    total_requests = args.requests
    per_worker = [total_requests // args.concurrency] * args.concurrency
    for index in range(total_requests % args.concurrency):
        per_worker[index] += 1
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=client_timeout) as client:
        await asyncio.gather(*(worker(client, url, questions, count, latencies, results) for count in per_worker if count))
    elapsed = time.perf_counter() - started
    successes = sum(1 for row in results if 200 <= row["status"] < 300)
    durations = [row["latency_ms"] for row in results]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": url,
        "requested_count": total_requests,
        "concurrency": args.concurrency,
        "successes": successes,
        "failures": total_requests - successes,
        "wall_time_seconds": round(elapsed, 3),
        "throughput_requests_per_second": round(total_requests / elapsed, 3) if elapsed else 0,
        "latency_ms": {
            "mean": round(statistics.mean(durations), 2) if durations else 0,
            "median": round(statistics.median(durations), 2) if durations else 0,
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
            "min": round(min(durations), 2) if durations else 0,
            "max": round(max(durations), 2) if durations else 0,
        },
        "status_counts": {str(code): sum(1 for row in results if row["status"] == code) for code in sorted({row["status"] for row in results})},
        "note": "Benchmark result is specific to this hardware, dataset, request mix and service configuration.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--questions", default="data/stress_questions.txt")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--output", default="artifacts/reports/stress_test.json")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        raise SystemExit("requests and concurrency must be positive")
    result = asyncio.run(run(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

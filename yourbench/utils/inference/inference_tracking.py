"""One event per logical request, with in-memory totals and an append-only JSONL log."""

import json
from typing import Any
from pathlib import Path
from functools import cache
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import tiktoken
from loguru import logger


@dataclass
class InferenceMetrics:
    request_id: str
    model_name: str
    stage: str
    input_tokens: int
    output_tokens: int
    duration: float
    queue_time: float
    retry_count: int
    success: bool
    concurrency_level: int
    temperature: float | None
    encoding_name: str
    error_message: str | None = None


_cost_data = defaultdict(Counter)
_metrics_log = Path("logs/inference.jsonl")


@cache
def _get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    try:
        return tiktoken.get_encoding(encoding_name)
    except ValueError:
        logger.warning("Unknown token encoding {}; using cl100k_base", encoding_name)
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Count source text literally, including strings that resemble special tokens."""
    return len(encoding.encode(text, disallowed_special=()))


def _count_message_tokens(messages: list[dict[str, Any]], encoding: tiktoken.Encoding) -> int:
    """Local text estimate, not provider billing or a vision-token accounting model."""
    return 3 + sum(
        3 + int("name" in message) + sum(_count_tokens(str(value), encoding) for value in message.values() if value)
        for message in messages
    )


def log_inference_metrics(metrics: InferenceMetrics) -> None:
    """Record totals even if the optional metrics file cannot be written."""
    update_aggregate_metrics(
        metrics.model_name,
        metrics.input_tokens,
        metrics.output_tokens,
        metrics.duration,
        metrics.success,
        metrics.queue_time,
        metrics.retry_count,
    )
    try:
        _metrics_log.parent.mkdir(parents=True, exist_ok=True)
        with _metrics_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(metrics)) + "\n")
    except OSError:
        logger.warning("Unable to write inference metrics log")


def update_aggregate_metrics(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    duration: float = 0.0,
    success: bool = True,
    queue_time: float = 0.0,
    retry_count: int = 0,
) -> None:
    _cost_data[model_name].update({
        "calls": 1,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "successes": int(success),
        "duration": duration,
        "retries": retry_count,
        "queue_time": queue_time,
    })


def get_performance_summary(model_name: str | None = None) -> dict[str, Any]:
    """Measured logical-request statistics; token counts are local estimates."""
    selected = {name: data for name, data in _cost_data.items() if model_name is None or name == model_name}
    totals = Counter()
    for data in selected.values():
        totals.update(data)
    denominator = max(1, totals["calls"])
    return {
        "model_name": model_name,
        "models": list(selected),
        "total_calls": totals["calls"],
        "total_input_tokens": totals["input_tokens"],
        "total_output_tokens": totals["output_tokens"],
        "success_rate": totals["successes"] / denominator,
        "avg_duration": totals["duration"] / denominator,
        "avg_request_size": totals["input_tokens"] / denominator,
        "avg_response_size": totals["output_tokens"] / denominator,
        "avg_retry_count": totals["retries"] / denominator,
        "avg_queue_time": totals["queue_time"] / denominator,
    }

import os
import csv
import atexit
import datetime
import collections
from typing import Any, Dict, List
from dataclasses import dataclass

import tiktoken
from loguru import logger


@dataclass
class InferenceMetrics:
    """Metrics for tracking inference requests."""

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


# Using defaultdict for easier accumulation
_cost_data = collections.defaultdict(
    lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "calls": 0,
        "successes": 0,
        "duration": 0.0,
        "retries": 0,
        "queue_time": 0.0,
    }
)
_individual_log_file = os.path.join("logs", "inference_cost_log_individual.csv")
_aggregate_log_file = os.path.join("logs", "inference_cost_log_aggregate.csv")


def _get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Gets a tiktoken encoding, defaulting to cl100k_base with fallback."""
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to get encoding '{encoding_name}'. Falling back to 'cl100k_base'. Error: {e}")
        return tiktoken.get_encoding("cl100k_base")


def _ensure_logs_dir():
    """Ensures the logs directory exists."""
    os.makedirs("logs", exist_ok=True)


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    """Counts tokens in a single string."""
    if not text:
        return 0
    try:
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0


def _count_message_tokens(messages: List[Dict[str, str]], encoding: tiktoken.Encoding) -> int:
    """Counts tokens in a list of messages, approximating OpenAI's format."""
    num_tokens = 0
    # Approximation based on OpenAI's cookbook: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    # This might not be perfectly accurate for all models/providers but is a reasonable estimate.
    tokens_per_message = 3
    tokens_per_name = 1

    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if value:
                num_tokens += _count_tokens(str(value), encoding)
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def _log_individual_call(model_name: str, input_tokens: int, output_tokens: int, tags: List[str], encoding_name: str):
    """Logs a single inference call's cost details."""
    try:
        _ensure_logs_dir()
        is_new_file = not os.path.exists(_individual_log_file)

        with open(_individual_log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header only if the file is completely new
            if is_new_file:
                writer.writerow(["timestamp", "model_name", "stage", "input_tokens", "output_tokens", "encoding_used"])

            stage = ";".join(tags) if tags else "unknown"
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            writer.writerow([timestamp, model_name, stage, input_tokens, output_tokens, encoding_name])
    except Exception as e:
        logger.error(f"Failed to write to individual cost log: {e}")


def _update_aggregate_cost(model_name: str, input_tokens: int, output_tokens: int):
    """Updates the global dictionary for aggregate costs."""
    try:
        _cost_data[model_name]["input_tokens"] += input_tokens
        _cost_data[model_name]["output_tokens"] += output_tokens
        _cost_data[model_name]["calls"] += 1
    except Exception as e:
        logger.error(f"Failed to update aggregate cost data: {e}")


def _write_aggregate_log():
    """Writes the aggregated cost data to a file at program exit."""
    try:
        if not _cost_data:
            return

        _ensure_logs_dir()
        with open(_aggregate_log_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model_name", "total_input_tokens", "total_output_tokens", "total_calls"])
            for model_name, data in sorted(_cost_data.items()):
                writer.writerow([model_name, data["input_tokens"], data["output_tokens"], data["calls"]])
    except Exception as e:
        # Silent fail during shutdown - logger/stdout may be closed
        pass


# Register the aggregate log function to run at exit
atexit.register(_write_aggregate_log)


def _categorize_error(error: Exception) -> str:
    """Categorize an error for tracking purposes."""
    error_type = type(error).__name__
    if "timeout" in str(error).lower() or "TimeoutError" in error_type:
        return "timeout"
    elif "rate_limit" in str(error).lower() or "RateLimitError" in error_type:
        return "rate_limit"
    elif "authentication" in str(error).lower() or "AuthenticationError" in error_type:
        return "auth_error"
    elif "connection" in str(error).lower() or "ConnectionError" in error_type:
        return "connection_error"
    else:
        return "other_error"


def log_inference_metrics(metrics: InferenceMetrics) -> None:
    """Log inference metrics to tracking system."""
    _log_individual_call(
        model_name=metrics.model_name,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        tags=[metrics.stage],
        encoding_name=metrics.encoding_name,
    )
    update_aggregate_metrics(
        metrics.model_name,
        metrics.input_tokens,
        metrics.output_tokens,
        metrics.duration,
        metrics.success,
        metrics.queue_time,
        metrics.retry_count,
    )


def get_performance_summary(model_name: str | None = None) -> Dict[str, Any]:
    """Measured logical-request statistics; token counts are local estimates."""
    records = (
        [_cost_data[model_name]] if model_name in _cost_data else ([] if model_name else list(_cost_data.values()))
    )
    totals = {
        key: sum(row[key] for row in records)
        for key in ("calls", "input_tokens", "output_tokens", "successes", "duration", "retries", "queue_time")
    }
    denominator = max(1, totals["calls"])
    return {
        "model_name": model_name,
        "models": [model_name] if model_name else list(_cost_data),
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


def update_aggregate_metrics(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    duration: float = 0.0,
    success: bool = True,
    queue_time: float = 0.0,
    retry_count: int = 0,
    error: Exception | None = None,
    concurrency_level: int = 1,
) -> None:
    """Update aggregate metrics for a model."""
    _update_aggregate_cost(model_name, input_tokens, output_tokens)
    data = _cost_data[model_name]
    data["successes"] += int(success)
    data["duration"] += duration
    data["retries"] += retry_count
    data["queue_time"] += queue_time

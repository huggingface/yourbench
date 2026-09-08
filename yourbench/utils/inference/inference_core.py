"""Bounded, ordered model execution with explicit failures and owned client lifetimes."""

import os
import time
import uuid
import asyncio
from typing import Any
from contextlib import AsyncExitStack
from dataclasses import field, dataclass

import httpx

from huggingface_hub import AsyncInferenceClient
from yourbench.utils.inference.inference_tracking import (
    InferenceMetrics,
    _count_tokens,
    _get_encoding,
    _count_message_tokens,
    log_inference_metrics,
)


GLOBAL_TIMEOUT = 300
MAX_BACKOFF_SECONDS = 30


class InferenceError(RuntimeError):
    """A logical model request failed; no fabricated empty output is returned."""


class InferenceConfigurationError(InferenceError):
    """Model selection or request configuration is invalid."""


@dataclass
class Model:
    model_name: str
    provider: str | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    bill_to: str | None = None
    max_concurrent_requests: int = 16
    encoding_name: str = "cl100k_base"
    extra_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("HF_TOKEN")
        if self.max_concurrent_requests < 1:
            raise InferenceConfigurationError("max_concurrent_requests must be positive")


@dataclass
class InferenceCall:
    messages: list[dict[str, str]]
    temperature: float | None = None
    tags: list[str] = field(default_factory=list)
    # Historical name: this is the total attempt limit, including the first attempt.
    max_retries: int = 3
    seed: int | None = None
    extra_parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 1 <= self.max_retries <= 10:
            raise InferenceConfigurationError("max_retries (total attempts) must be between 1 and 10")


def _load_models(base_config, step_name: str) -> list[Model]:
    configured = base_config.model_list
    names = [m.model_name for m in configured]
    if len(names) != len(set(names)):
        raise InferenceConfigurationError("Model names must be unique")
    selected = base_config.model_roles.get(step_name) or names[:1]
    if not selected or set(selected) - set(names):
        raise InferenceConfigurationError(f"No valid model assignment for stage '{step_name}'")
    return [
        Model(**{name: getattr(m, name) for name in Model.__dataclass_fields__})
        for m in configured
        if m.model_name in selected
    ]


def _new_client(model: Model) -> AsyncInferenceClient:
    return AsyncInferenceClient(
        base_url=model.base_url,
        api_key=model.api_key,
        provider=model.provider,
        bill_to=model.bill_to,
        timeout=GLOBAL_TIMEOUT,
    )


def _is_transient(error: Exception) -> bool:
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None) or getattr(error, "status", None)
    if status is not None:
        return status in {408, 425, 429, 500, 502, 503, 504}
    return isinstance(error, (TimeoutError, ConnectionError, httpx.TransportError)) or type(error).__name__ in {
        "ClientConnectionError",
        "ServerDisconnectedError",
        "ClientConnectorError",
        "InferenceTimeoutError",
    }


async def _get_response(
    model: Model,
    inference_call: InferenceCall,
    request_id: str | None = None,
    concurrency_level: int = 1,
    queue_start_time: float | None = None,
    *,
    client: AsyncInferenceClient | None = None,
) -> tuple[str, InferenceMetrics]:
    """Execute one attempt. The batch owns shared clients; standalone calls close theirs."""
    started = time.monotonic()
    encoding = _get_encoding(model.encoding_name)
    kwargs: dict[str, Any] = {"model": model.model_name, "messages": inference_call.messages}
    for name in ("temperature", "seed"):
        if (value := getattr(inference_call, name)) is not None:
            kwargs[name] = value
    extras = {**model.extra_parameters, **inference_call.extra_parameters}
    if extras:
        kwargs["extra_body"] = extras
    owns_client = client is None
    client = client if client is not None else _new_client(model)
    try:
        response = await client.chat_completion(**kwargs)
        choices = getattr(response, "choices", None)
        content = getattr(getattr(choices[0], "message", None), "content", None) if choices else None
        if not isinstance(content, str) or not content.strip():
            raise InferenceError(f"Model '{model.model_name}' returned no text")
        return content, InferenceMetrics(
            request_id=request_id or str(uuid.uuid4()),
            model_name=model.model_name,
            stage=";".join(inference_call.tags) or "unknown",
            input_tokens=_count_message_tokens(inference_call.messages, encoding),
            output_tokens=_count_tokens(content, encoding),
            duration=time.monotonic() - started,
            queue_time=max(0, started - queue_start_time) if queue_start_time is not None else 0,
            retry_count=0,
            success=True,
            concurrency_level=concurrency_level,
            temperature=inference_call.temperature,
            encoding_name=model.encoding_name,
        )
    finally:
        if owns_client:
            await client.close()


async def _retry_with_backoff(
    model: Model,
    inference_call: InferenceCall,
    semaphore: asyncio.Semaphore,
    concurrency_level: int,
    *,
    client: AsyncInferenceClient,
) -> str:
    started = time.monotonic()
    metrics = InferenceMetrics(
        request_id=str(uuid.uuid4()),
        model_name=model.model_name,
        stage=";".join(inference_call.tags) or "unknown",
        input_tokens=_count_message_tokens(inference_call.messages, _get_encoding(model.encoding_name)),
        output_tokens=0,
        duration=0,
        queue_time=0,
        retry_count=0,
        success=False,
        concurrency_level=concurrency_level,
        temperature=inference_call.temperature,
        encoding_name=model.encoding_name,
    )
    try:
        for attempt in range(inference_call.max_retries):
            metrics.retry_count = attempt
            queued = time.monotonic()
            try:
                async with semaphore:
                    metrics.queue_time += time.monotonic() - queued
                    output, response_metrics = await _get_response(
                        model, inference_call, metrics.request_id, concurrency_level, client=client
                    )
                metrics.output_tokens = response_metrics.output_tokens
                metrics.success = True
                return output
            except Exception as error:
                # Provider exception bodies can contain credentials or sensitive prompts.
                metrics.error_message = type(error).__name__
                if not _is_transient(error) or attempt + 1 == inference_call.max_retries:
                    raise InferenceError(
                        f"Model '{model.model_name}' request failed after {attempt + 1} attempt(s) "
                        f"({type(error).__name__}); check endpoint, credentials and provider availability"
                    ) from None
            await asyncio.sleep(min(2**attempt, MAX_BACKOFF_SECONDS))
    except asyncio.CancelledError:
        metrics.error_message = "CancelledError"
        raise
    finally:
        metrics.duration = time.monotonic() - started
        if metrics.success:
            metrics.error_message = None
        log_inference_metrics(metrics)


async def _run_inference_async_helper(
    models: list[Model], inference_calls: list[InferenceCall]
) -> dict[str, list[str]]:
    if not inference_calls:
        return {model.model_name: [] for model in models}
    async with AsyncExitStack() as stack:
        clients = []
        for model in models:
            client = _new_client(model)
            stack.push_async_callback(client.close)
            clients.append(client)
        tasks = []
        for model, client in zip(models, clients):
            semaphore = asyncio.Semaphore(model.max_concurrent_requests)
            for call in inference_calls:
                tasks.append(
                    asyncio.create_task(
                        _retry_with_backoff(model, call, semaphore, model.max_concurrent_requests, client=client)
                    )
                )
        try:
            results = await asyncio.gather(*tasks)
        finally:
            # gather does not cancel siblings after a request fails. Drain them before closing clients.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    count = len(inference_calls)
    return {model.model_name: results[i * count : (i + 1) * count] for i, model in enumerate(models)}


def run_inference(config, step_name: str, inference_calls: list[InferenceCall]) -> dict[str, list[str]]:
    """Return ordered text per model, or raise InferenceError. Never hide request failures."""
    models = _load_models(config, step_name)
    for call in inference_calls:
        if step_name not in call.tags:
            call.tags.append(step_name)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_inference_async_helper(models, inference_calls))
    raise InferenceConfigurationError("run_inference cannot run inside an active event loop")

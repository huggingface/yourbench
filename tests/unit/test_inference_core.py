import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from yourbench.utils.inference import inference_core as core
from yourbench.utils.inference import inference_tracking as tracking
from yourbench.utils.inference.inference_core import Model, InferenceCall, _get_response


class _DummyResponse:
    def __init__(self, content: str):
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message})()
        self.choices = [choice]


def test_get_response_merges_extra_parameters():
    created_clients = []

    class _DummyClient:
        def __init__(self, *_, **__):
            self.close = AsyncMock()
            self.latest_kwargs = None
            self.chat_completion = AsyncMock(side_effect=self._chat_completion)
            created_clients.append(self)

        async def _chat_completion(self, **kwargs):
            self.latest_kwargs = kwargs
            return _DummyResponse("ok")

    model = Model(
        model_name="openrouter/test",
        provider=None,
        base_url="https://example.com/v1",
        api_key="token",
        bill_to=None,
        max_concurrent_requests=2,
        encoding_name="cl100k_base",
        extra_parameters={"reasoning": {"effort": "medium"}},
    )
    call = InferenceCall(
        messages=[{"role": "user", "content": "Hello"}],
        temperature=None,
        tags=["unit"],
        extra_parameters={"metadata": {"trace": True}},
    )

    async def _run():
        with patch("yourbench.utils.inference.inference_core.AsyncInferenceClient", _DummyClient):
            return await _get_response(model, call)

    response_text, metrics = asyncio.run(_run())

    assert response_text == "ok"
    assert metrics.success is True
    assert metrics.model_name == "openrouter/test"

    assert created_clients, "Expected AsyncInferenceClient to be instantiated"
    sent_kwargs = created_clients[0].latest_kwargs
    assert sent_kwargs["extra_body"] == {
        "reasoning": {"effort": "medium"},
        "metadata": {"trace": True},
    }
    assert sent_kwargs["messages"] == call.messages


def _client(side_effect):
    return SimpleNamespace(chat_completion=AsyncMock(side_effect=side_effect), close=AsyncMock())


def test_retry_metrics_and_client_reuse(monkeypatch):
    client = _client([TimeoutError("secret"), _DummyResponse("one"), _DummyResponse("two")])
    created = []
    monkeypatch.setattr(core, "_new_client", lambda m: created.append(m) or client)
    monkeypatch.setattr(core.asyncio, "sleep", AsyncMock())
    emitted = []
    monkeypatch.setattr(core, "log_inference_metrics", emitted.append)
    calls = [InferenceCall(messages=[], seed=42), InferenceCall(messages=[])]
    result = asyncio.run(core._run_inference_async_helper([Model("test", max_concurrent_requests=1)], calls))
    assert result == {"test": ["one", "two"]}
    assert len(created) == 1
    client.close.assert_awaited_once()
    assert client.chat_completion.call_args_list[0].kwargs["seed"] == 42
    assert len(emitted) == 2
    assert sorted(m.retry_count for m in emitted) == [0, 1]
    assert all(m.success for m in emitted)


@pytest.mark.parametrize("error,attempts", [(ValueError("secret"), 1), (TimeoutError("secret"), 3)])
def test_failures_raise_without_sensitive_error_text(monkeypatch, error, attempts):
    client = _client(error)
    monkeypatch.setattr(core, "_new_client", lambda m: client)
    sleep = AsyncMock()
    monkeypatch.setattr(core.asyncio, "sleep", sleep)
    metrics = []
    monkeypatch.setattr(core, "log_inference_metrics", metrics.append)
    with pytest.raises(core.InferenceError) as raised:
        asyncio.run(core._run_inference_async_helper([Model("test")], [InferenceCall(messages=[])]))
    assert "secret" not in str(raised.value)
    assert client.chat_completion.await_count == attempts
    assert sleep.await_count == attempts - 1
    client.close.assert_awaited_once()
    assert len(metrics) == 1
    assert not metrics[0].success
    assert metrics[0].retry_count == attempts - 1


def test_failure_cancels_siblings_before_client_closes(monkeypatch):
    cancelled = []

    async def request(**kwargs):
        if kwargs["messages"][0]["content"] == "fail":
            await asyncio.sleep(0)
            raise ValueError("bad")
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(True)

    client = _client(request)

    async def close():
        assert cancelled == [True]

    client.close = AsyncMock(side_effect=close)
    monkeypatch.setattr(core, "_new_client", lambda m: client)
    monkeypatch.setattr(core, "log_inference_metrics", lambda m: None)
    with pytest.raises(core.InferenceError):
        asyncio.run(
            core._run_inference_async_helper(
                [Model("test")],
                [InferenceCall(messages=[{"role": "user", "content": content}]) for content in ("wait", "fail")],
            )
        )
    client.close.assert_awaited_once()


def test_external_cancellation_closes_client(monkeypatch):
    async def scenario():
        event = asyncio.Event()

        async def request(**kwargs):
            event.set()
            await asyncio.Event().wait()

        client = _client(request)
        monkeypatch.setattr(core, "_new_client", lambda m: client)
        monkeypatch.setattr(core, "log_inference_metrics", lambda m: None)
        task = asyncio.create_task(core._run_inference_async_helper([Model("test")], [InferenceCall(messages=[])]))
        await event.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        client.close.assert_awaited_once()

    asyncio.run(scenario())


def test_actual_aggregate_statistics(monkeypatch):
    monkeypatch.setattr(
        tracking,
        "_cost_data",
        __import__("collections").defaultdict(
            lambda: {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "successes": 0,
                "duration": 0.0,
                "retries": 0,
                "queue_time": 0.0,
            }
        ),
    )
    tracking.update_aggregate_metrics("test", 10, 20, duration=4, success=True, retry_count=2)
    tracking.update_aggregate_metrics("test", 10, 0, duration=2, success=False)
    result = tracking.get_performance_summary("test")
    assert result["total_calls"] == 2
    assert result["success_rate"] == 0.5
    assert result["avg_duration"] == 3
    assert result["avg_retry_count"] == 1
    assert result["total_input_tokens"] == 20


def test_no_model_fails_explicitly():
    with pytest.raises(core.InferenceConfigurationError):
        core.run_inference(SimpleNamespace(model_list=[], model_roles={}), "planning", [])


def test_empty_model_response_is_failure(monkeypatch):
    client = _client([_DummyResponse("")])
    monkeypatch.setattr(core, "_new_client", lambda m: client)
    monkeypatch.setattr(core, "log_inference_metrics", lambda m: None)
    with pytest.raises(core.InferenceError):
        asyncio.run(core._run_inference_async_helper([Model("test")], [InferenceCall(messages=[])]))
    assert client.chat_completion.await_count == 1


@pytest.mark.parametrize("status,expected", [(400, False), (401, False), (403, False), (429, True), (503, True)])
def test_http_status_controls_retries(status, expected):
    error = RuntimeError("private response")
    error.response = SimpleNamespace(status_code=status)
    assert core._is_transient(error) is expected


def test_batch_preserves_order_despite_completion_order(monkeypatch):
    async def scenario():
        later_finished = asyncio.Event()

        async def request(**kwargs):
            value = kwargs["messages"][0]["content"]
            if value == "first":
                await later_finished.wait()
            else:
                later_finished.set()
            return _DummyResponse(value)

        client = _client(request)
        monkeypatch.setattr(core, "_new_client", lambda m: client)
        monkeypatch.setattr(core, "log_inference_metrics", lambda m: None)
        result = await core._run_inference_async_helper(
            [Model("test")],
            [InferenceCall(messages=[{"role": "user", "content": value}]) for value in ("first", "second")],
        )
        assert result == {"test": ["first", "second"]}

    asyncio.run(scenario())

import pytest
from litellm import RateLimitError, ServiceUnavailableError

from agentic_rag.llm.circuit_breaker import llm_circuit_breaker
from agentic_rag.llm.gateway import (
    generate_chat_completion,
    generate_embeddings,
    stream_chat_completion,
)
from agentic_rag.monitoring.metrics import (
    LLM_PROVIDER_CIRCUIT_STATE,
    LLM_PROVIDER_FAILURE_COUNT,
    LLM_PROVIDER_RETRY_AFTER_SECONDS,
)
from agentic_rag.shared.schemas.auth import AuthContext, TokenType
from agentic_rag.shared.schemas.llm import (
    ChatCompletionRequest,
    EmbeddingRequest,
    LLMMessage,
    LLMStreamEventType,
)


class FakeMessage:
    content = " Grounded answer [1]. "


class FakeChoice:
    message = FakeMessage()


class FakeUsage:
    prompt_tokens = 42
    completion_tokens = 9


class FakeResponse:
    choices = [FakeChoice()]
    usage = FakeUsage()
    _hidden_params = {"response_cost": 0.002}


class FakeEmbeddingItem:
    def __init__(self, value: float = 0.1):
        self.embedding = [value] * 768


class FakeEmbeddingResponse:
    def __init__(self, count: int = 2):
        self.data = [FakeEmbeddingItem(0.1 + index) for index in range(count)]


class FakeStreamDelta:
    def __init__(self, content: str):
        self.content = content


class FakeStreamChoice:
    def __init__(self, content: str):
        self.delta = FakeStreamDelta(content)


class FakeStreamUsage:
    prompt_tokens = 12
    completion_tokens = 5


class FakeStreamChunk:
    def __init__(self, content: str, usage=None, cost: float | None = None):
        self.choices = [FakeStreamChoice(content)]
        self.usage = usage
        self._hidden_params = {}
        if cost is not None:
            self._hidden_params = {"response_cost": cost}


@pytest.fixture(autouse=True)
def clear_llm_circuit_breaker_state(monkeypatch):
    monkeypatch.setattr(
        "agentic_rag.llm.manager.settings.llm_provider_database_enabled",
        False,
    )
    llm_circuit_breaker.clear()
    yield
    llm_circuit_breaker.clear()


def test_generate_chat_completion_calls_litellm(monkeypatch) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "secret-key")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_temperature", 0.2)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_tokens", 500)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 600)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_timeout_seconds", 15)

    response = generate_chat_completion(
        ChatCompletionRequest(
            messages=[
                LLMMessage(role="system", content="Use context only."),
                LLMMessage(role="user", content="Question and context."),
            ],
            metadata={"task": "query_synthesis"},
        )
    )

    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 500
    assert captured["timeout"] == 15
    assert captured["api_key"] == "secret-key"
    assert "base_url" not in captured
    assert captured["messages"][0]["role"] == "system"
    assert response.text == "Grounded answer [1]."
    assert response.model == "test-model"
    assert response.provider == "litellm"
    assert response.input_tokens == 42
    assert response.output_tokens == 9
    assert response.cost_estimate == 0.002
    assert response.metadata == {"task": "query_synthesis"}


def test_generate_chat_completion_uses_ollama_base_url(monkeypatch) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "ollama/llama3.1")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr(
        "agentic_rag.llm.gateway.settings.ollama_base_url",
        "http://ollama:11434",
    )

    generate_chat_completion(
        ChatCompletionRequest(
            messages=[
                LLMMessage(role="user", content="Question and context."),
            ],
        )
    )

    assert captured["base_url"] == "http://ollama:11434"


def test_generate_chat_completion_uses_gemini_api_key(monkeypatch) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr(
        "agentic_rag.llm.gateway.settings.default_llm_model",
        "gemini/gemini-2.5-flash",
    )
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.gemini_api_key", "gemini-secret")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)

    response = generate_chat_completion(
        ChatCompletionRequest(
            messages=[
                LLMMessage(role="user", content="Question and context."),
            ],
        )
    )

    assert captured["model"] == "gemini/gemini-2.5-flash"
    assert captured["api_key"] == "gemini-secret"
    assert "base_url" not in captured
    assert response.model == "gemini/gemini-2.5-flash"


def test_generate_chat_completion_rejects_empty_response(monkeypatch) -> None:
    class EmptyResponse:
        choices: list[object] = []

    def fake_completion(**kwargs):
        return EmptyResponse()

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)

    try:
        generate_chat_completion(
            ChatCompletionRequest(
                messages=[
                    LLMMessage(role="user", content="Question and context."),
                ],
            )
        )
    except RuntimeError as exc:
        assert "choices" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_generate_chat_completion_rejects_input_over_budget(monkeypatch) -> None:
    def fake_completion(**kwargs):
        raise AssertionError("Provider should not be called for over-budget input.")

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_retries", 2)

    with pytest.raises(ValueError) as exc_info:
        generate_chat_completion(
            ChatCompletionRequest(
                messages=[
                    LLMMessage(role="user", content="x" * 1001),
                ],
            )
        )

    assert "input character budget" in str(exc_info.value)


def test_generate_chat_completion_rejects_output_over_budget(monkeypatch) -> None:
    def fake_completion(**kwargs):
        raise AssertionError("Provider should not be called for over-budget output.")

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 64000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 10)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_retries", 2)

    with pytest.raises(ValueError) as exc_info:
        generate_chat_completion(
            ChatCompletionRequest(
                messages=[
                    LLMMessage(role="user", content="Question and context."),
                ],
                max_tokens=11,
            )
        )

    assert "output token budget" in str(exc_info.value)


def test_generate_chat_completion_retries_transient_failure(monkeypatch) -> None:
    calls = []
    sleep_seconds: list[float] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ServiceUnavailableError(
                message="temporary provider failure",
                llm_provider="litellm",
                model="test-model",
            )
        return FakeResponse()

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.time.sleep", sleep_seconds.append)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.gemini_api_key", "gemini-secret")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_retries", 2)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_retry_backoff_seconds", 0.1)

    response = generate_chat_completion(
        ChatCompletionRequest(
            messages=[
                LLMMessage(role="user", content="Question and context."),
            ],
        )
    )

    assert len(calls) == 2
    assert sleep_seconds == [0.1]
    assert response.text == "Grounded answer [1]."


def test_generate_chat_completion_fails_after_retry_limit(monkeypatch) -> None:
    calls = []
    sleep_seconds: list[float] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise RateLimitError(
            message="temporary rate limit",
            llm_provider="litellm",
            model="test-model",
        )

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.time.sleep", sleep_seconds.append)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_retries", 2)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_retry_backoff_seconds", 0.1)

    with pytest.raises(RateLimitError):
        generate_chat_completion(
            ChatCompletionRequest(
                messages=[
                    LLMMessage(role="user", content="Question and context."),
                ],
            )
        )

    assert len(calls) == 3
    assert sleep_seconds == [0.1, 0.2]


def test_generate_chat_completion_opens_circuit_after_failures(monkeypatch) -> None:
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        raise RateLimitError(
            message="temporary rate limit",
            llm_provider="litellm",
            model="test-model",
        )

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.circuit_breaker.time.time", lambda: 1000.0)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_retries", 0)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_circuit_breaker_enabled", True)
    monkeypatch.setattr(
        "agentic_rag.llm.gateway.settings.llm_circuit_breaker_failure_threshold",
        2,
    )
    monkeypatch.setattr(
        "agentic_rag.llm.gateway.settings.llm_circuit_breaker_cooldown_seconds",
        60,
    )

    request = ChatCompletionRequest(
        messages=[
            LLMMessage(role="user", content="Question and context."),
        ],
    )

    with pytest.raises(RateLimitError):
        generate_chat_completion(request)

    with pytest.raises(RateLimitError):
        generate_chat_completion(request)

    assert len(calls) == 2
    circuit_state = llm_circuit_breaker.get_state("litellm", "test-model")
    assert circuit_state
    assert circuit_state.failure_count == 2
    assert circuit_state.opened_until == 1060.0
    assert (
        LLM_PROVIDER_CIRCUIT_STATE.labels(
            provider="litellm",
            model="test-model",
            state="open",
        )._value.get()
        == 1.0
    )
    assert (
        LLM_PROVIDER_FAILURE_COUNT.labels(
            provider="litellm",
            model="test-model",
        )._value.get()
        == 2.0
    )
    assert (
        LLM_PROVIDER_RETRY_AFTER_SECONDS.labels(
            provider="litellm",
            model="test-model",
        )._value.get()
        == 60.0
    )


def test_generate_chat_completion_fails_fast_when_circuit_open(monkeypatch) -> None:
    def fake_completion(**kwargs):
        raise AssertionError("Provider should not be called while circuit is open.")

    monkeypatch.setattr("agentic_rag.llm.circuit_breaker.time.time", lambda: 1000.0)
    llm_circuit_breaker.record_failure(
        provider="litellm",
        model="test-model",
        error=RateLimitError(
            message="temporary rate limit",
            llm_provider="litellm",
            model="test-model",
        ),
        failure_threshold=1,
        cooldown_seconds=60,
    )

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_circuit_breaker_enabled", True)

    with pytest.raises(RuntimeError) as exc_info:
        generate_chat_completion(
            ChatCompletionRequest(
                messages=[
                    LLMMessage(role="user", content="Question and context."),
                ],
            )
        )

    assert "circuit breaker is open" in str(exc_info.value)


def test_generate_chat_completion_resets_circuit_after_cooldown_success(
    monkeypatch,
) -> None:
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    current_time = [1000.0]
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.time.time",
        lambda: current_time[0],
    )
    llm_circuit_breaker.record_failure(
        provider="litellm",
        model="test-model",
        error=RateLimitError(
            message="temporary rate limit",
            llm_provider="litellm",
            model="test-model",
        ),
        failure_threshold=1,
        cooldown_seconds=60,
    )
    current_time[0] = 1061.0

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_retries", 0)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_circuit_breaker_enabled", True)

    response = generate_chat_completion(
        ChatCompletionRequest(
            messages=[
                LLMMessage(role="user", content="Question and context."),
            ],
        )
    )

    assert len(calls) == 1
    assert response.text == "Grounded answer [1]."
    assert llm_circuit_breaker.get_state("litellm", "test-model") is None
    assert (
        LLM_PROVIDER_CIRCUIT_STATE.labels(
            provider="litellm",
            model="test-model",
            state="closed",
        )._value.get()
        == 1.0
    )
    assert (
        LLM_PROVIDER_FAILURE_COUNT.labels(
            provider="litellm",
            model="test-model",
        )._value.get()
        == 0.0
    )


def test_generate_chat_completion_bypasses_circuit_when_disabled(monkeypatch) -> None:
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr("agentic_rag.llm.circuit_breaker.time.time", lambda: 1000.0)
    llm_circuit_breaker.record_failure(
        provider="litellm",
        model="test-model",
        error=RateLimitError(
            message="temporary rate limit",
            llm_provider="litellm",
            model="test-model",
        ),
        failure_threshold=1,
        cooldown_seconds=60,
    )

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_circuit_breaker_enabled", False)

    response = generate_chat_completion(
        ChatCompletionRequest(
            messages=[
                LLMMessage(role="user", content="Question and context."),
            ],
        )
    )

    assert len(calls) == 1
    assert response.text == "Grounded answer [1]."


def test_stream_chat_completion_yields_tokens_and_completed_event(monkeypatch) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return [
            {"choices": [{"delta": {"content": "Grounded "}}]},
            {"choices": [{"delta": {"content": "answer [1]."}}]},
            FakeStreamChunk("", usage=FakeStreamUsage(), cost=0.004),
        ]

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "secret-key")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_temperature", 0.2)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_tokens", 500)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 600)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_timeout_seconds", 15)

    events = list(
        stream_chat_completion(
            ChatCompletionRequest(
                messages=[
                    LLMMessage(role="system", content="Use context only."),
                    LLMMessage(role="user", content="Question and context."),
                ],
                metadata={"task": "query_synthesis"},
            )
        )
    )

    assert captured["model"] == "test-model"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 500
    assert captured["timeout"] == 15
    assert captured["stream"] is True
    assert captured["api_key"] == "secret-key"
    assert events[0].event == LLMStreamEventType.TOKEN
    assert events[0].text_delta == "Grounded "
    assert events[1].event == LLMStreamEventType.TOKEN
    assert events[1].text_delta == "answer [1]."
    assert events[2].event == LLMStreamEventType.COMPLETED
    assert events[2].text == "Grounded answer [1]."
    assert events[2].model == "test-model"
    assert events[2].provider == "litellm"
    assert events[2].input_tokens == 12
    assert events[2].output_tokens == 5
    assert events[2].cost_estimate == 0.004
    assert events[2].metadata == {"task": "query_synthesis"}


def test_stream_chat_completion_retries_startup_failure(monkeypatch) -> None:
    calls = []
    sleep_seconds: list[float] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ServiceUnavailableError(
                message="temporary provider failure",
                llm_provider="litellm",
                model="test-model",
            )
        return [
            {"choices": [{"delta": {"content": "Grounded answer [1]."}}]},
        ]

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.time.sleep", sleep_seconds.append)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_retries", 2)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_retry_backoff_seconds", 0.1)

    events = list(
        stream_chat_completion(
            ChatCompletionRequest(
                messages=[
                    LLMMessage(role="user", content="Question and context."),
                ],
            )
        )
    )

    assert len(calls) == 2
    assert sleep_seconds == [0.1]
    assert events[-1].event == LLMStreamEventType.COMPLETED
    assert events[-1].text == "Grounded answer [1]."


def test_stream_chat_completion_records_stream_failure(monkeypatch) -> None:
    class FailingStream:
        def __iter__(self):
            yield {"choices": [{"delta": {"content": "Partial answer "}}]}
            raise RateLimitError(
                message="temporary rate limit",
                llm_provider="litellm",
                model="test-model",
            )

    def fake_completion(**kwargs):
        return FailingStream()

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.circuit_breaker.time.time", lambda: 1000.0)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.default_llm_model", "test-model")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_provider", "litellm")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_retries", 0)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_circuit_breaker_enabled", True)
    monkeypatch.setattr(
        "agentic_rag.llm.gateway.settings.llm_circuit_breaker_failure_threshold",
        1,
    )
    monkeypatch.setattr(
        "agentic_rag.llm.gateway.settings.llm_circuit_breaker_cooldown_seconds",
        60,
    )

    stream = stream_chat_completion(
        ChatCompletionRequest(
            messages=[
                LLMMessage(role="user", content="Question and context."),
            ],
        )
    )

    first_event = next(stream)
    assert first_event.event == LLMStreamEventType.TOKEN
    assert first_event.text_delta == "Partial answer "

    with pytest.raises(RateLimitError):
        next(stream)

    circuit_state = llm_circuit_breaker.get_state("litellm", "test-model")
    assert circuit_state
    assert circuit_state.failure_count == 1
    assert circuit_state.opened_until == 1060.0


def test_stream_chat_completion_rejects_input_over_budget(monkeypatch) -> None:
    def fake_completion(**kwargs):
        raise AssertionError("Provider should not be called for over-budget input.")

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_completion", fake_completion)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_output_tokens", 8000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_max_retries", 2)

    with pytest.raises(ValueError) as exc_info:
        list(
            stream_chat_completion(
                ChatCompletionRequest(
                    messages=[
                        LLMMessage(role="user", content="x" * 1001),
                    ],
                )
            )
        )

    assert "input character budget" in str(exc_info.value)


def test_generate_embeddings_calls_litellm(monkeypatch) -> None:
    captured = {}

    def fake_embedding(**kwargs):
        captured.update(kwargs)
        return FakeEmbeddingResponse(count=2)

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_embedding", fake_embedding)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.embedding_provider", "litellm")
    monkeypatch.setattr(
        "agentic_rag.llm.gateway.settings.embedding_model_name",
        "test-embedding-model",
    )
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.embedding_dimension", 768)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.embedding_timeout_seconds", 12)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.embedding_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "secret-key")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")

    response = generate_embeddings(
        EmbeddingRequest(
            auth=AuthContext(
                user_id="embedding-worker",
                tenant_id="tenant-a",
                token_type=TokenType.SERVICE,
            ),
            texts=["first chunk", "second chunk"],
        )
    )

    assert captured["model"] == "test-embedding-model"
    assert captured["input"] == ["first chunk", "second chunk"]
    assert captured["timeout"] == 12
    assert captured["api_key"] == "secret-key"
    assert "base_url" not in captured
    assert len(response.embeddings) == 2
    assert response.model == "test-embedding-model"
    assert response.provider == "litellm"
    assert response.dimension == 768


def test_generate_embeddings_sends_dimensions_for_gemini(monkeypatch) -> None:
    captured = {}

    def fake_embedding(**kwargs):
        captured.update(kwargs)
        return FakeEmbeddingResponse(count=1)

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_embedding", fake_embedding)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.embedding_provider", "litellm")
    monkeypatch.setattr(
        "agentic_rag.llm.gateway.settings.embedding_model_name",
        "gemini/gemini-embedding-001",
    )
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.embedding_dimension", 768)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.embedding_timeout_seconds", 12)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.embedding_max_input_chars", 1000)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.llm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_api_key", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.gemini_api_key", "gemini-secret")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.litellm_base_url", "")
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.ollama_base_url", "")

    response = generate_embeddings(
        EmbeddingRequest(
            auth=AuthContext(
                user_id="embedding-worker",
                tenant_id="tenant-a",
                token_type=TokenType.SERVICE,
            ),
            texts=["first chunk"],
        )
    )

    assert captured["model"] == "gemini/gemini-embedding-001"
    assert captured["dimensions"] == 768
    assert captured["api_key"] == "gemini-secret"
    assert response.dimension == 768


def test_generate_embeddings_rejects_input_over_budget(monkeypatch) -> None:
    def fake_embedding(**kwargs):
        raise AssertionError("Provider should not be called for over-budget input.")

    monkeypatch.setattr("agentic_rag.llm.gateway.litellm_embedding", fake_embedding)
    monkeypatch.setattr("agentic_rag.llm.gateway.settings.embedding_max_input_chars", 10)

    with pytest.raises(ValueError) as exc_info:
        generate_embeddings(
            EmbeddingRequest(
                auth=AuthContext(
                    user_id="embedding-worker",
                    tenant_id="tenant-a",
                    token_type=TokenType.SERVICE,
                ),
                texts=["x" * 11],
            )
        )

    assert "input character budget" in str(exc_info.value)

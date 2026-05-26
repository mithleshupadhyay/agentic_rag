import pytest
from litellm import RateLimitError, ServiceUnavailableError

from agentic_rag.llm.gateway import generate_chat_completion
from agentic_rag.shared.schemas.llm import ChatCompletionRequest, LLMMessage


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


def test_generate_chat_completion_rejects_empty_response(monkeypatch) -> None:
    class EmptyResponse:
        choices = []

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
    sleep_seconds = []

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
    sleep_seconds = []

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

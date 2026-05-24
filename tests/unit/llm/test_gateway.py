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

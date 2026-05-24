import logging
import time
from typing import Any

from litellm import completion as litellm_completion

from agentic_rag.shared.config import settings
from agentic_rag.shared.schemas.llm import ChatCompletionRequest, LLMResponse


logger = logging.getLogger(__name__)


def generate_chat_completion(request: ChatCompletionRequest) -> LLMResponse:
    model = request.model or settings.default_llm_model
    provider = request.provider or settings.llm_provider
    temperature = (
        request.temperature
        if request.temperature is not None
        else settings.llm_temperature
    )
    max_tokens = request.max_tokens or settings.llm_max_tokens
    timeout_seconds = request.timeout_seconds or settings.llm_timeout_seconds

    logger.info(
        f"[LLMGateway] Chat completion started provider={provider} "
        f"model={model} max_tokens={max_tokens} timeout_seconds={timeout_seconds}"
    )
    started_at = time.perf_counter()

    completion_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            message.model_dump()
            for message in request.messages
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout_seconds,
    }

    api_key = settings.llm_api_key or settings.litellm_api_key
    if api_key:
        completion_kwargs["api_key"] = api_key

    if settings.litellm_base_url:
        completion_kwargs["base_url"] = settings.litellm_base_url
    elif model.startswith("ollama/") and settings.ollama_base_url:
        completion_kwargs["base_url"] = settings.ollama_base_url

    response = litellm_completion(**completion_kwargs)
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("LLM response did not include choices.")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if isinstance(message, dict):
        text = message.get("content") or ""
    else:
        text = getattr(message, "content", "") if message is not None else ""

    text = text.strip()
    if not text:
        raise RuntimeError("LLM response did not include answer text.")

    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        output_tokens = int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
    else:
        input_tokens = int(
            getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
        )
        output_tokens = int(
            getattr(usage, "completion_tokens", 0)
            or getattr(usage, "output_tokens", 0)
            or 0
        )

    hidden_params = getattr(response, "_hidden_params", {}) or {}
    cost_estimate = 0.0
    if isinstance(hidden_params, dict):
        cost_value = hidden_params.get("response_cost") or hidden_params.get("cost")
        if isinstance(cost_value, (int, float)):
            cost_estimate = float(cost_value)

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        f"[LLMGateway] Chat completion completed provider={provider} "
        f"model={model} input_tokens={input_tokens} output_tokens={output_tokens} "
        f"latency_ms={latency_ms}"
    )

    return LLMResponse(
        text=text,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_estimate=cost_estimate,
        latency_ms=latency_ms,
        metadata=request.metadata,
    )

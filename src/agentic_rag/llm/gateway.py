import logging
import time
from typing import Any

from litellm import (
    APIConnectionError,
    APIError,
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    completion as litellm_completion,
    embedding as litellm_embedding,
)

from agentic_rag.llm.circuit_breaker import (
    check_llm_circuit_breaker,
    record_llm_circuit_breaker_failure,
    reset_llm_circuit_breaker,
)
from agentic_rag.shared.config import settings
from agentic_rag.shared.schemas.llm import (
    ChatCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    LLMResponse,
)


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
    input_chars = 0
    for request_message in request.messages:
        input_chars += len(request_message.content)

    if input_chars > settings.llm_max_input_chars:
        logger.warning(
            f"[LLMGateway] Request rejected by input budget provider={provider} "
            f"model={model} input_chars={input_chars} "
            f"max_input_chars={settings.llm_max_input_chars}"
        )
        raise ValueError(
            "LLM request exceeds input character budget "
            f"({input_chars}>{settings.llm_max_input_chars})."
        )

    if max_tokens > settings.llm_max_output_tokens:
        logger.warning(
            f"[LLMGateway] Request rejected by output budget provider={provider} "
            f"model={model} max_tokens={max_tokens} "
            f"max_output_tokens={settings.llm_max_output_tokens}"
        )
        raise ValueError(
            "LLM request exceeds output token budget "
            f"({max_tokens}>{settings.llm_max_output_tokens})."
        )

    if settings.llm_circuit_breaker_enabled:
        check_llm_circuit_breaker(provider, model)

    logger.info(
        f"[LLMGateway] Chat completion started provider={provider} "
        f"model={model} input_chars={input_chars} max_tokens={max_tokens} "
        f"timeout_seconds={timeout_seconds}"
    )
    started_at = time.perf_counter()

    completion_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            request_message.model_dump()
            for request_message in request.messages
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

    response: Any | None = None
    max_attempts = settings.llm_max_retries + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = litellm_completion(**completion_kwargs)
            break
        except (
            APIConnectionError,
            APIError,
            BadGatewayError,
            InternalServerError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout,
        ) as e:
            if attempt >= max_attempts:
                if settings.llm_circuit_breaker_enabled:
                    record_llm_circuit_breaker_failure(
                        provider=provider,
                        model=model,
                        error=e,
                        failure_threshold=(
                            settings.llm_circuit_breaker_failure_threshold
                        ),
                        cooldown_seconds=(
                            settings.llm_circuit_breaker_cooldown_seconds
                        ),
                    )

                logger.exception(
                    f"[LLMGateway] Chat completion failed after retries "
                    f"provider={provider} model={model} attempts={attempt} "
                    f"error_type={type(e).__name__}"
                )
                raise

            retry_after_seconds = settings.llm_retry_backoff_seconds * (
                2 ** (attempt - 1)
            )
            logger.warning(
                f"[LLMGateway] Chat completion retry scheduled "
                f"provider={provider} model={model} attempt={attempt} "
                f"max_attempts={max_attempts} "
                f"retry_after_seconds={retry_after_seconds:.2f} "
                f"error_type={type(e).__name__}"
            )
            if retry_after_seconds > 0:
                time.sleep(retry_after_seconds)

    if response is None:
        raise RuntimeError("LLM response was not returned.")

    if settings.llm_circuit_breaker_enabled:
        reset_llm_circuit_breaker(provider, model)

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("LLM response did not include choices.")

    first_choice = choices[0]
    response_message = getattr(first_choice, "message", None)
    if isinstance(response_message, dict):
        text = response_message.get("content") or ""
    else:
        text = (
            getattr(response_message, "content", "")
            if response_message is not None
            else ""
        )

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


def generate_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    model = request.model or settings.embedding_model_name
    provider = request.provider or settings.embedding_provider
    timeout_seconds = request.timeout_seconds or settings.embedding_timeout_seconds
    input_chars = sum(len(text) for text in request.texts)

    if input_chars > settings.embedding_max_input_chars:
        logger.warning(
            f"[LLMGateway] Embedding request rejected by input budget "
            f"tenant={request.auth.tenant_id} provider={provider} model={model} "
            f"input_chars={input_chars} "
            f"max_input_chars={settings.embedding_max_input_chars}"
        )
        raise ValueError(
            "Embedding request exceeds input character budget "
            f"({input_chars}>{settings.embedding_max_input_chars})."
        )

    if settings.llm_circuit_breaker_enabled:
        check_llm_circuit_breaker(provider, model)

    logger.info(
        f"[LLMGateway] Embedding request started tenant={request.auth.tenant_id} "
        f"provider={provider} model={model} texts={len(request.texts)} "
        f"input_chars={input_chars} timeout_seconds={timeout_seconds}"
    )
    started_at = time.perf_counter()

    embedding_kwargs: dict[str, Any] = {
        "model": model,
        "input": request.texts,
        "timeout": timeout_seconds,
    }
    if model.startswith("gemini/"):
        embedding_kwargs["dimensions"] = settings.embedding_dimension

    api_key = settings.llm_api_key or settings.litellm_api_key
    if api_key:
        embedding_kwargs["api_key"] = api_key

    if settings.litellm_base_url:
        embedding_kwargs["base_url"] = settings.litellm_base_url
    elif model.startswith("ollama/") and settings.ollama_base_url:
        embedding_kwargs["base_url"] = settings.ollama_base_url

    response: Any | None = None
    max_attempts = settings.llm_max_retries + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = litellm_embedding(**embedding_kwargs)
            break
        except (
            APIConnectionError,
            APIError,
            BadGatewayError,
            InternalServerError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout,
        ) as e:
            if attempt >= max_attempts:
                if settings.llm_circuit_breaker_enabled:
                    record_llm_circuit_breaker_failure(
                        provider=provider,
                        model=model,
                        error=e,
                        failure_threshold=(
                            settings.llm_circuit_breaker_failure_threshold
                        ),
                        cooldown_seconds=(
                            settings.llm_circuit_breaker_cooldown_seconds
                        ),
                    )

                logger.exception(
                    f"[LLMGateway] Embedding request failed after retries "
                    f"tenant={request.auth.tenant_id} provider={provider} "
                    f"model={model} attempts={attempt} "
                    f"error_type={type(e).__name__}"
                )
                raise

            retry_after_seconds = settings.llm_retry_backoff_seconds * (
                2 ** (attempt - 1)
            )
            logger.warning(
                f"[LLMGateway] Embedding request retry scheduled "
                f"tenant={request.auth.tenant_id} provider={provider} "
                f"model={model} attempt={attempt} max_attempts={max_attempts} "
                f"retry_after_seconds={retry_after_seconds:.2f} "
                f"error_type={type(e).__name__}"
            )
            if retry_after_seconds > 0:
                time.sleep(retry_after_seconds)

    if response is None:
        raise RuntimeError("Embedding response was not returned.")

    if settings.llm_circuit_breaker_enabled:
        reset_llm_circuit_breaker(provider, model)

    response_data = None
    if isinstance(response, dict):
        response_data = response.get("data")
    else:
        response_data = getattr(response, "data", None)

    if not response_data:
        raise RuntimeError("Embedding response did not include data.")

    vectors: list[list[float]] = []
    for item in response_data:
        raw_vector = (
            item.get("embedding")
            if isinstance(item, dict)
            else getattr(item, "embedding", None)
        )
        if not raw_vector:
            raise RuntimeError("Embedding response included an empty vector.")
        vectors.append([float(value) for value in raw_vector])

    if len(vectors) != len(request.texts):
        raise RuntimeError(
            "Embedding response count did not match input text count "
            f"({len(vectors)}!={len(request.texts)})."
        )

    dimension = len(vectors[0])
    for vector in vectors:
        if len(vector) != dimension:
            raise RuntimeError("Embedding response included mixed vector dimensions.")

    if dimension != settings.embedding_dimension:
        logger.warning(
            f"[LLMGateway] Embedding dimension mismatch provider={provider} "
            f"model={model} expected={settings.embedding_dimension} actual={dimension}"
        )
        raise RuntimeError(
            "Embedding dimension does not match configured vector dimension "
            f"({dimension}!={settings.embedding_dimension})."
        )

    latency_ms = int((time.perf_counter() - started_at) * 1000)
    logger.info(
        f"[LLMGateway] Embedding request completed tenant={request.auth.tenant_id} "
        f"provider={provider} model={model} texts={len(vectors)} "
        f"dimension={dimension} latency_ms={latency_ms}"
    )

    return EmbeddingResponse(
        embeddings=vectors,
        model=model,
        provider=provider,
        dimension=dimension,
        latency_ms=latency_ms,
    )

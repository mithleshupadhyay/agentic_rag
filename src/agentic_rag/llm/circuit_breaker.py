import logging
import time
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class LLMCircuitBreakerState:
    failure_count: int = 0
    opened_until: float = 0.0
    half_open: bool = False
    last_error_type: str | None = None
    last_failure_at: float | None = None


class LLMCircuitBreakerOpenError(RuntimeError):
    pass


_llm_circuit_breakers: dict[str, LLMCircuitBreakerState] = {}


def check_llm_circuit_breaker(provider: str, model: str) -> None:
    circuit_key = f"{provider}:{model}"
    circuit_state = _llm_circuit_breakers.get(circuit_key)
    if not circuit_state:
        return

    opened_until = circuit_state.opened_until
    current_time = time.time()
    if opened_until > current_time:
        retry_after_seconds = int(opened_until - current_time)
        logger.warning(
            f"[LLMCircuitBreaker] Request blocked by open circuit "
            f"provider={provider} model={model} "
            f"retry_after_seconds={retry_after_seconds}"
        )
        raise LLMCircuitBreakerOpenError(
            "LLM circuit breaker is open for "
            f"{provider}:{model}. Retry after {retry_after_seconds} seconds."
        )

    if opened_until > 0:
        circuit_state.opened_until = 0.0
        circuit_state.half_open = True
        logger.info(
            f"[LLMCircuitBreaker] Circuit breaker half-open "
            f"provider={provider} model={model}"
        )


def record_llm_circuit_breaker_failure(
    provider: str,
    model: str,
    error: Exception,
    failure_threshold: int,
    cooldown_seconds: int,
) -> None:
    circuit_key = f"{provider}:{model}"
    circuit_state = _llm_circuit_breakers.get(circuit_key) or LLMCircuitBreakerState()
    failure_count = circuit_state.failure_count + 1
    circuit_state.failure_count = failure_count
    circuit_state.last_error_type = type(error).__name__
    circuit_state.last_failure_at = time.time()

    if failure_count >= failure_threshold:
        circuit_state.opened_until = time.time() + cooldown_seconds
        circuit_state.half_open = False
        logger.warning(
            f"[LLMCircuitBreaker] Circuit breaker opened "
            f"provider={provider} model={model} "
            f"failure_count={failure_count} cooldown_seconds={cooldown_seconds} "
            f"error_type={type(error).__name__}"
        )
    else:
        circuit_state.opened_until = 0.0
        logger.warning(
            f"[LLMCircuitBreaker] Circuit breaker failure recorded "
            f"provider={provider} model={model} "
            f"failure_count={failure_count} threshold={failure_threshold} "
            f"error_type={type(error).__name__}"
        )

    _llm_circuit_breakers[circuit_key] = circuit_state


def reset_llm_circuit_breaker(provider: str, model: str) -> None:
    circuit_key = f"{provider}:{model}"
    if circuit_key not in _llm_circuit_breakers:
        return

    _llm_circuit_breakers.pop(circuit_key, None)
    logger.info(
        f"[LLMCircuitBreaker] Circuit breaker reset provider={provider} model={model}"
    )


def get_llm_circuit_breaker_state(
    provider: str,
    model: str,
) -> LLMCircuitBreakerState | None:
    return _llm_circuit_breakers.get(f"{provider}:{model}")


def clear_llm_circuit_breakers() -> None:
    _llm_circuit_breakers.clear()

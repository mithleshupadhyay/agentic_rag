import json
import logging
import time
from dataclasses import asdict, dataclass

from redis import Redis
from redis.exceptions import RedisError

from agentic_rag.shared.config import settings


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


class LLMCircuitBreaker:
    def __init__(self) -> None:
        # Mutable state
        self._states: dict[str, LLMCircuitBreakerState] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check(self, provider: str, model: str) -> None:
        circuit_key = f"{provider}:{model}"
        redis_key = f"{settings.llm_circuit_breaker_redis_key_prefix}:{circuit_key}"
        circuit_state = self._states.get(circuit_key)

        # Load shared Redis state when configured.
        if settings.llm_circuit_breaker_state_backend == "redis":
            try:
                redis_client = Redis.from_url(
                    settings.redis_url,
                    socket_timeout=settings.redis_socket_timeout_seconds,
                    socket_connect_timeout=settings.redis_socket_timeout_seconds,
                    decode_responses=True,
                )
                redis_client.ping()
                raw_state = redis_client.get(redis_key)

                if raw_state:
                    payload = json.loads(raw_state)
                    if isinstance(payload, dict):
                        circuit_state = LLMCircuitBreakerState(
                            failure_count=int(payload.get("failure_count") or 0),
                            opened_until=float(payload.get("opened_until") or 0.0),
                            half_open=bool(payload.get("half_open") or False),
                            last_error_type=payload.get("last_error_type"),
                            last_failure_at=(
                                float(payload["last_failure_at"])
                                if payload.get("last_failure_at") is not None
                                else None
                            ),
                        )
                    else:
                        redis_client.delete(redis_key)
                        circuit_state = None
                else:
                    circuit_state = None

            except (OSError, RedisError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    f"[LLMCircuitBreaker] Redis state unavailable; "
                    f"using memory fallback key={redis_key} "
                    f"error_type={type(e).__name__}"
                )
                circuit_state = self._states.get(circuit_key)

        if not circuit_state:
            return

        current_time = time.time()
        if circuit_state.opened_until > current_time:
            retry_after_seconds = int(circuit_state.opened_until - current_time)
            logger.warning(
                f"[LLMCircuitBreaker] Request blocked by open circuit "
                f"provider={provider} model={model} "
                f"retry_after_seconds={retry_after_seconds}"
            )
            raise LLMCircuitBreakerOpenError(
                "LLM circuit breaker is open for "
                f"{provider}:{model}. Retry after {retry_after_seconds} seconds."
            )

        # Move an expired open circuit to half-open for one recovery probe.
        if circuit_state.opened_until > 0:
            circuit_state.opened_until = 0.0
            circuit_state.half_open = True

            if settings.llm_circuit_breaker_state_backend == "redis":
                try:
                    redis_client = Redis.from_url(
                        settings.redis_url,
                        socket_timeout=settings.redis_socket_timeout_seconds,
                        socket_connect_timeout=settings.redis_socket_timeout_seconds,
                        decode_responses=True,
                    )
                    redis_client.ping()
                    redis_client.set(redis_key, json.dumps(asdict(circuit_state)))
                    self._states.pop(circuit_key, None)
                except (OSError, RedisError) as e:
                    self._states[circuit_key] = circuit_state
                    logger.warning(
                        f"[LLMCircuitBreaker] Redis state write failed; "
                        f"using memory fallback key={redis_key} "
                        f"error_type={type(e).__name__}"
                    )
            else:
                self._states[circuit_key] = circuit_state

            logger.info(
                f"[LLMCircuitBreaker] Circuit breaker half-open "
                f"provider={provider} model={model}"
            )

    def record_failure(
        self,
        provider: str,
        model: str,
        error: Exception,
        failure_threshold: int,
        cooldown_seconds: int,
    ) -> None:
        circuit_key = f"{provider}:{model}"
        redis_key = f"{settings.llm_circuit_breaker_redis_key_prefix}:{circuit_key}"
        circuit_state = self._states.get(circuit_key)

        # Read latest state before incrementing the consecutive failure count.
        if settings.llm_circuit_breaker_state_backend == "redis":
            try:
                redis_client = Redis.from_url(
                    settings.redis_url,
                    socket_timeout=settings.redis_socket_timeout_seconds,
                    socket_connect_timeout=settings.redis_socket_timeout_seconds,
                    decode_responses=True,
                )
                redis_client.ping()
                raw_state = redis_client.get(redis_key)

                if raw_state:
                    payload = json.loads(raw_state)
                    if isinstance(payload, dict):
                        circuit_state = LLMCircuitBreakerState(
                            failure_count=int(payload.get("failure_count") or 0),
                            opened_until=float(payload.get("opened_until") or 0.0),
                            half_open=bool(payload.get("half_open") or False),
                            last_error_type=payload.get("last_error_type"),
                            last_failure_at=(
                                float(payload["last_failure_at"])
                                if payload.get("last_failure_at") is not None
                                else None
                            ),
                        )
                    else:
                        redis_client.delete(redis_key)
                        circuit_state = None
                else:
                    circuit_state = None

            except (OSError, RedisError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    f"[LLMCircuitBreaker] Redis state unavailable; "
                    f"using memory fallback key={redis_key} "
                    f"error_type={type(e).__name__}"
                )
                circuit_state = self._states.get(circuit_key)

        circuit_state = circuit_state or LLMCircuitBreakerState()
        circuit_state.failure_count += 1
        circuit_state.last_error_type = type(error).__name__
        circuit_state.last_failure_at = time.time()

        if circuit_state.failure_count >= failure_threshold:
            circuit_state.opened_until = time.time() + cooldown_seconds
            circuit_state.half_open = False
            logger.warning(
                f"[LLMCircuitBreaker] Circuit breaker opened "
                f"provider={provider} model={model} "
                f"failure_count={circuit_state.failure_count} "
                f"cooldown_seconds={cooldown_seconds} "
                f"error_type={type(error).__name__}"
            )
        else:
            circuit_state.opened_until = 0.0
            logger.warning(
                f"[LLMCircuitBreaker] Circuit breaker failure recorded "
                f"provider={provider} model={model} "
                f"failure_count={circuit_state.failure_count} "
                f"threshold={failure_threshold} error_type={type(error).__name__}"
            )

        # Persist state, or keep it in memory if Redis is unavailable.
        if settings.llm_circuit_breaker_state_backend == "redis":
            try:
                redis_client = Redis.from_url(
                    settings.redis_url,
                    socket_timeout=settings.redis_socket_timeout_seconds,
                    socket_connect_timeout=settings.redis_socket_timeout_seconds,
                    decode_responses=True,
                )
                redis_client.ping()
                redis_client.set(redis_key, json.dumps(asdict(circuit_state)))
                self._states.pop(circuit_key, None)
                return
            except (OSError, RedisError) as e:
                logger.warning(
                    f"[LLMCircuitBreaker] Redis state write failed; "
                    f"using memory fallback key={redis_key} "
                    f"error_type={type(e).__name__}"
                )

        self._states[circuit_key] = circuit_state

    def reset(self, provider: str, model: str) -> None:
        circuit_key = f"{provider}:{model}"
        redis_key = f"{settings.llm_circuit_breaker_redis_key_prefix}:{circuit_key}"
        deleted = False

        # Clear shared Redis state first so other replicas stop blocking traffic.
        if settings.llm_circuit_breaker_state_backend == "redis":
            try:
                redis_client = Redis.from_url(
                    settings.redis_url,
                    socket_timeout=settings.redis_socket_timeout_seconds,
                    socket_connect_timeout=settings.redis_socket_timeout_seconds,
                    decode_responses=True,
                )
                redis_client.ping()
                deleted = bool(redis_client.delete(redis_key))
            except (OSError, RedisError) as e:
                logger.warning(
                    f"[LLMCircuitBreaker] Redis state delete failed "
                    f"key={redis_key} error_type={type(e).__name__}"
                )

        if circuit_key in self._states:
            self._states.pop(circuit_key, None)
            deleted = True

        if deleted:
            logger.info(
                f"[LLMCircuitBreaker] Circuit breaker reset "
                f"provider={provider} model={model}"
            )

    def get_state(self, provider: str, model: str) -> LLMCircuitBreakerState | None:
        circuit_key = f"{provider}:{model}"
        redis_key = f"{settings.llm_circuit_breaker_redis_key_prefix}:{circuit_key}"
        circuit_state = self._states.get(circuit_key)

        # Read Redis state when configured; otherwise return memory state.
        if settings.llm_circuit_breaker_state_backend == "redis":
            try:
                redis_client = Redis.from_url(
                    settings.redis_url,
                    socket_timeout=settings.redis_socket_timeout_seconds,
                    socket_connect_timeout=settings.redis_socket_timeout_seconds,
                    decode_responses=True,
                )
                redis_client.ping()
                raw_state = redis_client.get(redis_key)

                if not raw_state:
                    return None

                payload = json.loads(raw_state)
                if not isinstance(payload, dict):
                    redis_client.delete(redis_key)
                    return None

                return LLMCircuitBreakerState(
                    failure_count=int(payload.get("failure_count") or 0),
                    opened_until=float(payload.get("opened_until") or 0.0),
                    half_open=bool(payload.get("half_open") or False),
                    last_error_type=payload.get("last_error_type"),
                    last_failure_at=(
                        float(payload["last_failure_at"])
                        if payload.get("last_failure_at") is not None
                        else None
                    ),
                )
            except (OSError, RedisError, TypeError, ValueError, json.JSONDecodeError) as e:
                logger.warning(
                    f"[LLMCircuitBreaker] Redis state unavailable; "
                    f"using memory fallback key={redis_key} "
                    f"error_type={type(e).__name__}"
                )

        return circuit_state

    def clear(self) -> None:
        self._states.clear()

        # Clear Redis-backed state during controlled resets.
        if settings.llm_circuit_breaker_state_backend == "redis":
            redis_key_prefix = f"{settings.llm_circuit_breaker_redis_key_prefix}:"
            try:
                redis_client = Redis.from_url(
                    settings.redis_url,
                    socket_timeout=settings.redis_socket_timeout_seconds,
                    socket_connect_timeout=settings.redis_socket_timeout_seconds,
                    decode_responses=True,
                )
                redis_client.ping()
                redis_keys = list(redis_client.scan_iter(f"{redis_key_prefix}*"))
                if redis_keys:
                    redis_client.delete(*redis_keys)
            except (OSError, RedisError) as e:
                logger.warning(
                    f"[LLMCircuitBreaker] Redis state clear failed "
                    f"error_type={type(e).__name__}"
                )


llm_circuit_breaker = LLMCircuitBreaker()

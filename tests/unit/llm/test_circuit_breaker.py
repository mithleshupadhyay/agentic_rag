import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from agentic_rag.llm.circuit_breaker import (
    LLMCircuitBreakerOpenError,
    llm_circuit_breaker,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str) -> bool:
        self.values[key] = value
        return True

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.values:
                deleted += 1
                self.values.pop(key)
        return deleted

    def scan_iter(self, pattern: str):
        prefix = pattern.removesuffix("*")
        for key in list(self.values):
            if key.startswith(prefix):
                yield key


class FailingRedis(FakeRedis):
    def get(self, key: str) -> str | None:
        raise RedisConnectionError("redis unavailable")

    def set(self, key: str, value: str) -> bool:
        raise RedisConnectionError("redis unavailable")

    def delete(self, *keys: str) -> int:
        raise RedisConnectionError("redis unavailable")


@pytest.fixture(autouse=True)
def clear_state(monkeypatch):
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.settings.llm_circuit_breaker_state_backend",
        "memory",
    )
    llm_circuit_breaker.clear()
    yield
    llm_circuit_breaker.clear()


def test_redis_backed_circuit_breaker_opens_and_blocks(monkeypatch) -> None:
    fake_redis = FakeRedis()
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.settings.llm_circuit_breaker_state_backend",
        "redis",
    )
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.settings.llm_circuit_breaker_redis_key_prefix",
        "test:llm:circuit",
    )
    monkeypatch.setattr("agentic_rag.llm.circuit_breaker.time.time", lambda: 1000.0)
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.Redis.from_url",
        lambda *args, **kwargs: fake_redis,
    )

    llm_circuit_breaker.record_failure(
        provider="litellm",
        model="test-model",
        error=RuntimeError("first failure"),
        failure_threshold=2,
        cooldown_seconds=60,
    )
    llm_circuit_breaker.record_failure(
        provider="litellm",
        model="test-model",
        error=RuntimeError("second failure"),
        failure_threshold=2,
        cooldown_seconds=60,
    )

    circuit_state = llm_circuit_breaker.get_state("litellm", "test-model")

    assert circuit_state is not None
    assert circuit_state.failure_count == 2
    assert circuit_state.opened_until == 1060.0
    assert "test:llm:circuit:litellm:test-model" in fake_redis.values

    with pytest.raises(LLMCircuitBreakerOpenError):
        llm_circuit_breaker.check("litellm", "test-model")


def test_redis_backed_circuit_breaker_enters_half_open_after_cooldown(
    monkeypatch,
) -> None:
    fake_redis = FakeRedis()
    current_time = [1000.0]
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.settings.llm_circuit_breaker_state_backend",
        "redis",
    )
    monkeypatch.setattr("agentic_rag.llm.circuit_breaker.time.time", lambda: current_time[0])
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.Redis.from_url",
        lambda *args, **kwargs: fake_redis,
    )

    llm_circuit_breaker.record_failure(
        provider="litellm",
        model="test-model",
        error=RuntimeError("provider failure"),
        failure_threshold=1,
        cooldown_seconds=60,
    )
    current_time[0] = 1061.0

    llm_circuit_breaker.check("litellm", "test-model")
    circuit_state = llm_circuit_breaker.get_state("litellm", "test-model")

    assert circuit_state is not None
    assert circuit_state.opened_until == 0.0
    assert circuit_state.half_open is True


def test_redis_backed_circuit_breaker_reset_deletes_state(monkeypatch) -> None:
    fake_redis = FakeRedis()
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.settings.llm_circuit_breaker_state_backend",
        "redis",
    )
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.Redis.from_url",
        lambda *args, **kwargs: fake_redis,
    )

    llm_circuit_breaker.record_failure(
        provider="litellm",
        model="test-model",
        error=RuntimeError("provider failure"),
        failure_threshold=1,
        cooldown_seconds=60,
    )

    llm_circuit_breaker.reset("litellm", "test-model")

    assert llm_circuit_breaker.get_state("litellm", "test-model") is None
    assert fake_redis.values == {}


def test_redis_failure_uses_memory_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.settings.llm_circuit_breaker_state_backend",
        "redis",
    )
    monkeypatch.setattr("agentic_rag.llm.circuit_breaker.time.time", lambda: 1000.0)
    failing_redis = FailingRedis()
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.Redis.from_url",
        lambda *args, **kwargs: failing_redis,
    )

    llm_circuit_breaker.record_failure(
        provider="litellm",
        model="test-model",
        error=RuntimeError("provider failure"),
        failure_threshold=1,
        cooldown_seconds=60,
    )

    circuit_state = llm_circuit_breaker.get_state("litellm", "test-model")

    assert circuit_state is not None
    assert circuit_state.failure_count == 1
    assert circuit_state.opened_until == 1060.0
    with pytest.raises(LLMCircuitBreakerOpenError):
        llm_circuit_breaker.check("litellm", "test-model")

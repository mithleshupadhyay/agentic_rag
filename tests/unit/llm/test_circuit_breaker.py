import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from agentic_rag.llm.circuit_breaker import (
    LLMCircuitBreakerOpenError,
    llm_circuit_breaker,
)
from agentic_rag.monitoring.metrics import (
    LLM_PROVIDER_CIRCUIT_STATE,
    LLM_PROVIDER_FAILURE_COUNT,
    LLM_PROVIDER_RETRY_AFTER_SECONDS,
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


def test_memory_circuit_breaker_snapshot_defaults_to_closed(monkeypatch) -> None:
    monkeypatch.setattr("agentic_rag.llm.circuit_breaker.time.time", lambda: 1000.0)

    snapshot = llm_circuit_breaker.get_snapshot(
        provider="snapshot-provider",
        model="snapshot-model",
    )

    assert snapshot.provider == "snapshot-provider"
    assert snapshot.model == "snapshot-model"
    assert snapshot.state == "closed"
    assert snapshot.failure_count == 0
    assert snapshot.retry_after_seconds == 0
    assert snapshot.opened_until == 0.0
    assert snapshot.half_open is False
    assert snapshot.last_error_type is None
    assert snapshot.last_failure_at is None
    assert snapshot.checked_at == 1000.0


def test_memory_circuit_breaker_records_provider_health_metrics(monkeypatch) -> None:
    provider = "metric-provider"
    model = "metric-model"
    current_time = [1000.0]
    monkeypatch.setattr(
        "agentic_rag.llm.circuit_breaker.time.time",
        lambda: current_time[0],
    )

    llm_circuit_breaker.record_failure(
        provider=provider,
        model=model,
        error=RuntimeError("first failure"),
        failure_threshold=2,
        cooldown_seconds=60,
    )

    snapshot = llm_circuit_breaker.get_snapshot(provider, model)

    assert snapshot.state == "closed"
    assert snapshot.failure_count == 1
    assert snapshot.retry_after_seconds == 0
    assert snapshot.last_error_type == "RuntimeError"
    assert snapshot.last_failure_at == 1000.0
    assert (
        LLM_PROVIDER_CIRCUIT_STATE.labels(
            provider=provider,
            model=model,
            state="closed",
        )._value.get()
        == 1.0
    )
    assert (
        LLM_PROVIDER_CIRCUIT_STATE.labels(
            provider=provider,
            model=model,
            state="open",
        )._value.get()
        == 0.0
    )
    assert (
        LLM_PROVIDER_CIRCUIT_STATE.labels(
            provider=provider,
            model=model,
            state="half_open",
        )._value.get()
        == 0.0
    )
    assert (
        LLM_PROVIDER_FAILURE_COUNT.labels(
            provider=provider,
            model=model,
        )._value.get()
        == 1.0
    )
    assert (
        LLM_PROVIDER_RETRY_AFTER_SECONDS.labels(
            provider=provider,
            model=model,
        )._value.get()
        == 0.0
    )

    llm_circuit_breaker.record_failure(
        provider=provider,
        model=model,
        error=RuntimeError("second failure"),
        failure_threshold=2,
        cooldown_seconds=60,
    )

    snapshot = llm_circuit_breaker.get_snapshot(provider, model)

    assert snapshot.state == "open"
    assert snapshot.failure_count == 2
    assert snapshot.retry_after_seconds == 60
    assert snapshot.opened_until == 1060.0
    assert (
        LLM_PROVIDER_CIRCUIT_STATE.labels(
            provider=provider,
            model=model,
            state="closed",
        )._value.get()
        == 0.0
    )
    assert (
        LLM_PROVIDER_CIRCUIT_STATE.labels(
            provider=provider,
            model=model,
            state="open",
        )._value.get()
        == 1.0
    )
    assert (
        LLM_PROVIDER_FAILURE_COUNT.labels(
            provider=provider,
            model=model,
        )._value.get()
        == 2.0
    )
    assert (
        LLM_PROVIDER_RETRY_AFTER_SECONDS.labels(
            provider=provider,
            model=model,
        )._value.get()
        == 60.0
    )

    current_time[0] = 1061.0
    llm_circuit_breaker.check(provider, model)
    snapshot = llm_circuit_breaker.get_snapshot(provider, model)

    assert snapshot.state == "half_open"
    assert snapshot.retry_after_seconds == 0
    assert (
        LLM_PROVIDER_CIRCUIT_STATE.labels(
            provider=provider,
            model=model,
            state="half_open",
        )._value.get()
        == 1.0
    )

    llm_circuit_breaker.reset(provider, model)
    snapshot = llm_circuit_breaker.get_snapshot(provider, model)

    assert snapshot.state == "closed"
    assert snapshot.failure_count == 0
    assert (
        LLM_PROVIDER_CIRCUIT_STATE.labels(
            provider=provider,
            model=model,
            state="closed",
        )._value.get()
        == 1.0
    )
    assert (
        LLM_PROVIDER_FAILURE_COUNT.labels(
            provider=provider,
            model=model,
        )._value.get()
        == 0.0
    )


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

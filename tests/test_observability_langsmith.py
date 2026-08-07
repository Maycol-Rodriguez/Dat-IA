from collections.abc import Callable
from typing import Any

import pytest

from app.observability import langsmith as langsmith_observability


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_custom_tracing_flag_enables_langsmith(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("USE_LANGSMITH_TRACING", value)

    assert langsmith_observability.langsmith_tracing_enabled() is True


def test_langsmith_connection_status_requires_flag_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("USE_LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")

    assert (
        langsmith_observability.langsmith_connection_status()
        == "connected"
    )

    monkeypatch.delenv("LANGSMITH_API_KEY")

    assert (
        langsmith_observability.langsmith_connection_status()
        == "not_connected"
    )


def test_langsmith_client_is_optional_without_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("USE_LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    langsmith_observability.get_langsmith_client.cache_clear()

    try:
        assert langsmith_observability.get_langsmith_client() is None
        assert (
            langsmith_observability.langsmith_connection_status()
            == "not_connected"
        )
    finally:
        langsmith_observability.get_langsmith_client.cache_clear()


def test_langsmith_uses_shared_test_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING_SAMPLING_RATE", raising=False)

    # DEFAULT_LANGSMITH_PROJECT pasó a "dat_ia_prd" en
    # 3da238f ("cambio tag en langsmith y otros menores"); esta aserción
    # había quedado sin actualizar.
    assert langsmith_observability.langsmith_project_name() == "dat_ia_prd"
    assert langsmith_observability.langsmith_tracing_sampling_rate() == 1.0


def test_trace_inputs_summarize_database_rows() -> None:
    sanitized = langsmith_observability.sanitize_trace_inputs(
        {
            "question": "¿Cuál es el mejor transportista?",
            "rows": [
                {
                    "carrier_name": "DHL",
                    "on_time_rate": 0.97,
                }
            ],
        }
    )

    assert sanitized["question"] == "¿Cuál es el mejor transportista?"
    assert sanitized["rows_summary"] == {
        "row_count": 1,
        "columns": ["carrier_name", "on_time_rate"],
    }
    assert "DHL" not in str(sanitized)


def test_traceable_stage_passes_custom_flag_and_project_to_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    client = object()

    def fake_traceable(**kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        captured.update(kwargs)
        return lambda function: function

    monkeypatch.setenv("USE_LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "dat_ia_test")
    monkeypatch.setattr(langsmith_observability, "traceable", fake_traceable)
    monkeypatch.setattr(
        langsmith_observability,
        "get_langsmith_client",
        lambda: client,
    )

    langsmith_observability.traceable_stage(name="test-stage")

    assert captured["enabled"] is True
    assert captured["client"] is client
    assert captured["project_name"] == "dat_ia_test"


def test_traceable_stage_stays_disabled_without_configured_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_traceable(**kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        captured.update(kwargs)
        return lambda function: function

    monkeypatch.setenv("USE_LANGSMITH_TRACING", "true")
    monkeypatch.setattr(langsmith_observability, "traceable", fake_traceable)
    monkeypatch.setattr(
        langsmith_observability,
        "get_langsmith_client",
        lambda: None,
    )

    langsmith_observability.traceable_stage(name="test-stage")

    assert captured["enabled"] is False

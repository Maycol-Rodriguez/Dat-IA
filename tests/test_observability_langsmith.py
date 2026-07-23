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


def test_traceable_stage_passes_custom_flag_and_project_to_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    client = object()

    def fake_traceable(**kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        captured.update(kwargs)
        return lambda function: function

    monkeypatch.setenv("USE_LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "dat-ia-test")
    monkeypatch.setattr(langsmith_observability, "traceable", fake_traceable)
    monkeypatch.setattr(
        langsmith_observability,
        "get_langsmith_client",
        lambda: client,
    )

    langsmith_observability.traceable_stage(name="test-stage")

    assert captured["enabled"] is True
    assert captured["client"] is client
    assert captured["project_name"] == "dat-ia-test"


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

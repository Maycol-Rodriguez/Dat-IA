"""Base opcional y segura para integrar LangSmith en Dat-IA.

Este módulo no activa el tracing ni crea conexiones al importarse. LangSmith
solo se habilita cuando las variables de entorno se configuran de forma
explícita. Además, centraliza metadatos, etiquetas y sanitización para evitar
que credenciales o filas completas de la base de datos lleguen a las trazas.

La sanitización reduce el riesgo de exposición accidental, pero no sustituye
una política formal de clasificación de datos. Para información especialmente
sensible se debe desactivar el tracing de la petición o esconder por completo
sus entradas y salidas.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any

from langsmith import Client, traceable

REDACTED_VALUE = "[REDACTED]"
MAX_SANITIZE_DEPTH = 10

_TRUE_VALUES = {"1", "true", "yes", "on"}
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cf_api_key",
    "cloudflare_api_key",
    "cookie",
    "database_url",
    "db_url",
    "google_api_key",
    "langsmith_api_key",
    "password",
    "secret",
    "token",
}

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
_OLIST_ID_PATTERN = re.compile(r"\b[0-9a-fA-F]{32}\b")
_DATABASE_URL_PATTERN = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mariadb|sqlite)"
    r"(?:\+[A-Za-z0-9_.-]+)?://[^\s]+",
    flags=re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]+",
    flags=re.IGNORECASE,
)
_LANGSMITH_KEY_PATTERN = re.compile(r"\blsv2_[A-Za-z0-9_-]+\b")
_GOOGLE_KEY_PATTERN = re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")
_NAMED_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|password|secret|token)\s*[:=]\s*"
    r"([^\s,;]+)"
)
_TAG_CLEANUP_PATTERN = re.compile(r"[^a-z0-9_.:/-]+")


def langsmith_tracing_enabled() -> bool:
    """Indica si el tracing fue activado explícitamente en el entorno."""
    value = os.getenv("USE_LANGSMITH_TRACING", "")
    return value.strip().casefold() in _TRUE_VALUES


def langsmith_configuration_status() -> dict[str, Any]:
    """Expone el estado de configuración sin revelar credenciales."""
    api_key_configured = bool(os.getenv("LANGSMITH_API_KEY"))
    tracing_requested = langsmith_tracing_enabled()

    return {
        "enabled": tracing_requested,
        "ready": tracing_requested and api_key_configured,
        "api_key_configured": api_key_configured,
        "project": os.getenv("LANGSMITH_PROJECT") or None,
        "endpoint_configured": bool(os.getenv("LANGSMITH_ENDPOINT")),
        "workspace_configured": bool(os.getenv("LANGSMITH_WORKSPACE_ID")),
        "sampling_rate": os.getenv("LANGSMITH_TRACING_SAMPLING_RATE") or None,
    }


@lru_cache(maxsize=1)
def get_langsmith_client() -> Client | None:
    """Crea el cliente solo cuando tracing y API key están configurados.

    La función es deliberadamente perezosa: importar este módulo nunca debe
    conectar la aplicación con LangSmith ni impedir que Dat-IA arranque.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")

    if not langsmith_tracing_enabled() or not api_key:
        return None

    return Client(
        api_key=api_key,
        api_url=os.getenv("LANGSMITH_ENDPOINT") or None,
        workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID") or None,
        anonymizer=redact_trace_payload,
    )


def redact_trace_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Elimina secretos y PII básica de un payload antes de enviarlo."""
    sanitized = _sanitize_value(payload, depth=0)
    return sanitized if isinstance(sanitized, dict) else {"value": sanitized}


def sanitize_trace_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Procesador predeterminado para entradas de funciones trazadas."""
    return redact_trace_payload(inputs)


def sanitize_trace_outputs(output: Any) -> dict[str, Any]:
    """Procesa salidas sin registrar filas completas de PostgreSQL.

    Los campos ``data`` y ``rows`` se reemplazan por un resumen. En una tabla
    formateada se conservan columnas, cantidad de filas y locale, pero se
    descarta el contenido de sus filas.
    """
    output_mapping = _as_mapping(output)

    if output_mapping is None:
        return {"output": _sanitize_value(output, depth=0)}

    sanitized: dict[str, Any] = {}

    for raw_key, value in output_mapping.items():
        key = str(raw_key)
        normalized_key = key.casefold()

        if normalized_key in {"data", "rows"}:
            sanitized[f"{key}_summary"] = summarize_rows(value)
            continue

        if normalized_key == "table":
            sanitized[key] = _sanitize_result_table(value)
            continue

        sanitized[key] = _sanitize_value(value, depth=1, key=key)

    return sanitized


def summarize_rows(rows: Any) -> dict[str, Any]:
    """Resume un resultado tabular sin conservar valores de sus celdas."""
    if not _is_row_sequence(rows):
        return {"row_count": 0, "columns": []}

    columns: list[str] = []
    seen_columns: set[str] = set()

    for row in rows:
        row_mapping = _as_mapping(row)

        if row_mapping is None:
            continue

        for raw_column in row_mapping:
            column = str(raw_column)

            if column not in seen_columns:
                seen_columns.add(column)
                columns.append(column)

    return {
        "row_count": len(rows),
        "columns": columns,
    }


def build_trace_metadata(
    *,
    environment: str | None = None,
    app_version: str | None = None,
    endpoint: str | None = None,
    request_id: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    database_dialect: str | None = None,
    optimizer: str | None = None,
    intent: str | None = None,
    operation: str | None = None,
    retrieved_tables: Sequence[str] | None = None,
    retrieval_distances: Sequence[float] | None = None,
    ddl_threshold: float | None = None,
    memory_hit: bool | None = None,
    execution_status: str | None = None,
    row_count: int | None = None,
    error_type: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construye metadata consistente y filtrable para una traza."""
    metadata: dict[str, Any] = {
        "service": "dat-ia-api",
        "observability_schema": "1",
    }
    optional_values: dict[str, Any] = {
        "environment": environment,
        "app_version": app_version,
        "endpoint": endpoint,
        "request_id": request_id,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "database_dialect": database_dialect,
        "optimizer": optimizer,
        "intent": intent,
        "operation": operation,
        "retrieved_tables": list(retrieved_tables) if retrieved_tables else None,
        "retrieval_distances": (
            [float(distance) for distance in retrieval_distances]
            if retrieval_distances
            else None
        ),
        "ddl_threshold": ddl_threshold,
        "memory_hit": memory_hit,
        "execution_status": execution_status,
        "row_count": row_count,
        "error_type": error_type,
    }

    metadata.update(
        {key: value for key, value in optional_values.items() if value is not None}
    )

    if extra:
        metadata.update(extra)

    return redact_trace_payload(metadata)


def build_trace_tags(
    *,
    environment: str | None = None,
    endpoint: str | None = None,
    llm_provider: str | None = None,
    execution_status: str | None = None,
    extra: Sequence[str] | None = None,
) -> list[str]:
    """Crea etiquetas de baja cardinalidad para filtrar trazas."""
    tags = ["dat-ia"]
    dimensions = {
        "env": environment,
        "endpoint": endpoint,
        "provider": llm_provider,
        "status": execution_status,
    }

    for prefix, value in dimensions.items():
        if value:
            tags.append(f"{prefix}:{_normalize_tag(value)}")

    for tag in extra or []:
        normalized_tag = _normalize_tag(tag)

        if normalized_tag:
            tags.append(normalized_tag)

    return list(dict.fromkeys(tags))


def traceable_stage(
    *,
    name: str,
    run_type: str = "chain",
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    process_inputs: Callable[[dict[str, Any]], dict[str, Any]] = (
        sanitize_trace_inputs
    ),
    process_outputs: Callable[[Any], dict[str, Any]] = sanitize_trace_outputs,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Crea un decorador consistente para una etapa del pipeline.

    ``USE_LANGSMITH_TRACING`` controla explícitamente si el SDK crea la traza.
    El cliente se inyecta para aplicar la anonimización centralizada y el
    proyecto se obtiene del entorno para separar test de otros ambientes.
    """
    safe_metadata = redact_trace_payload(dict(metadata or {}))
    safe_tags = [_normalize_tag(tag) for tag in tags or [] if tag]
    client = get_langsmith_client()
    tracing_enabled = langsmith_tracing_enabled() and client is not None

    return traceable(
        name=name,
        run_type=run_type,
        client=client,
        project_name=os.getenv("LANGSMITH_PROJECT") or None,
        enabled=tracing_enabled,
        metadata=safe_metadata,
        tags=safe_tags,
        process_inputs=process_inputs,
        process_outputs=process_outputs,
        dangerously_allow_filesystem=False,
    )


def _sanitize_value(
    value: Any,
    *,
    depth: int,
    key: str | None = None,
) -> Any:
    if key is not None and _is_sensitive_key(key):
        return REDACTED_VALUE

    if depth >= MAX_SANITIZE_DEPTH:
        return "[MAX_DEPTH_REACHED]"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return _redact_text(value)

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    value_mapping = _as_mapping(value)

    if value_mapping is not None:
        return {
            str(child_key): _sanitize_value(
                child_value,
                depth=depth + 1,
                key=str(child_key),
            )
            for child_key, child_value in value_mapping.items()
        }

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_sanitize_value(item, depth=depth + 1) for item in value]

    if isinstance(value, set):
        return [
            _sanitize_value(item, depth=depth + 1) for item in sorted(value, key=str)
        ]

    return f"<{type(value).__name__}>"


def _redact_text(value: str) -> str:
    redacted = _DATABASE_URL_PATTERN.sub(REDACTED_VALUE, value)
    redacted = _BEARER_PATTERN.sub(f"Bearer {REDACTED_VALUE}", redacted)
    redacted = _LANGSMITH_KEY_PATTERN.sub(REDACTED_VALUE, redacted)
    redacted = _GOOGLE_KEY_PATTERN.sub(REDACTED_VALUE, redacted)
    redacted = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
    redacted = _UUID_PATTERN.sub("[REDACTED_UUID]", redacted)
    redacted = _OLIST_ID_PATTERN.sub("[REDACTED_ID]", redacted)
    return _NAMED_SECRET_PATTERN.sub(r"\1=" + REDACTED_VALUE, redacted)


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")

    return normalized in _SENSITIVE_KEYS or any(
        normalized.endswith(f"_{suffix}")
        for suffix in ("api_key", "password", "secret", "token")
    )


def _as_mapping(value: Any) -> Mapping[Any, Any] | None:
    if isinstance(value, Mapping):
        return value

    model_dump = getattr(value, "model_dump", None)

    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, Mapping) else None

    if is_dataclass(value) and not isinstance(value, type):
        dumped = asdict(value)
        return dumped if isinstance(dumped, Mapping) else None

    return None


def _is_row_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _sanitize_result_table(value: Any) -> Any:
    table = _as_mapping(value)

    if table is None:
        return _sanitize_value(value, depth=1)

    sanitized = {
        str(key): _sanitize_value(item, depth=2, key=str(key))
        for key, item in table.items()
        if str(key).casefold() != "rows"
    }

    if "rows" in table:
        sanitized["rows_summary"] = summarize_rows(table["rows"])

    return sanitized


def _normalize_tag(value: str) -> str:
    normalized = _TAG_CLEANUP_PATTERN.sub("-", value.strip().casefold())
    return normalized.strip("-")

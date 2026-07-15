"""Modelo de datos para la memoria semántica de consultas V2.

La memoria V2 separa el texto usado para embeddings de la metadata técnica.
El SQL no forma parte del documento vectorizado para evitar que domine la
similitud semántica entre preguntas.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

QUERY_MEMORY_V2_COLLECTION = "query_memory_v2"
QUERY_MEMORY_V2_EMBEDDING_VERSION = "query-memory-v2-question-structure"


@dataclass(frozen=True)
class QueryMemoryV2Record:
    """Representa una consulta almacenada en la memoria V2."""

    original_question: str
    normalized_question: str
    intent: str
    metrics: list[str]
    filters: list[dict[str, str]]
    date_range: dict[str, str] | None
    context: list[str]
    sql: str
    sources: str
    status: str
    validated: bool
    execution_status: str
    fingerprint: str
    usage_count: int
    created_at: str
    updated_at: str
    last_used_at: str
    model: str
    embedding_version: str = QUERY_MEMORY_V2_EMBEDDING_VERSION

    @property
    def memory_id(self) -> str:
        """Devuelve un ID determinístico para evitar duplicados."""
        return f"query-memory-v2-{self.fingerprint[:32]}"

    def to_metadata(self) -> dict[str, str | int | bool]:
        """Convierte el registro a metadata compatible con ChromaDB.

        ChromaDB acepta valores escalares en metadata. Las listas y
        estructuras se guardan como JSON canónico.
        """
        return {
            "original_question": self.original_question,
            "normalized_question": self.normalized_question,
            "intent": self.intent,
            "metrics_json": _canonical_json(self.metrics),
            "filters_json": _canonical_json(self.filters),
            "date_range_json": (
                _canonical_json(self.date_range)
                if self.date_range is not None
                else ""
            ),
            "context_json": _canonical_json(self.context),
            "sql": self.sql,
            "sources": self.sources,
            "status": self.status,
            "validated": self.validated,
            "execution_status": self.execution_status,
            "fingerprint": self.fingerprint,
            "usage_count": self.usage_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
            "model": self.model,
            "embedding_version": self.embedding_version,
        }


def create_query_memory_v2_record(
    *,
    original_question: str,
    normalized_question: str,
    intent: str,
    metrics: list[str] | None = None,
    filters: list[dict[str, str]] | None = None,
    date_range: dict[str, str] | None = None,
    context: list[str] | None = None,
    sql: str,
    sources: str = "",
    status: str = "success",
    validated: bool = False,
    execution_status: str = "not_executed",
    model: str = "",
    now: datetime | None = None,
) -> QueryMemoryV2Record:
    """Crea un registro V2 normalizado y con fingerprint determinístico."""
    cleaned_original_question = _clean_text(original_question)
    cleaned_normalized_question = _clean_text(normalized_question)

    if not cleaned_original_question:
        raise ValueError("La pregunta original no puede estar vacía.")

    if not cleaned_normalized_question:
        raise ValueError("La pregunta normalizada no puede estar vacía.")

    normalized_intent = _normalize_token(intent) or "detail"
    normalized_metrics = _normalize_text_list(metrics or [])
    normalized_filters = _normalize_filters(filters or [])
    normalized_date_range = _normalize_date_range(date_range)
    normalized_context = _normalize_text_list(context or [])

    fingerprint = build_query_memory_v2_fingerprint(
        normalized_question=cleaned_normalized_question,
        intent=normalized_intent,
        metrics=normalized_metrics,
        filters=normalized_filters,
        date_range=normalized_date_range,
    )

    current_time = now or datetime.now(timezone.utc)

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    timestamp = current_time.astimezone(timezone.utc).isoformat()

    return QueryMemoryV2Record(
        original_question=cleaned_original_question,
        normalized_question=cleaned_normalized_question,
        intent=normalized_intent,
        metrics=normalized_metrics,
        filters=normalized_filters,
        date_range=normalized_date_range,
        context=normalized_context,
        sql=sql.strip(),
        sources=sources.strip(),
        status=status.strip() or "unknown",
        validated=validated,
        execution_status=execution_status.strip() or "unknown",
        fingerprint=fingerprint,
        usage_count=1,
        created_at=timestamp,
        updated_at=timestamp,
        last_used_at="",
        model=model.strip(),
    )


def build_query_memory_v2_fingerprint(
    *,
    normalized_question: str,
    intent: str,
    metrics: list[str],
    filters: list[dict[str, str]],
    date_range: dict[str, str] | None,
) -> str:
    """Construye una huella estable para deduplicar consultas equivalentes."""
    payload = {
        "normalized_question": _normalize_for_fingerprint(
            normalized_question
        ),
        "intent": _normalize_token(intent),
        "metrics": _normalize_text_list(metrics),
        "filters": _normalize_filters(filters),
        "date_range": _normalize_date_range(date_range),
    }

    encoded_payload = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def build_query_memory_v2_document(record: QueryMemoryV2Record) -> str:
    """Construye el texto que será convertido en embedding.

    El SQL, modelo, estado de ejecución y notas técnicas permanecen en
    metadata. El embedding representa principalmente la intención de negocio.
    """
    metrics_text = ", ".join(record.metrics) or "ninguna"
    context_text = ", ".join(record.context) or "ninguno"
    filters_text = (
        _canonical_json(record.filters)
        if record.filters
        else "ninguno"
    )
    date_range_text = (
        _canonical_json(record.date_range)
        if record.date_range is not None
        else "ninguno"
    )

    return "\n".join(
        [
            f"Pregunta normalizada: {record.normalized_question}",
            f"Intención: {record.intent}",
            f"Métricas: {metrics_text}",
            f"Filtros: {filters_text}",
            f"Rango de fechas: {date_range_text}",
            f"Contexto: {context_text}",
        ]
    )


def parse_query_memory_v2_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Restaura listas y estructuras JSON desde metadata de ChromaDB."""
    return {
        **metadata,
        "metrics": _parse_json_value(
            metadata.get("metrics_json"),
            default=[],
        ),
        "filters": _parse_json_value(
            metadata.get("filters_json"),
            default=[],
        ),
        "date_range": _parse_json_value(
            metadata.get("date_range_json"),
            default=None,
        ),
        "context": _parse_json_value(
            metadata.get("context_json"),
            default=[],
        ),
    }


def _normalize_filters(
    filters: list[dict[str, str]],
) -> list[dict[str, str]]:
    normalized_filters = []

    for query_filter in filters:
        field = _normalize_token(str(query_filter.get("field") or ""))
        operator = _clean_text(str(query_filter.get("operator") or "="))
        value = _normalize_for_fingerprint(
            str(query_filter.get("value") or "")
        )

        if not field or not value:
            continue

        normalized_filters.append(
            {
                "field": field,
                "operator": operator,
                "value": value,
            }
        )

    return sorted(
        normalized_filters,
        key=lambda item: (
            item["field"],
            item["operator"],
            item["value"],
        ),
    )


def _normalize_date_range(
    date_range: dict[str, str] | None,
) -> dict[str, str] | None:
    if not date_range:
        return None

    start_date = _clean_text(str(date_range.get("start_date") or ""))
    end_date = _clean_text(str(date_range.get("end_date") or ""))

    if not start_date or not end_date:
        return None

    return {
        "start_date": start_date,
        "end_date": end_date,
    }


def _normalize_text_list(values: list[str]) -> list[str]:
    normalized_values = {
        _normalize_token(str(value))
        for value in values
        if _normalize_token(str(value))
    }

    return sorted(normalized_values)


def _normalize_token(value: str) -> str:
    normalized = _normalize_for_fingerprint(value)
    return normalized.replace(" ", "_")


def _normalize_for_fingerprint(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    lowered = without_accents.lower()
    without_punctuation = re.sub(r"[^\w\s-]", " ", lowered)
    return _clean_text(without_punctuation)


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _parse_json_value(value: Any, *, default: Any) -> Any:
    if not value:
        return default

    try:
        return json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return default
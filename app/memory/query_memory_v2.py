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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

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
    group_by: list[str]
    context: list[str]
    sql: str
    sources: str
    status: str
    validated: bool
    execution_status: str
    fingerprint: str
    usage_count: int
    retrieval_count: int
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
            "group_by_json": _canonical_json(self.group_by),
            "context_json": _canonical_json(self.context),
            "sql": self.sql,
            "sources": self.sources,
            "status": self.status,
            "validated": self.validated,
            "execution_status": self.execution_status,
            "fingerprint": self.fingerprint,
            "usage_count": self.usage_count,
            "retrieval_count": self.retrieval_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_used_at": self.last_used_at,
            "model": self.model,
            "embedding_version": self.embedding_version,
            "memory_id": self.memory_id,
        }


def create_query_memory_v2_record(
    *,
    original_question: str,
    normalized_question: str,
    intent: str,
    metrics: list[str] | None = None,
    filters: list[dict[str, str]] | None = None,
    date_range: dict[str, str] | None = None,
    group_by: list[str] | None = None,
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
    normalized_group_by = _normalize_text_list(group_by or [])
    normalized_context = _normalize_text_list(context or [])

    fingerprint = build_query_memory_v2_fingerprint(
        normalized_question=cleaned_normalized_question,
        intent=normalized_intent,
        metrics=normalized_metrics,
        filters=normalized_filters,
        date_range=normalized_date_range,
        group_by=normalized_group_by,
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
        group_by=normalized_group_by,
        context=normalized_context,
        sql=sql.strip(),
        sources=sources.strip(),
        status=status.strip() or "unknown",
        validated=validated,
        execution_status=execution_status.strip() or "unknown",
        fingerprint=fingerprint,
        usage_count=1,
        retrieval_count=0,
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
    group_by: list[str] | None = None,
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
        "group_by": _normalize_text_list(group_by or []),
    }

    encoded_payload = _canonical_json(payload).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def build_query_memory_v2_document(record: QueryMemoryV2Record) -> str:
    """Construye el texto que será convertido en embedding.

    El SQL, modelo, estado de ejecución y notas técnicas permanecen en
    metadata. El embedding representa principalmente la intención de negocio.
    """
    metrics_text = ", ".join(record.metrics) or "ninguna"
    group_by_text = ", ".join(record.group_by) or "ninguna"
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
            f"Agrupaciones: {group_by_text}",
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
        "group_by": _parse_json_value(
            metadata.get("group_by_json"),
            default=[],
        ),
        "context": _parse_json_value(
            metadata.get("context_json"),
            default=[],
        ),
    }



def get_or_create_query_memory_v2_collection(
    chroma_client: Any,
    embeddings: Embeddings,
) -> Chroma:
    """Obtiene o crea la colección independiente de memoria V2."""
    return Chroma(
        client=chroma_client,
        collection_name=QUERY_MEMORY_V2_COLLECTION,
        embedding_function=embeddings,
    )


def upsert_query_memory_v2(
    collection: Chroma,
    record: QueryMemoryV2Record,
    *,
    now: datetime | None = None,
) -> QueryMemoryV2Record:
    """Guarda o actualiza una memoria usando su ID determinístico.

    Si ya existe una memoria validada, una ejecución posterior no validada
    no puede degradar ni reemplazar el SQL que ya fue ejecutado con éxito.
    """
    existing_result = collection._collection.get(
        ids=[record.memory_id],
        include=["metadatas"],
    )
    existing_ids = existing_result.get("ids") or []
    existing_metadata: dict[str, Any] = {}

    if existing_ids:
        metadatas = existing_result.get("metadatas") or []
        if metadatas:
            existing_metadata = metadatas[0] or {}

    current_timestamp = _as_utc_iso(now)

    if existing_metadata:
        existing_validated = _metadata_bool(
            existing_metadata.get("validated"),
        )
        incoming_is_authoritative = record.validated or not existing_validated

        merged_record = replace(
            record,
            sql=(
                record.sql
                if incoming_is_authoritative
                else str(existing_metadata.get("sql") or "")
            ),
            sources=(
                record.sources
                if incoming_is_authoritative
                else str(existing_metadata.get("sources") or "")
            ),
            status=(
                record.status
                if incoming_is_authoritative
                else str(existing_metadata.get("status") or "success")
            ),
            validated=record.validated or existing_validated,
            execution_status=(
                record.execution_status
                if incoming_is_authoritative
                else str(
                    existing_metadata.get("execution_status")
                    or "success"
                )
            ),
            usage_count=int(
                existing_metadata.get("usage_count") or 1
            ) + 1,
            retrieval_count=int(
                existing_metadata.get("retrieval_count") or 0
            ),
            created_at=str(
                existing_metadata.get("created_at")
                or record.created_at
            ),
            updated_at=current_timestamp,
            last_used_at=str(
                existing_metadata.get("last_used_at") or ""
            ),
            model=(
                record.model
                if incoming_is_authoritative
                else str(existing_metadata.get("model") or "")
            ),
        )
    else:
        merged_record = replace(
            record,
            created_at=record.created_at or current_timestamp,
            updated_at=current_timestamp,
        )

    collection.add_texts(
        texts=[build_query_memory_v2_document(merged_record)],
        metadatas=[merged_record.to_metadata()],
        ids=[merged_record.memory_id],
    )

    return merged_record


def search_query_memory_v2(
    collection: Chroma,
    *,
    query: str,
    n_results: int = 3,
    distance_threshold: float = 0.7,
    validated_only: bool = True,
    intent: str | None = None,
    required_metrics: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Busca memorias similares aplicando controles de calidad.

    Recupera más candidatos de los solicitados para después aplicar:
    distancia máxima, validación, ejecución exitosa, intención, métricas
    requeridas y deduplicación por fingerprint.
    """
    cleaned_query = _clean_text(query)

    if not cleaned_query or n_results <= 0:
        return []

    total = collection._collection.count()

    if total == 0:
        return []

    candidate_count = min(
        total,
        max(n_results * 5, n_results),
    )
    raw_results = collection.similarity_search_with_score(
        cleaned_query,
        k=candidate_count,
    )

    normalized_intent = (
        _normalize_token(intent)
        if intent is not None
        else None
    )
    normalized_required_metrics = set(
        _normalize_text_list(required_metrics or [])
    )

    best_by_fingerprint: dict[str, dict[str, Any]] = {}

    for document, raw_distance in raw_results:
        distance = float(raw_distance)

        if distance > distance_threshold:
            continue

        metadata = parse_query_memory_v2_metadata(document.metadata)

        if (
            metadata.get("embedding_version")
            != QUERY_MEMORY_V2_EMBEDDING_VERSION
        ):
            continue

        if validated_only:
            if not _metadata_bool(metadata.get("validated")):
                continue

            if metadata.get("execution_status") != "success":
                continue

        if (
            normalized_intent is not None
            and metadata.get("intent") != normalized_intent
        ):
            continue

        memory_metrics = set(metadata.get("metrics") or [])

        if (
            normalized_required_metrics
            and not normalized_required_metrics.issubset(
                memory_metrics,
            )
        ):
            continue

        fingerprint = str(metadata.get("fingerprint") or "")

        if not fingerprint:
            continue

        candidate = {
            "document": document.page_content,
            "metadata": metadata,
            "distance": distance,
        }
        previous = best_by_fingerprint.get(fingerprint)

        if (
            previous is None
            or distance < float(previous["distance"])
        ):
            best_by_fingerprint[fingerprint] = candidate

    ranked_results = sorted(
        best_by_fingerprint.values(),
        key=lambda result: float(result["distance"]),
    )

    return ranked_results[:n_results]




def mark_query_memory_v2_results_used(
    collection: Chroma,
    results: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> int:
    """Registra qué memorias fueron recuperadas como ejemplos del RAG.

    Actualiza retrieval_count y last_used_at sin recalcular embeddings.
    Cada ID se actualiza como máximo una vez por llamada.
    """
    timestamp = _as_utc_iso(now)
    updated_count = 0
    processed_ids: set[str] = set()

    for result in results:
        metadata = result.get("metadata") or {}
        memory_id = str(metadata.get("memory_id") or "").strip()

        if not memory_id or memory_id in processed_ids:
            continue

        processed_ids.add(memory_id)

        stored = collection._collection.get(
            ids=[memory_id],
            include=["metadatas"],
        )
        stored_ids = stored.get("ids") or []

        if not stored_ids:
            continue

        stored_metadatas = stored.get("metadatas") or []

        if not stored_metadatas:
            continue

        raw_metadata = dict(stored_metadatas[0] or {})
        retrieval_count = int(
            raw_metadata.get("retrieval_count") or 0
        ) + 1

        raw_metadata["retrieval_count"] = retrieval_count
        raw_metadata["last_used_at"] = timestamp

        collection._collection.update(
            ids=[memory_id],
            metadatas=[raw_metadata],
        )

        metadata["retrieval_count"] = retrieval_count
        metadata["last_used_at"] = timestamp
        updated_count += 1

    return updated_count

def search_query_memory_v2_for_record(
    collection: Chroma,
    record: QueryMemoryV2Record,
    *,
    n_results: int = 3,
    distance_threshold: float = 0.7,
) -> list[dict[str, Any]]:
    """Busca memorias compatibles con toda la estructura de una consulta.

    Además de la similitud semántica, exige coincidencia de intención,
    métricas, filtros y rango de fechas. Esta es la función recomendada
    cuando la consulta ya fue procesada por el optimizador.
    """
    if n_results <= 0:
        return []

    candidate_limit = max(n_results * 5, n_results)

    candidates = search_query_memory_v2(
        collection,
        query=build_query_memory_v2_document(record),
        n_results=candidate_limit,
        distance_threshold=distance_threshold,
        validated_only=True,
        intent=record.intent,
        required_metrics=record.metrics,
    )

    compatible_results = []

    for candidate in candidates:
        metadata = candidate["metadata"]

        memory_metrics = _normalize_text_list(
            metadata.get("metrics") or [],
        )
        memory_filters = _normalize_filters(
            metadata.get("filters") or [],
        )
        memory_date_range = _normalize_date_range(
            metadata.get("date_range"),
        )
        memory_group_by = _normalize_text_list(
            metadata.get("group_by") or [],
        )
        memory_context = set(
            _normalize_text_list(metadata.get("context") or [])
        )

        if memory_metrics != record.metrics:
            continue

        if memory_filters != record.filters:
            continue

        if memory_date_range != record.date_range:
            continue

        if memory_group_by != record.group_by:
            continue

        if (
            record.context
            and memory_context
            and set(record.context).isdisjoint(memory_context)
        ):
            continue

        compatible_results.append(candidate)

    return compatible_results[:n_results]

def _as_utc_iso(now: datetime | None = None) -> str:
    current_time = now or datetime.now(timezone.utc)

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=timezone.utc)

    return current_time.astimezone(timezone.utc).isoformat()


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return bool(value)

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
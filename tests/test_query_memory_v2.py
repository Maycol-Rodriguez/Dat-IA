from datetime import datetime, timezone

import pytest

from app.memory.query_memory_v2 import (
    QUERY_MEMORY_V2_COLLECTION,
    QUERY_MEMORY_V2_EMBEDDING_VERSION,
    build_query_memory_v2_document,
    build_query_memory_v2_fingerprint,
    create_query_memory_v2_record,
    parse_query_memory_v2_metadata,
)


def test_query_memory_v2_constants() -> None:
    assert QUERY_MEMORY_V2_COLLECTION == "query_memory_v2"
    assert QUERY_MEMORY_V2_EMBEDDING_VERSION == (
        "query-memory-v2-question-structure"
    )


def test_fingerprint_is_stable_for_equivalent_queries() -> None:
    first = build_query_memory_v2_fingerprint(
        normalized_question=(
            "Listar transportistas por tasa de cumplimiento."
        ),
        intent="ranking",
        metrics=["on_time_rate", "order_count"],
        filters=[
            {"field": "state", "operator": "=", "value": "SP"},
            {
                "field": "order_status",
                "operator": "=",
                "value": "delivered",
            },
        ],
        date_range={
            "start_date": "2018-01-01",
            "end_date": "2018-12-31",
        },
    )

    second = build_query_memory_v2_fingerprint(
        normalized_question=(
            "  listar TRANSPORTISTAS por tasa de cumplimiento! "
        ),
        intent="RANKING",
        metrics=["order_count", "on_time_rate", "on_time_rate"],
        filters=[
            {
                "field": "order_status",
                "operator": "=",
                "value": "DELIVERED",
            },
            {"field": "state", "operator": "=", "value": "sp"},
        ],
        date_range={
            "end_date": "2018-12-31",
            "start_date": "2018-01-01",
        },
    )

    assert first == second


def test_fingerprint_changes_when_filter_changes() -> None:
    common_arguments = {
        "normalized_question": "Calcular ventas por estado.",
        "intent": "aggregation",
        "metrics": ["revenue"],
        "date_range": None,
    }

    fingerprint_sp = build_query_memory_v2_fingerprint(
        **common_arguments,
        filters=[
            {"field": "state", "operator": "=", "value": "SP"},
        ],
    )

    fingerprint_rj = build_query_memory_v2_fingerprint(
        **common_arguments,
        filters=[
            {"field": "state", "operator": "=", "value": "RJ"},
        ],
    )

    assert fingerprint_sp != fingerprint_rj


def test_document_contains_semantic_fields_but_not_sql() -> None:
    record = create_query_memory_v2_record(
        original_question="Que transportista cumple mejor?",
        normalized_question=(
            "Listar transportistas por mayor tasa de cumplimiento."
        ),
        intent="ranking",
        metrics=["on_time_rate"],
        filters=[],
        date_range=None,
        context=["logistica", "transportistas"],
        sql=(
            "SELECT carrier_name FROM carriers "
            "ORDER BY on_time_rate DESC LIMIT 1;"
        ),
        sources="carriers",
        validated=True,
        execution_status="success",
        model="gemini-test",
    )

    document = build_query_memory_v2_document(record)

    assert "Pregunta normalizada:" in document
    assert "Intención: ranking" in document
    assert "on_time_rate" in document
    assert "logistica" in document
    assert "SELECT carrier_name" not in document
    assert "carriers ORDER BY" not in document


def test_record_has_deterministic_id_and_serializable_metadata() -> None:
    fixed_time = datetime(
        2026,
        7,
        15,
        12,
        30,
        tzinfo=timezone.utc,
    )

    record = create_query_memory_v2_record(
        original_question="Ventas por mes durante 2018",
        normalized_question="Calcular ventas totales por mes en 2018.",
        intent="temporal_trend",
        metrics=["revenue"],
        filters=[],
        date_range={
            "start_date": "2018-01-01",
            "end_date": "2018-12-31",
        },
        context=["ventas"],
        sql="SELECT DATE_TRUNC('month', order_date) FROM orders;",
        sources="olist_orders_dataset",
        validated=False,
        execution_status="not_executed",
        model="gemini-test",
        now=fixed_time,
    )

    metadata = record.to_metadata()

    assert record.memory_id.startswith("query-memory-v2-")
    assert len(record.memory_id) == len("query-memory-v2-") + 32
    assert metadata["validated"] is False
    assert metadata["usage_count"] == 1
    assert metadata["created_at"] == "2026-07-15T12:30:00+00:00"
    assert metadata["embedding_version"] == (
        QUERY_MEMORY_V2_EMBEDDING_VERSION
    )
    assert isinstance(metadata["metrics_json"], str)
    assert isinstance(metadata["date_range_json"], str)


def test_metadata_can_be_parsed_back_to_structures() -> None:
    record = create_query_memory_v2_record(
        original_question="Ordenes canceladas con tarjeta en SP",
        normalized_question=(
            "Contar órdenes canceladas pagadas con tarjeta en SP."
        ),
        intent="count",
        metrics=["order_count"],
        filters=[
            {"field": "state", "operator": "=", "value": "SP"},
            {
                "field": "order_status",
                "operator": "=",
                "value": "canceled",
            },
        ],
        date_range=None,
        context=["ventas"],
        sql="SELECT COUNT(*) FROM olist_orders_dataset;",
        sources="olist_orders_dataset",
        model="gemini-test",
    )

    parsed = parse_query_memory_v2_metadata(record.to_metadata())

    assert parsed["metrics"] == ["order_count"]
    assert parsed["context"] == ["ventas"]
    assert parsed["date_range"] is None
    assert {
        "field": "state",
        "operator": "=",
        "value": "sp",
    } in parsed["filters"]


def test_record_rejects_empty_normalized_question() -> None:
    with pytest.raises(
        ValueError,
        match="pregunta normalizada no puede estar vacía",
    ):
        create_query_memory_v2_record(
            original_question="Una pregunta",
            normalized_question="   ",
            intent="detail",
            sql="SELECT 1;",
        )
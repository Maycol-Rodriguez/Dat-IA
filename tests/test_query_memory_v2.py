from datetime import datetime, timezone

import chromadb
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.memory.query_memory_v2 import (
    QUERY_MEMORY_V2_COLLECTION,
    QUERY_MEMORY_V2_EMBEDDING_VERSION,
    build_query_memory_v2_document,
    build_query_memory_v2_fingerprint,
    create_query_memory_v2_record,
    get_or_create_query_memory_v2_collection,
    parse_query_memory_v2_metadata,
    search_query_memory_v2,
    search_query_memory_v2_for_record,
    upsert_query_memory_v2,
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


@pytest.fixture
def memory_v2_collection():
    chroma_client = chromadb.EphemeralClient()

    try:
        chroma_client.delete_collection(
            QUERY_MEMORY_V2_COLLECTION,
        )
    except Exception:
        pass

    embeddings = DeterministicFakeEmbedding(size=64)
    collection = get_or_create_query_memory_v2_collection(
        chroma_client,
        embeddings,
    )

    yield collection

    try:
        chroma_client.delete_collection(
            QUERY_MEMORY_V2_COLLECTION,
        )
    except Exception:
        pass


def _create_test_record(
    *,
    normalized_question: str,
    filters: list[dict[str, str]] | None = None,
    validated: bool = True,
    execution_status: str = "success",
    sql: str = "SELECT 1;",
    intent: str = "ranking",
    metrics: list[str] | None = None,
    group_by: list[str] | None = None,
):
    return create_query_memory_v2_record(
        original_question=normalized_question,
        normalized_question=normalized_question,
        intent=intent,
        metrics=metrics or ["on_time_rate"],
        filters=filters or [],
        date_range=None,
        group_by=group_by or [],
        context=["logistica"],
        sql=sql,
        sources="carriers",
        validated=validated,
        execution_status=execution_status,
        model="test-model",
    )


def test_get_or_create_query_memory_v2_collection(
    memory_v2_collection,
) -> None:
    assert memory_v2_collection._collection.name == (
        QUERY_MEMORY_V2_COLLECTION
    )
    assert memory_v2_collection._collection.count() == 0


def test_upsert_deduplicates_and_increments_usage_count(
    memory_v2_collection,
) -> None:
    record = _create_test_record(
        normalized_question=(
            "Listar transportistas por mayor cumplimiento."
        ),
    )

    first = upsert_query_memory_v2(
        memory_v2_collection,
        record,
        now=datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc),
    )
    second = upsert_query_memory_v2(
        memory_v2_collection,
        record,
        now=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )

    stored = memory_v2_collection._collection.get(
        ids=[record.memory_id],
        include=["metadatas"],
    )
    metadata = stored["metadatas"][0]

    assert memory_v2_collection._collection.count() == 1
    assert first.memory_id == second.memory_id
    assert second.usage_count == 2
    assert metadata["usage_count"] == 2
    assert metadata["updated_at"] == "2026-07-15T11:00:00+00:00"


def test_unvalidated_upsert_does_not_degrade_validated_memory(
    memory_v2_collection,
) -> None:
    valid_record = _create_test_record(
        normalized_question="Contar pedidos entregados.",
        sql="SELECT COUNT(*) FROM orders WHERE status = 'delivered';",
    )
    invalid_record = _create_test_record(
        normalized_question="Contar pedidos entregados.",
        validated=False,
        execution_status="not_executed",
        sql="SELECT consulta_incorrecta;",
    )

    upsert_query_memory_v2(memory_v2_collection, valid_record)
    merged = upsert_query_memory_v2(
        memory_v2_collection,
        invalid_record,
    )

    assert merged.validated is True
    assert merged.execution_status == "success"
    assert merged.sql == valid_record.sql
    assert merged.usage_count == 2


def test_different_filters_create_different_memories(
    memory_v2_collection,
) -> None:
    question = "Calcular ventas por estado."

    sp_record = _create_test_record(
        normalized_question=question,
        filters=[
            {"field": "state", "operator": "=", "value": "SP"},
        ],
    )
    rj_record = _create_test_record(
        normalized_question=question,
        filters=[
            {"field": "state", "operator": "=", "value": "RJ"},
        ],
    )

    upsert_query_memory_v2(memory_v2_collection, sp_record)
    upsert_query_memory_v2(memory_v2_collection, rj_record)

    assert sp_record.memory_id != rj_record.memory_id
    assert memory_v2_collection._collection.count() == 2


class _FakeRawCollection:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _FakeSearchCollection:
    def __init__(
        self,
        results: list[tuple[Document, float]],
    ) -> None:
        self.results = results
        self._collection = _FakeRawCollection(len(results))

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        _ = query
        return self.results[:k]


def test_search_excludes_unvalidated_and_far_memories() -> None:
    unvalidated = _create_test_record(
        normalized_question="Consulta no validada.",
        validated=False,
        execution_status="not_executed",
    )
    valid = _create_test_record(
        normalized_question="Consulta validada.",
    )
    far = _create_test_record(
        normalized_question="Consulta lejana.",
        filters=[
            {"field": "state", "operator": "=", "value": "RJ"},
        ],
    )

    collection = _FakeSearchCollection(
        [
            (
                Document(
                    page_content=build_query_memory_v2_document(
                        unvalidated,
                    ),
                    metadata=unvalidated.to_metadata(),
                ),
                0.10,
            ),
            (
                Document(
                    page_content=build_query_memory_v2_document(valid),
                    metadata=valid.to_metadata(),
                ),
                0.20,
            ),
            (
                Document(
                    page_content=build_query_memory_v2_document(far),
                    metadata=far.to_metadata(),
                ),
                0.90,
            ),
        ]
    )

    results = search_query_memory_v2(
        collection,
        query="consulta",
        distance_threshold=0.70,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["fingerprint"] == (
        valid.fingerprint
    )
    assert results[0]["distance"] == 0.20


def test_search_can_filter_by_intent_and_required_metrics() -> None:
    ranking = _create_test_record(
        normalized_question="Ranking de transportistas.",
        intent="ranking",
        metrics=["on_time_rate"],
    )
    aggregation = _create_test_record(
        normalized_question="Ventas totales.",
        intent="aggregation",
        metrics=["revenue"],
        filters=[
            {"field": "state", "operator": "=", "value": "SP"},
        ],
    )

    collection = _FakeSearchCollection(
        [
            (
                Document(
                    page_content=build_query_memory_v2_document(ranking),
                    metadata=ranking.to_metadata(),
                ),
                0.10,
            ),
            (
                Document(
                    page_content=build_query_memory_v2_document(
                        aggregation,
                    ),
                    metadata=aggregation.to_metadata(),
                ),
                0.20,
            ),
        ]
    )

    results = search_query_memory_v2(
        collection,
        query="ventas",
        distance_threshold=0.70,
        intent="aggregation",
        required_metrics=["revenue"],
    )

    assert len(results) == 1
    assert results[0]["metadata"]["intent"] == "aggregation"
    assert results[0]["metadata"]["metrics"] == ["revenue"]



def test_fingerprint_changes_when_group_by_changes() -> None:
    common_arguments = {
        "normalized_question": "Calcular ventas totales.",
        "intent": "aggregation",
        "metrics": ["revenue"],
        "filters": [],
        "date_range": None,
    }

    monthly = build_query_memory_v2_fingerprint(
        **common_arguments,
        group_by=["month"],
    )
    by_state = build_query_memory_v2_fingerprint(
        **common_arguments,
        group_by=["state"],
    )

    assert monthly != by_state


def test_structured_search_excludes_different_group_by() -> None:
    monthly = _create_test_record(
        normalized_question="Calcular ventas totales.",
        intent="aggregation",
        metrics=["revenue"],
        group_by=["month"],
    )
    by_state = _create_test_record(
        normalized_question="Calcular ventas totales.",
        intent="aggregation",
        metrics=["revenue"],
        group_by=["state"],
    )
    query_record = _create_test_record(
        normalized_question="Calcular ventas totales.",
        intent="aggregation",
        metrics=["revenue"],
        group_by=["month"],
        validated=False,
        execution_status="not_executed",
    )

    collection = _FakeSearchCollection(
        [
            (
                Document(
                    page_content=build_query_memory_v2_document(
                        by_state,
                    ),
                    metadata=by_state.to_metadata(),
                ),
                0.05,
            ),
            (
                Document(
                    page_content=build_query_memory_v2_document(
                        monthly,
                    ),
                    metadata=monthly.to_metadata(),
                ),
                0.10,
            ),
        ]
    )

    results = search_query_memory_v2_for_record(
        collection,
        query_record,
        distance_threshold=0.70,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["fingerprint"] == (
        monthly.fingerprint
    )

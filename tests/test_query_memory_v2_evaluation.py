import json
import re
import unicodedata
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.memory.query_memory_v2 import (
    build_query_memory_v2_document,
    create_query_memory_v2_record,
    search_query_memory_v2_for_record,
)


BANK_PATH = (
    Path(__file__).parent
    / "evaluation"
    / "query_memory_cases.json"
)


def _load_bank() -> dict:
    return json.loads(BANK_PATH.read_text(encoding="utf-8-sig"))


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents.lower()).strip()


def _semantic_features(text: str) -> set[str]:
    normalized = _normalize_text(text)

    known_features = {
        "ranking",
        "count",
        "aggregation",
        "detail",
        "temporal_trend",
        "on_time_rate",
        "revenue",
        "order_count",
        "stock_qty",
        "reorder_point",
        "returns_count",
        "review_score",
        "resolution_time_hr",
        "price",
        "logistica",
        "transportistas",
        "ventas",
        "inventario",
        "devoluciones",
        "resenas",
        "soporte",
        "precios",
        "2017",
        "2018",
    }

    features = {
        feature
        for feature in known_features
        if feature in normalized
    }

    if re.search(r"\bsp\b", normalized):
        features.add("state_sp")

    if re.search(r"\brj\b", normalized):
        features.add("state_rj")

    if '"resolved","operator":"=","value":"false"' in normalized:
        features.add("resolved_false")

    return features or {"unknown"}


def _semantic_distance(first: str, second: str) -> float:
    first_features = _semantic_features(first)
    second_features = _semantic_features(second)

    intersection = first_features & second_features
    union = first_features | second_features

    return 1.0 - (len(intersection) / len(union))


class _FakeRawCollection:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self) -> int:
        return self._count


class _SemanticEvaluationCollection:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._collection = _FakeRawCollection(len(documents))

    def similarity_search_with_score(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        ranked = sorted(
            (
                (
                    document,
                    _semantic_distance(query, document.page_content),
                )
                for document in self.documents
            ),
            key=lambda result: result[1],
        )

        return ranked[:k]


def _record_from_memory(memory: dict):
    return create_query_memory_v2_record(
        original_question=memory["original_question"],
        normalized_question=memory["normalized_question"],
        intent=memory["intent"],
        metrics=memory["metrics"],
        filters=memory["filters"],
        date_range=memory["date_range"],
        context=memory["context"],
        sql=memory["sql"],
        sources=memory["sources"],
        validated=memory["validated"],
        execution_status=memory["execution_status"],
        model="evaluation-model",
    )


def _record_from_case(case: dict):
    return create_query_memory_v2_record(
        original_question=case["question"],
        normalized_question=case["normalized_question"],
        intent=case["intent"],
        metrics=case["metrics"],
        filters=case["filters"],
        date_range=case["date_range"],
        context=case["context"],
        sql="",
        sources="",
        validated=False,
        execution_status="not_executed",
        model="evaluation-query",
    )


@pytest.fixture(scope="module")
def evaluation_environment():
    bank = _load_bank()
    documents = []

    for memory in bank["memories"]:
        record = _record_from_memory(memory)
        metadata = record.to_metadata()
        metadata["evaluation_key"] = memory["key"]

        documents.append(
            Document(
                page_content=build_query_memory_v2_document(record),
                metadata=metadata,
            )
        )

    return bank, _SemanticEvaluationCollection(documents)


def test_query_memory_bank_has_unique_identifiers() -> None:
    bank = _load_bank()

    memory_keys = [memory["key"] for memory in bank["memories"]]
    case_ids = [case["id"] for case in bank["cases"]]

    assert len(memory_keys) == len(set(memory_keys))
    assert len(case_ids) == len(set(case_ids))
    assert len(bank["memories"]) >= 8
    assert len(bank["cases"]) >= 12


@pytest.mark.parametrize(
    "case",
    _load_bank()["cases"],
    ids=lambda case: case["id"],
)
def test_query_memory_evaluation_case(
    evaluation_environment,
    case: dict,
) -> None:
    _, collection = evaluation_environment
    query_record = _record_from_case(case)

    results = search_query_memory_v2_for_record(
        collection,
        query_record,
        n_results=3,
        distance_threshold=0.7,
    )

    returned_keys = [
        result["metadata"]["evaluation_key"]
        for result in results
    ]

    expected_memory = case["expected_memory"]

    if expected_memory is None:
        assert returned_keys == []
    else:
        assert returned_keys
        assert returned_keys[0] == expected_memory


def test_query_memory_bank_recall_and_false_positives(
    evaluation_environment,
) -> None:
    bank, collection = evaluation_environment

    positive_cases = [
        case
        for case in bank["cases"]
        if case["expected_memory"] is not None
    ]
    negative_cases = [
        case
        for case in bank["cases"]
        if case["expected_memory"] is None
    ]

    hits_at_1 = 0
    hits_at_3 = 0
    false_positives = 0

    for case in bank["cases"]:
        query_record = _record_from_case(case)
        results = search_query_memory_v2_for_record(
            collection,
            query_record,
            n_results=3,
            distance_threshold=0.7,
        )
        returned_keys = [
            result["metadata"]["evaluation_key"]
            for result in results
        ]
        expected_memory = case["expected_memory"]

        if expected_memory is None:
            if returned_keys:
                false_positives += 1
            continue

        if returned_keys and returned_keys[0] == expected_memory:
            hits_at_1 += 1

        if expected_memory in returned_keys[:3]:
            hits_at_3 += 1

    recall_at_1 = hits_at_1 / len(positive_cases)
    recall_at_3 = hits_at_3 / len(positive_cases)
    false_positive_rate = (
        false_positives / len(negative_cases)
    )

    assert recall_at_1 == 1.0
    assert recall_at_3 == 1.0
    assert false_positive_rate == 0.0

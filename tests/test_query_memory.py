import chromadb
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from app.memory.query_memory import (
    QUERY_MEMORY_COLLECTION,
    build_query_memory_document,
    get_or_create_query_memory_collection,
    save_query_memory,
    search_query_memory,
)


@pytest.fixture
def memory_collection():
    chroma_client = chromadb.EphemeralClient()
    embeddings = DeterministicFakeEmbedding(size=32)
    return get_or_create_query_memory_collection(chroma_client, embeddings)


def test_get_or_create_query_memory_collection(memory_collection) -> None:
    assert memory_collection._collection.name == QUERY_MEMORY_COLLECTION


def test_build_query_memory_document() -> None:
    document = build_query_memory_document(
        question="Que transportista tiene mayor cumplimiento?",
        sql="SELECT carrier_name FROM carriers;",
        sources="carriers",
        confidence_note="Usa on_time_rate.",
    )

    assert "Pregunta:" in document
    assert "SQL:" in document
    assert "carriers" in document
    assert "on_time_rate" in document


def test_search_query_memory_returns_empty_when_collection_is_empty(memory_collection) -> None:
    results = search_query_memory(memory_collection, query="algo")

    assert results == []


def test_save_and_search_query_memory(memory_collection) -> None:
    question = "Que transportista tiene mayor cumplimiento?"

    memory_id = save_query_memory(
        memory_collection,
        question=question,
        sql="SELECT carrier_name FROM carriers ORDER BY on_time_rate DESC LIMIT 1;",
        sources="carriers",
        confidence_note="La tabla carriers contiene on_time_rate.",
        status="success",
        model="gemini-test",
    )

    results = search_query_memory(memory_collection, query=question)

    assert memory_id.startswith("query-")
    assert len(results) == 1
    assert results[0]["metadata"]["question"] == question
    assert "on_time_rate" in results[0]["metadata"]["sql"]
    assert results[0]["metadata"]["sources"] == "carriers"

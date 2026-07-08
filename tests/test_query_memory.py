from typing import Any

from app.memory.query_memory import (
    QUERY_MEMORY_COLLECTION,
    build_query_memory_document,
    get_or_create_query_memory_collection,
    save_query_memory,
    search_query_memory,
)


class FakeCollection:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def count(self) -> int:
        return len(self.items)

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for item_id, document, embedding, metadata in zip(
            ids,
            documents,
            embeddings,
            metadatas,
        ):
            self.items[item_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, list[list[Any]]]:
        _ = query_embeddings
        _ = include

        selected_items = list(self.items.values())[:n_results]

        return {
            "documents": [[item["document"] for item in selected_items]],
            "metadatas": [[item["metadata"] for item in selected_items]],
            "distances": [[0.1 for _item in selected_items]],
        }


class FakeChromaClient:
    def __init__(self) -> None:
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(
        self,
        name: str,
        embedding_function: Any = None,
    ) -> FakeCollection:
        _ = embedding_function

        if name not in self.collections:
            self.collections[name] = FakeCollection()

        return self.collections[name]


def test_get_or_create_query_memory_collection() -> None:
    chroma_client = FakeChromaClient()

    collection = get_or_create_query_memory_collection(chroma_client)

    assert QUERY_MEMORY_COLLECTION in chroma_client.collections
    assert collection is chroma_client.collections[QUERY_MEMORY_COLLECTION]


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


def test_search_query_memory_returns_empty_when_collection_is_empty() -> None:
    collection = FakeCollection()

    results = search_query_memory(
        collection,
        embedding=[0.1, 0.2, 0.3],
    )

    assert results == []


def test_save_and_search_query_memory() -> None:
    collection = FakeCollection()

    memory_id = save_query_memory(
        collection,
        question="Que transportista tiene mayor cumplimiento?",
        sql="SELECT carrier_name FROM carriers ORDER BY on_time_rate DESC LIMIT 1;",
        embedding=[0.1, 0.2, 0.3],
        sources="carriers",
        confidence_note="La tabla carriers contiene on_time_rate.",
        status="success",
        model="gemini-test",
    )

    results = search_query_memory(
        collection,
        embedding=[0.1, 0.2, 0.3],
    )

    assert memory_id.startswith("query-")
    assert len(results) == 1
    assert results[0]["metadata"]["question"] == (
        "Que transportista tiene mayor cumplimiento?"
    )
    assert "on_time_rate" in results[0]["metadata"]["sql"]
    assert results[0]["metadata"]["sources"] == "carriers"
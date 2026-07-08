"""Memoria semántica de consultas para Dat-IA.

Esta memoria usa una colección separada de ChromaDB para almacenar
preguntas previas, SQL generado y metadata útil.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

QUERY_MEMORY_COLLECTION = "query_memory"


def get_or_create_query_memory_collection(chroma_client: Any) -> Any:
    """Obtiene o crea la colección ChromaDB para memoria de consultas."""
    return chroma_client.get_or_create_collection(
        QUERY_MEMORY_COLLECTION,
        embedding_function=None,
    )


def build_query_memory_document(
    *,
    question: str,
    sql: str,
    sources: str = "",
    confidence_note: str = "",
) -> str:
    """Construye el texto que será vectorizado para búsqueda semántica."""
    return "\n".join(
        [
            f"Pregunta: {question}",
            f"SQL: {sql}",
            f"Fuentes: {sources}",
            f"Nota de confianza: {confidence_note}",
        ]
    )


def save_query_memory(
    collection: Any,
    *,
    question: str,
    sql: str,
    embedding: list[float],
    sources: str = "",
    confidence_note: str = "",
    status: str = "success",
    model: str = "",
) -> str:
    """Guarda una consulta resuelta en la memoria vectorial."""
    memory_id = f"query-{uuid4()}"
    document = build_query_memory_document(
        question=question,
        sql=sql,
        sources=sources,
        confidence_note=confidence_note,
    )

    collection.upsert(
        ids=[memory_id],
        documents=[document],
        embeddings=[embedding],
        metadatas=[
            {
                "question": question,
                "sql": sql,
                "sources": sources,
                "confidence_note": confidence_note,
                "status": status,
                "model": model,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )

    return memory_id


def search_query_memory(
    collection: Any,
    *,
    embedding: list[float],
    n_results: int = 3,
) -> list[dict[str, Any]]:
    """Busca consultas previas similares en la memoria vectorial."""
    total = collection.count()

    if total == 0:
        return []

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, total),
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "document": document,
            "metadata": metadata,
            "distance": distance,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]
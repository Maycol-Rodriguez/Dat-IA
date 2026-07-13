"""Memoria semántica de consultas para Dat-IA.

Esta memoria usa una colección separada de ChromaDB (vía langchain_chroma)
para almacenar preguntas previas, SQL generado y metadata útil.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

QUERY_MEMORY_COLLECTION = "query_memory"


def get_or_create_query_memory_collection(chroma_client: Any, embeddings: Embeddings) -> Chroma:
    """Obtiene o crea el VectorStore de LangChain para memoria de consultas."""
    return Chroma(
        client=chroma_client,
        collection_name=QUERY_MEMORY_COLLECTION,
        embedding_function=embeddings,
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
    collection: Chroma,
    *,
    question: str,
    sql: str,
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

    collection.add_texts(
        texts=[document],
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
        ids=[memory_id],
    )

    return memory_id


def search_query_memory(
    collection: Chroma,
    *,
    query: str,
    n_results: int = 3,
) -> list[dict[str, Any]]:
    """Busca consultas previas similares en la memoria vectorial."""
    total = collection._collection.count()

    if total == 0:
        return []

    results = collection.similarity_search_with_score(query, k=min(n_results, total))

    return [
        {"document": doc.page_content, "metadata": doc.metadata, "distance": distance}
        for doc, distance in results
    ]

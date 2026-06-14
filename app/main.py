"""API FastAPI para consultar esquemas DDL con Gemini y ChromaDB."""

import json
import os
from contextlib import asynccontextmanager
from typing import Optional
from typing import Literal

import chromadb
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from google import genai
from google.genai import types
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MODEL = "gemini-3.1-flash-lite-preview"
EMBED_MODEL = "gemini-embedding-2"
CHROMA_PATH = "./chroma_db"
CHROMA_HOST = os.environ.get("CHROMA_HOST")          # set by docker-compose
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))

# Estos se inicializan en el lifespan para no bloquear el import
gemini_client: genai.Client = None
chroma_client = None  # chromadb.HttpClient o PersistentClient según entorno
text_collection = None
image_collection = None


# ---------------------------------------------------------------------------
# Lifespan: inicialización al arrancar la app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa clientes al arrancar. Se ejecuta una sola vez."""
    global gemini_client, chroma_client, text_collection, image_collection

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no encontrada en variables de entorno.")

    # Inicializar Gemini
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    print("[startup] Gemini client inicializado.")

    # Inicializar ChromaDB
    # Si CHROMA_HOST está definido (ej: docker-compose), usar el servidor HTTP externo.
    # Si no, usar PersistentClient local (desarrollo fuera de Docker).
    if CHROMA_HOST:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        print(f"[startup] ChromaDB: conectado a http://{CHROMA_HOST}:{CHROMA_PORT}")
    else:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        print(f"[startup] ChromaDB: PersistentClient en {CHROMA_PATH}")
    text_collection = chroma_client.get_or_create_collection("ddls", embedding_function=None)
    # image_collection = chroma_client.get_or_create_collection("vouchers_financieros")
    print(f"[startup] ChromaDB: {text_collection.count()} esquemas registrados.")
    # print(f"[startup] ChromaDB: {image_collection.count()} docs en vouchers_financieros.")

    yield  # La app corre entre yield y el bloque de cleanup

    # Cleanup (opcional aquí, ChromaDB persiste solo)
    print("[shutdown] Cerrando app.")


app = FastAPI(
    title="Dat-IA API",
    version="0.1.0",
    description="API inicial para el agente analista de datos Dat-IA.",
    lifespan=lifespan
)


# ---------------------------------------------------------------------------
# Schemas de request / response
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    date_filter: Optional[str] = None   # ej: "2024-01" para solo normas >= esa fecha
    n_results: int = 3


class RAGResponse(BaseModel):
    answer: str
    sources: str
    confidence_note: str


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    collection: str
    chunks: list

class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str

# ---------------------------------------------------------------------------
# Utilidades internas (mismas funciones que en el notebook)
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Genera embeddings con gemini-embedding-2 (una llamada por texto).

    Nota: embed_content con una lista de strings devuelve un único embedding
    (los concatena). Se llama una vez por texto y se agregan los resultados.
    """
    embeddings = []
    for text in texts:
        result = gemini_client.models.embed_content(
            model=EMBED_MODEL,
            contents=text
        )
        embeddings.append(result.embeddings[0].values)
    return embeddings


def cargar_tablas(tablas: list) -> list[dict]:
    """
    Recibe la lista ya parseada del JSON y retorna una lista de diccionarios
    con la estructura: {"id": ..., "nombre": ..., "descripcion": ..., "ddl": ...}
    """
    return [
        {
            "id":          tabla["id"],
            "nombre":      tabla["nombre"],
            "descripcion": tabla["descripcion"],
            "ddl":         tabla["ddl"],
        }
        for tabla in tablas
    ]

def retrieve_chunks(
    query: str,
    collection,
    n_results: int = 3,
    where: Optional[dict] = None
) -> list[dict]:
    """Retrieval semántico contra una colección ChromaDB."""
    total = collection.count()
    if total == 0:
        return []

    query_emb = embed_texts([query])[0]

    kwargs = {
        "query_embeddings": [query_emb],
        "n_results": min(n_results, total),  # nunca pedir más de lo que hay
        "include": ["documents", "metadatas", "distances"]
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]


def build_rag_response(question: str, ddl: str) -> RAGResponse:
    """
    Construye el prompt de augmentation y llama a Gemini.
    Retorna RAGResponse estructurado.
    """

    augmented_prompt = f"""
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'

### Database Schema
The query will run on a database with the following schema:
{ddl}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""

    # Contenido: imagen (si hay) + prompt
    contents = [augmented_prompt]

    response = gemini_client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=600,
            response_mime_type="application/json",
            response_schema=RAGResponse
        )
    )

    parsed = json.loads(response.text)
    if "i do not know" in parsed.get("answer", "").lower():
        parsed["sources"] = ""

    return RAGResponse(**parsed)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "ok",
        "model": MODEL,
        "embed_model": EMBED_MODEL,
        "text_docs": text_collection.count()
    }

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="dat-ia-api",
        version=app.version,
    )

@app.get("/ready")
def ready() -> dict:
    return {
        "status": "ok",
        "database": "not_configured",
        "message": "La conexión a Supabase se configurará en una siguiente etapa.",
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: Optional[UploadFile] = File(default=None)
):
    # -- Indexación de texto (MD/TXT) --
    global text_collection

    raw = await file.read()
    content = json.loads(raw.decode("utf-8"))
    chunks = cargar_tablas(content)

    if not chunks:
        raise HTTPException(400, "No se encontraron tablas.")


    # Embed e indexar
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embed_texts([chunk["descripcion"] for chunk in batch])

        text_collection.upsert(
            ids        = [chunk["id"]          for chunk in batch],
            documents  = [chunk["descripcion"] for chunk in batch],
            embeddings = embeddings,
            metadatas  = [{"nombre": chunk["nombre"], "ddl": chunk["ddl"]} for chunk in batch],
        )

    return IngestResponse(status="ok", chunks_indexed=len(chunks), collection="ddls", chunks=chunks)


@app.post("/query/json", response_model=RAGResponse)
async def query_json(request: QueryRequest):
    """Consulta una tabla relevante y devuelve la respuesta generada por Gemini."""
    if text_collection is None or text_collection.count() == 0:
        raise HTTPException(503, "Colección vacía. Ingesta documentos primero.")

    chunks = retrieve_chunks(request.question, text_collection, n_results=1)

    if not chunks:
        raise HTTPException(404, "No se encontró ninguna tabla relevante.")

    best = chunks[0]
    return build_rag_response(request.question, best["metadata"]["ddl"])

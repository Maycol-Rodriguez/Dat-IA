"""API FastAPI para consultar esquemas DDL con LangChain + Gemini/ChromaDB.

Migración a LangChain
----------------------
Este módulo dejó de llamar directamente al SDK `google-genai` para las dos
operaciones que le pertenecen (generar SQL y generar embeddings). Ambas
pasan ahora por LangChain:

- `sql_llm`         -> LangChain `Runnable` usado para generar SQL.
- `embeddings_model` -> LangChain `Embeddings` usado para vectorizar texto.

La idea es que el resto de la app (retrieval, memoria de consultas,
ingesta) no conozca el proveedor concreto: solo llama a `embed_texts(...)`
y `build_rag_response(...)`. Esto permite reemplazar Gemini por
`defog/sqlcoder` (u otro modelo) cambiando únicamente `build_sql_llm()`,
sin tocar el resto del archivo. Ver el docstring de esa función para el
plan de migración concreto.

Nota sobre `app/optimizer/query_optimizer.py`: ese módulo no se modificó
(no se compartió su código fuente) y sigue esperando un cliente nativo de
`google-genai` con la interfaz `client.models.generate_content(...)`. Por
eso `gemini_client` (el cliente nativo) se mantiene junto a los nuevos
objetos de LangChain, solo para pasárselo a `optimize_query`. Si quieres
migrar también ese módulo a LangChain, compárteme su código y lo adapto
igual que aquí.
"""

import json
import os
import re
from contextlib import asynccontextmanager
from typing import Optional
from typing import Literal

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from app.memory.query_memory import (
    get_or_create_query_memory_collection,
    save_query_memory,
    search_query_memory,
)

from app.optimizer.query_optimizer import optimize_query

import torch
from transformers import AutoTokenizer\
    , AutoModelForSequenceClassification#, BitsAndBytesConfig, AutoModelForCausalLM


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
gemini_client: genai.Client = None    # cliente nativo, solo para app.optimizer.query_optimizer
sql_llm: Runnable = None              # LLM de generación de SQL, envuelto en LangChain
embeddings_model: Embeddings = None   # Modelo de embeddings, envuelto en LangChain
chroma_client = None  # chromadb.HttpClient o PersistentClient según entorno
text_collection = None
query_memory_collection = None
image_collection = None
shield_tokenizer = None
shield_model = None


# ---------------------------------------------------------------------------
# Fábricas de componentes LangChain (punto único de swap de proveedor)
# ---------------------------------------------------------------------------

def build_sql_llm() -> Runnable:
    """Devuelve el LLM usado para generar SQL, envuelto en LangChain.

    Hoy: Gemini vía `langchain-google-genai`.

    Plan de migración a defog/sqlcoder
    -----------------------------------
    `sqlcoder` es un modelo causal de HuggingFace (no un chat model), y solo
    sabe completar el bloque `[SQL]` de un prompt con el estilo usado en
    `SQL_GENERATION_PROMPT` (que ya sigue el formato oficial de sqlcoder:
    Task / Instructions / Database Schema / Answer). Por eso
    `build_rag_response` no le pide al LLM que devuelva JSON: solo texto SQL
    plano, que es exactamente lo que sqlcoder puede producir.

    Para migrar, basta con reemplazar el cuerpo de esta función por algo
    como:

        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
        causal_model = AutoModelForCausalLM.from_pretrained(
            "defog/sqlcoder-7b-2",
            device_map="auto",
            torch_dtype=torch.float16,
        )
        text_generation_pipeline = pipeline(
            "text-generation",
            model=causal_model,
            tokenizer=tokenizer,
            max_new_tokens=600,
            do_sample=False,
        )
        return HuggingFacePipeline(pipeline=text_generation_pipeline)

    Ni `build_sql_chain` ni `build_rag_response` necesitan cambiar: ambos
    proveedores exponen la interfaz `Runnable` de LangChain (`invoke`,
    composición con `|`).
    """
    return ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_output_tokens=600,
    )


def build_embeddings_model() -> Embeddings:
    """Devuelve el modelo de embeddings envuelto en LangChain (`Embeddings`).

    Mantenerlo detrás de esta función permite cambiar de proveedor (por
    ejemplo, a un modelo de embeddings local) sin tocar `embed_texts` ni
    ningún otro código, que solo conoce la interfaz `Embeddings` de
    LangChain.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )


# Prompt de generación de SQL (formato compatible con defog/sqlcoder)
SQL_GENERATION_PROMPT = PromptTemplate.from_template(
    """### Task
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
)


def build_sql_chain() -> Runnable:
    """Arma la cadena LCEL: prompt -> LLM -> texto plano.

    `sql_llm` se resuelve en runtime desde la variable global inicializada
    en el lifespan, así que cambiar de proveedor (ver `build_sql_llm`) no
    requiere tocar esta función.
    """
    return SQL_GENERATION_PROMPT | sql_llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Lifespan: inicialización al arrancar la app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa clientes al arrancar. Se ejecuta una sola vez."""
    global gemini_client, sql_llm, embeddings_model, chroma_client, text_collection, image_collection
    global query_memory_collection, shield_tokenizer, shield_model

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no encontrada en variables de entorno.")

    # Cliente nativo, requerido únicamente por app.optimizer.query_optimizer
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    print("[startup] Gemini client (nativo, para el optimizer) inicializado.")

    # Componentes LangChain usados por este módulo (generación SQL + embeddings)
    sql_llm = build_sql_llm()
    embeddings_model = build_embeddings_model()
    print("[startup] LangChain: sql_llm y embeddings_model inicializados.")

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

    query_memory_collection = get_or_create_query_memory_collection(chroma_client)
    print(
        f"[startup] Query memory: {query_memory_collection.count()} consultas registradas."
    )
    # print(f"[startup] ChromaDB: {image_collection.count()} docs en vouchers_financieros.")

    # Ingesta automática
    if text_collection.count() == 0:
            print("[startup] Colección vacía. Iniciando ingesta automática desde data/ddl.json...")
            try:
                with open("data/ddl.json", "r", encoding="utf-8") as f:
                    content = json.load(f)
                
                chunks = cargar_tablas(content)
                
                if chunks:
                    batch_size = 50
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i : i + batch_size]
                        embeddings = embed_texts([chunk["descripcion"] for chunk in batch])

                        text_collection.upsert(
                            ids        = [str(chunk["id"])       for chunk in batch],
                            documents  = [chunk["descripcion"]   for chunk in batch],
                            embeddings = embeddings,
                            metadatas  = [{"nombre": chunk["nombre"], "ddl": chunk["ddl"]} for chunk in batch],
                        )
                    print(f"[startup] Ingesta completada exitosamente. {len(chunks)} tablas indexadas.")
            except FileNotFoundError:
                print("[startup] ADVERTENCIA: No se encontró 'data/ddl.json' para la ingesta inicial.")
            except Exception as e:
                print(f"[startup] ERROR durante la ingesta automática: {e}")

    # Inicializar SQLPromptShield
    print("[startup] Cargando modelo SQLPromptShield...")
    shield_tokenizer = AutoTokenizer.from_pretrained("salmane11/SQLPromptShield")
    shield_model = AutoModelForSequenceClassification.from_pretrained("salmane11/SQLPromptShield")
    # shield_model.eval() # Recomendado: poner el modelo en modo evaluación
    print("[startup] SQLPromptShield cargado exitosamente.")

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
    question: str = Field(..., min_length=1)
    
class ShieldRequest(BaseModel):
    text_input: str

class RAGResponse(BaseModel):
    sql: str
    sources: str
    confidence_note: str
    status: str

class SHIELDResponse(BaseModel):
    sql: str
    sources: str
    confidence_note: str
    status: str


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    collection: str
    chunks: list

class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str

class EmbeddingsResponse(BaseModel):
    tabla: list[str]
    descripcion: list[str]
    distance: list[float]
    ddl: str


class MemorySearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    n_results: int = Field(default=3, ge=1, le=10)


class MemorySearchResult(BaseModel):
    question: str
    sql: str
    sources: str
    confidence_note: str
    status: str
    distance: float


class MemorySearchResponse(BaseModel):
    results: list[MemorySearchResult]


class MemoryStatsResponse(BaseModel):
    collection: str
    count: int
    status: str


class QueryOptimizeFilter(BaseModel):
    field: str
    operator: str
    value: str


class QueryOptimizeResponse(BaseModel):
    original_question: str
    normalized_question: str
    intent: str
    metrics: list[str]
    filters: list[QueryOptimizeFilter]
    date_range: dict[str, str] | None
    group_by: list[str]
    context: list[str]
    suggested_tables: list[str]
    optimizer: str

# ---------------------------------------------------------------------------
# Utilidades internas (mismas funciones que en el notebook)
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Genera embeddings usando LangChain (`embeddings_model`).

    El proveedor concreto (hoy Gemini vía `GoogleGenerativeAIEmbeddings`)
    queda encapsulado acá: el resto de la app (retrieval, memoria de
    consultas, ingesta) llama siempre a esta función y nunca al SDK
    subyacente.
    """
    return embeddings_model.embed_documents(texts)


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

def query_embeddings(collection, query: str, distance_threshold: float = 1.95) -> EmbeddingsResponse:
    """
    Consulta vectorial filtrando por distancia semántica.
    Solo retorna resultados con distancia <= threshold.
    """
    query_embedding = embed_texts([query])

    resultados = collection.query(
        query_embeddings=query_embedding,
        n_results=30,                        # trae más candidatos
        include=["metadatas", "documents", "distances"]  # incluir distancias
    )

    print(len(resultados))

    metadatas  = resultados['metadatas'][0]
    documents  = resultados['documents'][0]
    distances  = resultados['distances'][0]

    # Filtrar por umbral de distancia
    filtrados = [
        (meta, doc, dist)
        for meta, doc, dist in zip(metadatas, documents, distances)
        if dist <= distance_threshold
    ]

    if not filtrados:
        return EmbeddingsResponse(tabla=[], descripcion=[], ddl="")

    listTablas       = [meta['nombre'] for meta, doc, dist in filtrados]
    listDescripciones = [doc           for meta, doc, dist in filtrados]
    listDistances    = [dist          for meta, doc, dist in filtrados]
    listDdls         = [meta['ddl']    for meta, doc, dist in filtrados]

    ddls = '\n'.join(listDdls)

    return EmbeddingsResponse(tabla=listTablas, descripcion=listDescripciones, ddl=ddls, distance=listDistances)


def _clean_sql_output(text: str) -> str:
    """Limpia el texto crudo devuelto por el LLM (quita fences de markdown)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        first_newline = cleaned.find("\n")
        if first_newline != -1 and cleaned[:first_newline].strip().lower() in {"sql", ""}:
            cleaned = cleaned[first_newline + 1:]
    return cleaned.strip()


def _extract_table_names(ddl: str) -> str:
    """Extrae los nombres de tabla de un bloque de DDL (para `sources`)."""
    tablas = re.findall(r"CREATE TABLE\s+([A-Za-z0-9_\".]+)", ddl, flags=re.IGNORECASE)
    return ", ".join(tablas)


def build_rag_response(question: str, ddl: str) -> RAGResponse:
    """
    Ejecuta la cadena LangChain de generación de SQL (`build_sql_chain`) y
    arma un `RAGResponse`.

    A diferencia de la versión anterior (que le pedía a Gemini un JSON
    completo vía `response_schema`), aquí el LLM solo devuelve texto SQL
    plano: es la única salida que un modelo como defog/sqlcoder puede
    producir de forma confiable, dado que no soporta salida estructurada
    tipo function-calling. El resto de los campos de `RAGResponse` se arman
    en código, así que esta función funciona igual sin importar qué LLM
    haya detrás de `sql_llm`.
    """
    chain = build_sql_chain()
    raw_output = chain.invoke({"question": question, "ddl": ddl})

    if "i do not know" in raw_output.lower():
        return RAGResponse(
            sql="I do not know",
            sources="",
            confidence_note=(
                "El modelo no encontró una consulta SQL que responda la "
                "pregunta con el esquema disponible."
            ),
            status="no_match",
        )

    sql = _clean_sql_output(raw_output)

    return RAGResponse(
        sql=sql,
        sources=_extract_table_names(ddl),
        confidence_note="Generado a partir del esquema recuperado por similitud semántica.",
        status="success",
    )


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

@app.post("/query/optimize", response_model=QueryOptimizeResponse)
def query_optimize(request: QueryRequest) -> QueryOptimizeResponse:
    try:
        optimized_query = optimize_query(
            request.question,
            gemini_client=gemini_client,
            model=MODEL,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    return QueryOptimizeResponse(**optimized_query.to_dict())


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


@app.get("/memory/stats", response_model=MemoryStatsResponse)
def memory_stats() -> MemoryStatsResponse:
    if query_memory_collection is None:
        return MemoryStatsResponse(
            collection="query_memory",
            count=0,
            status="not_initialized",
        )

    return MemoryStatsResponse(
        collection="query_memory",
        count=query_memory_collection.count(),
        status="ok",
    )


@app.post("/memory/search", response_model=MemorySearchResponse)
def memory_search(request: MemorySearchRequest) -> MemorySearchResponse:
    if query_memory_collection is None:
        raise HTTPException(503, "La memoria de consultas no está inicializada.")

    query_embedding = embed_texts([request.question])[0]
    results = search_query_memory(
        query_memory_collection,
        embedding=query_embedding,
        n_results=request.n_results,
    )

    return MemorySearchResponse(
        results=[
            MemorySearchResult(
                question=str(result["metadata"].get("question", "")),
                sql=str(result["metadata"].get("sql", "")),
                sources=str(result["metadata"].get("sources", "")),
                confidence_note=str(result["metadata"].get("confidence_note", "")),
                status=str(result["metadata"].get("status", "")),
                distance=float(result["distance"]),
            )
            for result in results
        ]
    )


@app.post("/query/json", response_model=RAGResponse)
async def query_json(request: QueryRequest):
    """Consulta una tabla relevante y devuelve la respuesta generada por el LLM."""
    if text_collection is None or text_collection.count() == 0:
        return RAGResponse(sql="SELECT 1 AS prototype_result;", status="prototype",
                           sources="",confidence_note="")

    try:
        optimized_query = optimize_query(
            request.question,
            gemini_client=gemini_client,
            model=MODEL,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    query_for_generation = optimized_query.normalized_question

    resp = query_embeddings(
        text_collection,
        query_for_generation,
        distance_threshold=0.8,
    )

    if resp.ddl == "":
        raise HTTPException(422, "No se encontró ninguna tabla relevante.")

    rag_response = build_rag_response(query_for_generation, resp.ddl)

    if query_memory_collection is not None:
        try:
            memory_embedding = embed_texts([request.question])[0]
            save_query_memory(
                query_memory_collection,
                question=request.question,
                sql=rag_response.sql,
                embedding=memory_embedding,
                sources=rag_response.sources,
                confidence_note=rag_response.confidence_note,
                status=rag_response.status,
                model=MODEL,
            )
        except Exception as exc:
            print(f"[memory] ADVERTENCIA: No se pudo guardar la consulta: {exc}")

    return rag_response

@app.post("/query/shield", response_model=SHIELDResponse)
async def sql_shield(request: ShieldRequest):
    # Usamos las variables globales inicializadas en el lifespan
    inputs = shield_tokenizer(
        request.text_input, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    )

    with torch.no_grad():
        outputs = shield_model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    label = shield_model.config.id2label[predicted_class_id]
    score = probabilities[0][predicted_class_id].item()
    
    # IMPORTANTE: Tu función original retornaba una tupla, pero tienes 
    # response_model=SHIELDResponse. FastAPI dará error si no devuelves 
    # la estructura correcta de SHIELDResponse. Aquí te lo adapto:
    return SHIELDResponse(
        sql=request.text_input,
        sources="SQLPromptShield",
        confidence_note=f"Score: {score:.4f}",
        status=label
    )
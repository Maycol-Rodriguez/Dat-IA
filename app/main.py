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
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google import genai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from app.db.connect_db import create_db_engine
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
DATABASE_URL = os.environ.get("DATABASE_URL")
MODEL = "gemini-3.1-flash-lite-preview"
EMBED_MODEL = "gemini-embedding-2"
CHROMA_PATH = "./chroma_db"
CHROMA_HOST = os.environ.get("CHROMA_HOST")          # set by docker-compose
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))

# Estos se inicializan en el lifespan para no bloquear el import
rag_llm = None  # ChatGoogleGenerativeAI con salida estructurada (RAGResponse)
optimizer_llm = None  # ChatGoogleGenerativeAI usado por optimize_query (with_structured_output)
answer_llm = None  # ChatGoogleGenerativeAI usado por synthesize_answer (with_structured_output)
embeddings_model: GoogleGenerativeAIEmbeddings = None
chroma_client = None  # chromadb.HttpClient o PersistentClient según entorno
text_collection = None
query_memory_collection = None
image_collection = None
shield_tokenizer = None
shield_model = None
sql_database: SQLDatabase = None  # None si DATABASE_URL no está configurada


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
    global rag_llm, optimizer_llm, answer_llm, embeddings_model, chroma_client, text_collection, image_collection
    global query_memory_collection, shield_tokenizer, shield_model, sql_database

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no encontrada en variables de entorno.")

    # Inicializar LLM de generación SQL (LangChain) con salida estructurada
    rag_llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_output_tokens=600,
    ).with_structured_output(RAGResponse)
    print("[startup] LangChain ChatGoogleGenerativeAI (RAG) inicializado.")

    # Inicializar LLM del optimizer (LangChain, salida estructurada dentro de optimize_query)
    optimizer_llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_output_tokens=700,
    )
    print("[startup] LangChain ChatGoogleGenerativeAI (optimizer) inicializado.")

    # Inicializar LLM de síntesis de respuesta (LangChain, salida estructurada)
    answer_llm = ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_output_tokens=600,
    )
    print("[startup] LangChain ChatGoogleGenerativeAI (answer) inicializado.")

    # Inicializar embeddings (LangChain)
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    print("[startup] LangChain GoogleGenerativeAIEmbeddings inicializado.")

    # Inicializar ChromaDB
    # Si CHROMA_HOST está definido (ej: docker-compose), usar el servidor HTTP externo.
    # Si no, usar PersistentClient local (desarrollo fuera de Docker).
    if CHROMA_HOST:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        print(f"[startup] ChromaDB: conectado a http://{CHROMA_HOST}:{CHROMA_PORT}")
    else:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        print(f"[startup] ChromaDB: PersistentClient en {CHROMA_PATH}")
    text_collection = Chroma(
        client=chroma_client,
        collection_name="ddls",
        embedding_function=embeddings_model,
    )
    # image_collection = chroma_client.get_or_create_collection("vouchers_financieros")
    print(f"[startup] ChromaDB: {text_collection._collection.count()} esquemas registrados.")

    query_memory_collection = get_or_create_query_memory_collection(chroma_client, embeddings_model)
    print(
        f"[startup] Query memory: {query_memory_collection._collection.count()} consultas registradas."
    )
    # print(f"[startup] ChromaDB: {image_collection.count()} docs en vouchers_financieros.")

    # Inicializar SQLDatabase (LangChain) contra Supabase/Postgres, si está configurada.
    # Es opcional: si falla o no hay DATABASE_URL, /query/answer queda deshabilitado
    # pero el resto de la app (generación de SQL sin ejecutar) sigue funcionando.
    if DATABASE_URL:
        try:
            db_engine = create_db_engine(DATABASE_URL)
            sql_database = SQLDatabase(db_engine, lazy_table_reflection=True)
            print(f"[startup] SQLDatabase conectado (dialecto: {sql_database.dialect}).")
        except Exception as e:
            print(f"[startup] ADVERTENCIA: No se pudo conectar a DATABASE_URL: {e}")
    else:
        print("[startup] DATABASE_URL no configurada: /query/answer no podrá ejecutar SQL.")

    # Ingesta automática
    if text_collection._collection.count() == 0:
            print("[startup] Colección vacía. Iniciando ingesta automática desde data/ddl.json...")
            try:
                with open("data/ddl.json", "r", encoding="utf-8") as f:
                    content = json.load(f)

                chunks = cargar_tablas(content)

                if chunks:
                    batch_size = 50
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i : i + batch_size]

                        text_collection.add_texts(
                            texts     = [chunk["descripcion"] for chunk in batch],
                            metadatas = [{"nombre": chunk["nombre"], "ddl": chunk["ddl"]} for chunk in batch],
                            ids       = [str(chunk["id"]) for chunk in batch],
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


class AnswerResponse(BaseModel):
    answer: str
    sql: str
    data: list[dict]
    sources: str
    status: str


class _AnswerPayload(BaseModel):
    answer: str


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
    """Genera embeddings con gemini-embedding-2 vía LangChain (batching interno)."""
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

def query_embeddings(collection, query: str, distance_threshold: float = 0.7) -> EmbeddingsResponse:
    """
    Consulta vectorial filtrando por distancia semántica.
    Solo retorna resultados con distancia <= threshold.
    """
    resultados = collection.similarity_search_with_score(query, k=10)  # trae más candidatos

    # Filtrar por umbral de distancia
    filtrados = [
        (doc, dist) for doc, dist in resultados if dist <= distance_threshold
    ]

    if not filtrados:
        return EmbeddingsResponse(tabla=[], descripcion=[], ddl="", distance=[])

    listTablas        = [doc.metadata["nombre"] for doc, dist in filtrados]
    listDescripciones = [doc.page_content       for doc, dist in filtrados]
    listDistances     = [dist                   for doc, dist in filtrados]
    listDdls          = [doc.metadata["ddl"]    for doc, dist in filtrados]

    ddls = '\n'.join(listDdls)

    return EmbeddingsResponse(tabla=listTablas, descripcion=listDescripciones, ddl=ddls, distance=listDistances)


def build_rag_response(question: str, ddl: str) -> RAGResponse:
    """
    Construye el prompt de augmentation y llama al LLM (LangChain) con
    salida estructurada. Retorna RAGResponse.
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

    parsed: RAGResponse = rag_llm.invoke(augmented_prompt)

    if "i do not know" in parsed.sql.lower():
        parsed = parsed.model_copy(update={"sources": ""})

    return parsed


def execute_sql(db: SQLDatabase, sql: str, row_limit: int = 200) -> dict:
    """Ejecuta SQL de solo lectura contra Supabase con guardas de seguridad.

    Nunca lanza excepción: devuelve {"rows": [...]} en éxito o
    {"error": "..."} si el SQL no pasa las validaciones o falla al
    ejecutarse (defensa en profundidad, aunque el rol de BD ya sea de
    solo lectura).
    """
    stripped = sql.strip().rstrip(";")

    if not re.match(r"(?is)^select\b", stripped):
        return {"error": "Solo se permiten sentencias SELECT."}

    if ";" in stripped:
        return {"error": "Solo se permite una sentencia SQL por consulta."}

    result = db.run_no_throw(stripped, fetch="cursor")

    if isinstance(result, str):
        return {"error": result}

    rows = [dict(row) for row in result.mappings()]

    return {"rows": rows[:row_limit]}


def classify_shield(text_input: str) -> tuple[str, float]:
    """Clasifica un texto con SQLPromptShield. Devuelve (label, score).

    label es "SAFE" o "MALICIOUS" (id2label del modelo).
    """
    inputs = shield_tokenizer(
        text_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = shield_model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    label = shield_model.config.id2label[predicted_class_id]
    score = probabilities[0][predicted_class_id].item()

    return label, score


def synthesize_answer(llm, question: str, sql: str, rows: list[dict]) -> str:
    """Sintetiza una respuesta en lenguaje natural a partir del resultado SQL.

    Responde siempre en español, sin importar el idioma de la pregunta
    original (mismo criterio que optimize_query para normalized_question).
    """
    prompt = f"""
    Eres un analista de datos. Responde la pregunta del usuario en español,
    de forma clara y concisa, usando exclusivamente el resultado de la
    consulta SQL de abajo. Menciona el número de filas si es relevante.
    No inventes datos que no estén en el resultado.

    Pregunta: {question}
    SQL ejecutado: {sql}
    Resultado ({len(rows)} filas): {rows}
    """

    structured_llm = llm.with_structured_output(_AnswerPayload)
    return structured_llm.invoke(prompt).answer


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
        "text_docs": text_collection._collection.count()
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
        "database": "connected" if sql_database is not None else "not_configured",
        "message": (
            f"Conectado (dialecto: {sql_database.dialect})."
            if sql_database is not None
            else "DATABASE_URL no configurada o la conexión a Supabase falló al arrancar."
        ),
    }

@app.post("/query/optimize", response_model=QueryOptimizeResponse)
def query_optimize(request: QueryRequest) -> QueryOptimizeResponse:
    try:
        optimized_query = optimize_query(
            request.question,
            llm=optimizer_llm,
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

        text_collection.add_texts(
            texts     = [chunk["descripcion"] for chunk in batch],
            metadatas = [{"nombre": chunk["nombre"], "ddl": chunk["ddl"]} for chunk in batch],
            ids       = [chunk["id"] for chunk in batch],
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
        count=query_memory_collection._collection.count(),
        status="ok",
    )


@app.post("/memory/search", response_model=MemorySearchResponse)
def memory_search(request: MemorySearchRequest) -> MemorySearchResponse:
    if query_memory_collection is None:
        raise HTTPException(503, "La memoria de consultas no está inicializada.")

    results = search_query_memory(
        query_memory_collection,
        query=request.question,
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
    """Consulta una tabla relevante y devuelve la respuesta generada por Gemini."""
    if text_collection is None or text_collection._collection.count() == 0:
        return RAGResponse(sql="SELECT 1 AS prototype_result;", status="prototype",
                           sources="",confidence_note="")

    try:
        optimized_query = optimize_query(
            request.question,
            llm=optimizer_llm,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    query_for_generation = optimized_query.normalized_question

    print(f"Query for generation: {query_for_generation}")

    resp = query_embeddings(
        text_collection,
        query_for_generation,
        distance_threshold=0.7,
    )

    if resp.ddl == "":
        raise HTTPException(422, "No se encontró ninguna tabla relevante.")
    
    print(f"Found table: {resp.ddl}")

    rag_response = build_rag_response(query_for_generation, resp.ddl)

    if query_memory_collection is not None:
        try:
            save_query_memory(
                query_memory_collection,
                question=request.question,
                sql=rag_response.sql,
                sources=rag_response.sources,
                confidence_note=rag_response.confidence_note,
                status=rag_response.status,
                model=MODEL,
            )
        except Exception as exc:
            print(f"[memory] ADVERTENCIA: No se pudo guardar la consulta: {exc}")

    return rag_response


@app.post("/query/answer", response_model=AnswerResponse)
async def query_answer(request: QueryRequest):
    """Flujo completo: shield -> optimizer -> retrieval -> SQL -> ejecución -> respuesta."""
    label, _score = classify_shield(request.question)
    if label == "MALICIOUS":
        raise HTTPException(422, "La pregunta fue bloqueada por el filtro de seguridad.")

    if text_collection is None or text_collection._collection.count() == 0:
        return AnswerResponse(
            answer="La base de conocimiento todavía no tiene tablas indexadas.",
            sql="SELECT 1 AS prototype_result;",
            data=[],
            sources="",
            status="prototype",
        )

    try:
        optimized_query = optimize_query(
            request.question,
            llm=optimizer_llm,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    query_for_generation = optimized_query.normalized_question

    resp = query_embeddings(
        text_collection,
        query_for_generation,
        distance_threshold=0.7,
    )

    if resp.ddl == "":
        raise HTTPException(422, "No se encontró ninguna tabla relevante.")

    rag_response = build_rag_response(query_for_generation, resp.ddl)

    if rag_response.sources == "":
        return AnswerResponse(
            answer="No encontré información suficiente en el esquema disponible para responder esta pregunta.",
            sql=rag_response.sql,
            data=[],
            sources="",
            status=rag_response.status,
        )

    if sql_database is None:
        raise HTTPException(503, "La ejecución de SQL no está configurada (DATABASE_URL faltante).")

    execution = execute_sql(sql_database, rag_response.sql)

    if "error" in execution:
        return AnswerResponse(
            answer=f"La consulta generada falló al ejecutarse: {execution['error']}",
            sql=rag_response.sql,
            data=[],
            sources=rag_response.sources,
            status="error",
        )

    rows = execution["rows"]
    answer_text = synthesize_answer(answer_llm, request.question, rag_response.sql, rows)

    if query_memory_collection is not None:
        try:
            save_query_memory(
                query_memory_collection,
                question=request.question,
                sql=rag_response.sql,
                sources=rag_response.sources,
                confidence_note=rag_response.confidence_note,
                status="success",
                model=MODEL,
            )
        except Exception as exc:
            print(f"[memory] ADVERTENCIA: No se pudo guardar la consulta: {exc}")

    return AnswerResponse(
        answer=answer_text,
        sql=rag_response.sql,
        data=rows,
        sources=rag_response.sources,
        status="success",
    )


@app.post("/query/shield", response_model=SHIELDResponse)
async def sql_shield(request: ShieldRequest):
    label, score = classify_shield(request.text_input)

    return SHIELDResponse(
        sql=request.text_input,
        sources="SQLPromptShield",
        confidence_note=f"Score: {score:.4f}",
        status=label
    )
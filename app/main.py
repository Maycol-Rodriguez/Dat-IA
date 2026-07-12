"""API FastAPI para consultar esquemas DDL con LangChain y ChromaDB.

Migración de google-genai directo → LangChain.
Preparado para reemplazar el LLM por defog/sqlcoder (HuggingFace) en el futuro.
"""

import json
import os
from contextlib import asynccontextmanager
from typing import Optional, Literal

import chromadb
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------------------------
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# LLM principal (fácil de intercambiar por defog/sqlcoder — ver comentario al final)
LLM_MODEL = "gemini-3.5-flash"

# Modelo de embeddings (Google)
EMBED_MODEL = "models/gemini-embedding-001"

CHROMA_PATH = "./chroma_db"
CHROMA_HOST = os.environ.get("CHROMA_HOST")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
COLLECTION_NAME = "ddls"

# Globals inicializados en lifespan
llm: ChatGoogleGenerativeAI = None
embeddings_model: GoogleGenerativeAIEmbeddings = None
vectorstore: Chroma = None          # wrapper LangChain sobre ChromaDB
chroma_client = None
text_collection = None              # colección ChromaDB nativa (para conteos / upsert directo)
shield_tokenizer = None
shield_model = None

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, embeddings_model, vectorstore, chroma_client, text_collection
    global shield_tokenizer, shield_model

    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY no encontrada en variables de entorno.")

    # -- LLM (LangChain wrapper) -------------------------------------------
    # Para migrar a defog/sqlcoder reemplaza este bloque por:
    #
    #   from langchain_huggingface import HuggingFacePipeline
    #   from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    #   tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
    #   model     = AutoModelForCausalLM.from_pretrained("defog/sqlcoder-7b-2", ...)
    #   pipe      = pipeline("text-generation", model=model, tokenizer=tokenizer, ...)
    #   llm       = HuggingFacePipeline(pipeline=pipe)
    #
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
        max_output_tokens=600,
    )
    print(f"[startup] LLM inicializado: {LLM_MODEL}")

    # -- Embeddings (LangChain wrapper) ------------------------------------
    # Para migrar a defog/sqlcoder puedes conservar estos embeddings de Google
    # o cambiarlos por HuggingFaceEmbeddings:
    #
    #   from langchain_huggingface import HuggingFaceEmbeddings
    #   embeddings_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    #
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    print(f"[startup] Embeddings inicializados: {EMBED_MODEL}")

    # -- ChromaDB ----------------------------------------------------------
    if CHROMA_HOST:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        print(f"[startup] ChromaDB HTTP: {CHROMA_HOST}:{CHROMA_PORT}")
    else:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        print(f"[startup] ChromaDB Persistent: {CHROMA_PATH}")

    # Colección nativa (para conteos y upsert directo con embeddings pre-calculados)
    text_collection = chroma_client.get_or_create_collection(
        COLLECTION_NAME, embedding_function=None
    )

    # Wrapper LangChain sobre la misma colección
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
    )
    print(f"[startup] {text_collection.count()} esquemas en ChromaDB.")

    # -- Ingesta automática ------------------------------------------------
    if text_collection.count() == 0:
        print("[startup] Colección vacía — ingesta desde data/ddl.json ...")
        try:
            with open("data/ddl.json", "r", encoding="utf-8") as f:
                content = json.load(f)
            await _ingest_chunks(cargar_tablas(content))
            print(f"[startup] Ingesta completada: {text_collection.count()} tablas.")
        except FileNotFoundError:
            print("[startup] ADVERTENCIA: data/ddl.json no encontrado.")
        except Exception as exc:
            print(f"[startup] ERROR en ingesta automática: {exc}")

    # -- SQLPromptShield ---------------------------------------------------
    print("[startup] Cargando SQLPromptShield ...")
    shield_tokenizer = AutoTokenizer.from_pretrained("salmane11/SQLPromptShield")
    shield_model = AutoModelForSequenceClassification.from_pretrained(
        "salmane11/SQLPromptShield"
    )
    shield_model.eval()
    print("[startup] SQLPromptShield listo.")

    yield

    print("[shutdown] Cerrando app.")


app = FastAPI(
    title="Dat-IA API",
    version="0.2.0",
    description="API del agente analista Dat-IA — backend LangChain.",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

class ShieldRequest(BaseModel):
    text_input: str

class RAGResponse(BaseModel):
    sql: str
    sources: str = ""
    confidence_note: str = ""
    status: str = "ok"

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


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def cargar_tablas(tablas: list) -> list[dict]:
    """Normaliza el JSON de DDL al formato interno."""
    return [
        {
            "id":          str(tabla["id"]),
            "nombre":      tabla["nombre"],
            "descripcion": tabla["descripcion"],
            "ddl":         tabla["ddl"],
        }
        for tabla in tablas
    ]


async def _ingest_chunks(chunks: list[dict], batch_size: int = 50) -> None:
    """Indexa chunks en ChromaDB usando embeddings de LangChain."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["descripcion"] for c in batch]

        # Genera embeddings con el wrapper LangChain
        batch_embeddings = embeddings_model.embed_documents(texts)

        text_collection.upsert(
            ids        = [c["id"]    for c in batch],
            documents  = texts,
            embeddings = batch_embeddings,
            metadatas  = [{"nombre": c["nombre"], "ddl": c["ddl"]} for c in batch],
        )


def _retrieve_ddl(question: str, distance_threshold: float = 0.8) -> EmbeddingsResponse:
    """
    Recupera DDLs relevantes consultando ChromaDB directamente.
 
    Usa embed_query de LangChain para generar el vector de la pregunta y luego
    consulta text_collection con la API nativa de ChromaDB.
 
    Por qué no usamos vectorstore.similarity_search_with_relevance_scores():
    La colección fue indexada pasando embeddings pre-calculados (embedding_function=None),
    por lo que el wrapper LangChain/Chroma no puede normalizar correctamente los scores
    y devuelve valores negativos, rompiendo el filtro por umbral.
    Consultando ChromaDB directamente obtenemos la distancia coseno cruda [0, 2],
    donde 0 = idéntico y 2 = opuesto, igual que hacía el código original.
    """
    query_embedding = embeddings_model.embed_query(question)
 
    results = text_collection.query(
        query_embeddings=[query_embedding],
        n_results=min(10, text_collection.count()),
        include=["metadatas", "documents", "distances"],
    )
 
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]
 
    # Log de diagnóstico: muestra las 3 distancias más cercanas
    top = sorted(zip(distances, [m["nombre"] for m in metadatas]))[:3]
    print(f"[retrieve] Top-3 distancias (umbral={distance_threshold}): "
          + ", ".join(f"{n}={d:.4f}" for d, n in top))
 
    filtrados = [
        (meta, doc, dist)
        for meta, doc, dist in zip(metadatas, documents, distances)
        if dist <= distance_threshold
    ]
 
    if not filtrados:
        print(f"[retrieve] Ningún resultado bajo el umbral. Min distancia: {min(distances):.4f}")
        return EmbeddingsResponse(tabla=[], descripcion=[], distance=[], ddl="")
 
    tablas        = [meta["nombre"] for meta, _, __ in filtrados]
    descripciones = [doc            for _, doc, __ in filtrados]
    dists         = [dist           for _, __, dist in filtrados]
    ddls          = "\n".join(meta["ddl"] for meta, _, __ in filtrados)
 
    return EmbeddingsResponse(
        tabla=tablas,
        descripcion=descripciones,
        distance=dists,
        ddl=ddls,
    )


# ---------------------------------------------------------------------------
# Cadena RAG con LangChain (LCEL)
# ---------------------------------------------------------------------------

# Prompt de augmentación
# Al migrar a defog/sqlcoder, sólo necesitarás ajustar este template al
# formato que espera ese modelo (usa ### Instruction / ### Context / ### Response).
_SQL_PROMPT = PromptTemplate(
    input_variables=["question", "ddl"],
    template="""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- If you cannot answer the question with the available database schema, return 'I do not know'
- Return ONLY valid JSON with keys: sql, sources, confidence_note, status
- Do not include markdown fences or extra text

### Database Schema
{ddl}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
""",
)

_json_parser = JsonOutputParser()


def _build_rag_chain():
    """
    Construye la cadena LCEL: prompt | llm | parser.

    Separar la construcción en una función hace que sea trivial
    intercambiar `llm` por HuggingFacePipeline(defog/sqlcoder) en el futuro.
    """
    return _SQL_PROMPT | llm | _json_parser


def _run_rag(question: str, ddl: str) -> RAGResponse:
    """Ejecuta la cadena RAG y devuelve un RAGResponse."""
    chain = _build_rag_chain()
    parsed = chain.invoke({"question": question, "ddl": ddl})

    if "i do not know" in parsed.get("sql", "").lower():
        parsed["sources"] = ""

    return RAGResponse(**parsed)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "text_docs": text_collection.count(),
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="dat-ia-api", version=app.version)


@app.get("/ready")
def ready() -> dict:
    return {
        "status": "ok",
        "database": "not_configured",
        "message": "La conexión a Supabase se configurará en una siguiente etapa.",
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: Optional[UploadFile] = File(default=None)):
    raw = await file.read()
    content = json.loads(raw.decode("utf-8"))
    chunks = cargar_tablas(content)

    if not chunks:
        raise HTTPException(400, "No se encontraron tablas en el archivo.")

    await _ingest_chunks(chunks)

    return IngestResponse(
        status="ok",
        chunks_indexed=len(chunks),
        collection=COLLECTION_NAME,
        chunks=chunks,
    )


@app.post("/query/json", response_model=RAGResponse)
async def query_json(request: QueryRequest):
    """Recupera DDLs relevantes y genera SQL con la cadena LangChain."""
    if text_collection is None or text_collection.count() == 0:
        return RAGResponse(
            sql="SELECT 1 AS prototype_result;",
            status="prototype",
            sources="",
            confidence_note="",
        )

    resp = _retrieve_ddl(request.question, distance_threshold=0.8)

    if not resp.ddl:
        raise HTTPException(422, "No se encontró ninguna tabla relevante.")

    return _run_rag(request.question, resp.ddl)


@app.post("/query/shield", response_model=SHIELDResponse)
async def sql_shield(request: ShieldRequest):
    """Clasifica el input con SQLPromptShield (modelo HuggingFace local)."""
    inputs = shield_tokenizer(
        request.text_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = shield_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    label = shield_model.config.id2label[pred_id]
    score = probs[0][pred_id].item()

    return SHIELDResponse(
        sql=request.text_input,
        sources="SQLPromptShield",
        confidence_note=f"Score: {score:.4f}",
        status=label,
    )


# ---------------------------------------------------------------------------
# NOTA: Migración futura a defog/sqlcoder
# ---------------------------------------------------------------------------
# 1. Instala: pip install langchain-huggingface transformers accelerate bitsandbytes
#
# 2. En el bloque `llm` del lifespan, reemplaza ChatGoogleGenerativeAI por:
#
#    from langchain_huggingface import HuggingFacePipeline
#    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
#
#    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
#    tokenizer  = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
#    model      = AutoModelForCausalLM.from_pretrained(
#                     "defog/sqlcoder-7b-2",
#                     quantization_config=bnb_config,
#                     device_map="auto",
#                 )
#    pipe = pipeline(
#        "text-generation", model=model, tokenizer=tokenizer,
#        max_new_tokens=300, temperature=0.0, do_sample=False,
#    )
#    llm = HuggingFacePipeline(pipeline=pipe)
#
# 3. Ajusta `_SQL_PROMPT` al formato que espera sqlcoder:
#
#    ### Task
#    {question}
#    ### Database Schema
#    {ddl}
#    ### SQL Query
#
# El resto del código (cadena LCEL, endpoints, ChromaDB) no cambia.
# ---------------------------------------------------------------------------
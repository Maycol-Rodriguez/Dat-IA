from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Dat-IA API",
    version="0.1.0",
    description="API inicial para el agente analista de datos Dat-IA.",
)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Pregunta de negocio en lenguaje natural.",
    )


class AskResponse(BaseModel):
    status: Literal["prototype"]
    question: str
    answer: str
    sql: str | None = None


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


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    prototype_sql = "SELECT 1 AS prototype_result;"

    return AskResponse(
        status="prototype",
        question=payload.question,
        answer=(
            "Esta es una respuesta provisional del prototipo. "
            "El módulo Text-to-SQL y la conexión a Supabase se integrarán en una siguiente etapa."
        ),
        sql=prototype_sql,
    )

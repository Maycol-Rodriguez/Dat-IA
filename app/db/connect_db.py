"""Conexión a la base de datos relacional (Supabase/Postgres) para Dat-IA."""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def create_db_engine(database_url: str) -> Engine:
    """Crea el engine de SQLAlchemy contra la BD relacional.

    No se ejecuta a nivel de módulo: si DATABASE_URL no está configurada,
    importar este módulo no debe fallar (lo llama el lifespan de FastAPI,
    que decide si la conexión es obligatoria u opcional).

    `statement_timeout` acota el tiempo de ejecución en Postgres a 60s:
    validate_sql ya no acota el LIMIT del lado de la BD (ver
    `app/validation/sql_validator.py`), así que esta es la única red de
    seguridad contra un SQL generado sin filtro que dispare un scan largo.
    """
    return create_engine(database_url, connect_args={"options": "-c statement_timeout=60000"})

"""Conexión a la base de datos relacional (Supabase/Postgres) para Dat-IA."""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def create_db_engine(database_url: str) -> Engine:
    """Crea el engine de SQLAlchemy contra la BD relacional.

    No se ejecuta a nivel de módulo: si DATABASE_URL no está configurada,
    importar este módulo no debe fallar (lo llama el lifespan de FastAPI,
    que decide si la conexión es obligatoria u opcional).
    """
    return create_engine(database_url)

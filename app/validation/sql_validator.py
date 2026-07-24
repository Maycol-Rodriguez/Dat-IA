"""Validación determinística del SQL generado, previa a su ejecución.

Compone capas en orden barato → caro: forma de la sentencia (solo SELECT,
sin apilar), parsear a un AST, comparar las tablas citadas contra el esquema
recuperado, acotar el LIMIT, y un dry-run con `EXPLAIN` contra la base de
datos real (sin leer filas) para validar columnas y tipos. No sustituye al
juez LLM: solo captura los niveles de error más baratos de detectar.

Las guardas de solo-SELECT y anti-stacking vivían antes dentro de
`execute_sql` (`app/main.py`), donde se aplicaban justo antes de ejecutar
—después de gastar el juez LLM y el bucle de reintento en un SQL que iba a
ser rechazado de todos modos. Aquí son el primer filtro, antes de cualquier
llamada cara.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import sqlglot
from langchain_community.utilities import SQLDatabase
from sqlglot import exp

SqlValidationStage = Literal["syntax", "statement", "tables", "dry_run", "ok"]

DEFAULT_ROW_LIMIT = 200


@dataclass(frozen=True)
class SqlValidation:
    """Resultado de validar un SQL generado antes de ejecutarlo.

    `sql` trae la sentencia final a ejecutar (con el `LIMIT` ya acotado)
    cuando `is_valid=True`; queda vacío en cualquier rechazo.
    """

    is_valid: bool
    stage: SqlValidationStage
    error: str
    sql: str = ""


def dry_run_explain(db: SQLDatabase, sql: str) -> str | None:
    """Corre `EXPLAIN` sobre el SQL sin ejecutarlo.

    Reutiliza `db.run_no_throw`, el mismo patrón que ya usa `execute_sql`
    (`app/main.py`): devuelve un `str` con el error en vez de lanzar una
    excepción. `EXPLAIN` (nunca `EXPLAIN ANALYZE`) construye el plan de
    ejecución sin leer una sola fila, lo que exige resolver tablas, columnas
    y tipos igual que lo haría la ejecución real.

    Returns:
        `None` si el plan se construyó sin error, o el mensaje de error si
        `EXPLAIN` falló (columna/tabla inexistente, tipo incompatible, etc.).
    """
    result = db.run_no_throw(f"EXPLAIN {sql}", fetch="cursor")
    return result if isinstance(result, str) else None


def _read_limit_value(tree: exp.Expression) -> int | None:
    """Lee el valor numérico del `LIMIT` del AST, si existe y es interpretable."""
    limit_node = tree.find(exp.Limit)

    if limit_node is None:
        return None

    try:
        return int(limit_node.expression.this)
    except (AttributeError, TypeError, ValueError):
        return None


def _enforce_row_limit(tree: exp.Select, max_rows: int) -> exp.Select:
    """Garantiza que el AST tenga un `LIMIT` <= `max_rows`.

    `tree.limit(n)` reemplaza cualquier `LIMIT` existente en vez de
    duplicarlo, así que basta con recalcular el valor efectivo y aplicarlo
    siempre. Un `LIMIT` ausente, no numérico (p. ej. `LIMIT ALL`, que
    sqlglot descarta silenciosamente) o mayor a `max_rows` se acota a
    `max_rows`; uno ya menor se conserva.
    """
    current = _read_limit_value(tree)
    effective = min(current, max_rows) if current is not None else max_rows
    return tree.limit(effective)


def validate_sql(
    sql: str,
    allowed_tables: list[str],
    db: SQLDatabase | None = None,
    max_rows: int = DEFAULT_ROW_LIMIT,
) -> SqlValidation:
    """Valida forma, sintaxis, tablas citadas, acota el LIMIT y hace dry-run.

    Args:
        sql: SQL generado por el LLM, aún sin ejecutar.
        allowed_tables: nombres de tabla recuperados para esta pregunta
            (ver `retrieve_ddl_context` en `app/main.py`).
        db: conexión para el dry-run con `EXPLAIN`. Si es `None` (por
            ejemplo, `DATABASE_URL` no configurada), esa etapa se omite sin
            marcarse como error.
        max_rows: tope de filas exigido en el `LIMIT` final.

    Returns:
        `SqlValidation` con `is_valid=True`, `stage="ok"` y el SQL final
        (con `LIMIT` acotado) en `sql`, si pasa todas las etapas aplicables.
    """
    stripped = sql.strip().rstrip(";")

    if not stripped:
        return SqlValidation(is_valid=False, stage="syntax", error="El SQL está vacío.")

    if ";" in stripped:
        return SqlValidation(
            is_valid=False,
            stage="statement",
            error="Solo se permite una sentencia SQL por consulta.",
        )

    try:
        tree = sqlglot.parse_one(stripped, read="postgres")
    except sqlglot.errors.ParseError as exc:
        return SqlValidation(is_valid=False, stage="syntax", error=str(exc))

    if not isinstance(tree, exp.Select):
        return SqlValidation(
            is_valid=False,
            stage="statement",
            error="Solo se permiten sentencias SELECT.",
        )

    cte_aliases = {cte.alias for cte in tree.find_all(exp.CTE)}
    cited_tables = {table.name for table in tree.find_all(exp.Table)} - cte_aliases
    unknown_tables = cited_tables - set(allowed_tables)

    if unknown_tables:
        return SqlValidation(
            is_valid=False,
            stage="tables",
            error=f"Tablas no reconocidas en el esquema recuperado: {sorted(unknown_tables)}",
        )

    tree = _enforce_row_limit(tree, max_rows)
    bounded_sql = tree.sql(dialect="postgres")

    if db is not None:
        dry_run_error = dry_run_explain(db, bounded_sql)
        if dry_run_error is not None:
            return SqlValidation(is_valid=False, stage="dry_run", error=dry_run_error)

    return SqlValidation(is_valid=True, stage="ok", error="", sql=bounded_sql)

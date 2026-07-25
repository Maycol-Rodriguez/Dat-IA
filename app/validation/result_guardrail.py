"""Guardrail ligero sobre el resultado ejecutado y la redacción final.

El SQL ya pasó el validador determinístico y el juez (Clases 2-3) antes de
ejecutarse. Lo que queda sin cubrir ocurre después: `execute_sql` puede
truncar filas en silencio o devolver una métrica en NULL, y el LLM que
redacta la respuesta en lenguaje natural puede inventar o redondear
números que no están en las filas.

Deliberadamente NO es un segundo juez semántico: eso duplicaría el trabajo
de `sql_judge` y añadiría otra llamada cara antes de responder. Son
chequeos deterministicos sobre las filas (`check_result`) más una
verificación de groundedness barata sobre el texto (`check_groundedness`),
sin LLM en la primera pasada.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from app.optimizer.query_optimizer import OptimizedQuery
from app.validation.sql_validator import DEFAULT_ROW_LIMIT


@dataclass(frozen=True)
class ResultCheck:
    """Señales de alerta detectadas mirando solo las filas, sin LLM."""

    ok: bool
    warnings: list[str]


@dataclass(frozen=True)
class GroundednessCheck:
    """Números del texto redactado que no aparecen entre los valores de las filas."""

    ok: bool
    unsupported_numbers: list[str]


def _find_metric_column(metric: str, row: dict) -> str | None:
    """Busca una columna cuyo nombre corresponda al identificador de metric.

    El generador nunca recibe instrucción estricta de nombrar la columna
    exactamente como el metric (ej. una fila puede traer
    `avg_resolution_time_hr` en vez de `resolution_time_hr`), así que una
    igualdad exacta produce falsos positivos casi siempre. Se acepta una
    coincidencia por substring en cualquier dirección, sin distinguir
    mayúsculas/minúsculas, como heurística tolerante.
    """
    metric_lower = metric.lower()

    for key in row:
        key_lower = key.lower()

        if metric_lower in key_lower or key_lower in metric_lower:
            return key

    return None


def check_result(
    rows: list[dict],
    optimized_query: OptimizedQuery,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> ResultCheck:
    """Detecta resultado vacío, métrica en NULL o truncación silenciosa.

    Args:
        rows: filas devueltas por `execute_sql`.
        optimized_query: para saber qué columnas son métricas a verificar.
        row_limit: el mismo tope pasado a `execute_sql`; si `len(rows)`
            lo iguala, es señal de que probablemente hay más filas de las
            mostradas (la truncación de `execute_sql` es silenciosa). Usa
            el mismo `DEFAULT_ROW_LIMIT` que ya acota el SQL en
            `validate_sql`, para que ambos topes no diverjan.

    Returns:
        `ResultCheck` con `ok=True` solo si no hay ninguna advertencia.
        Las advertencias no bloquean la respuesta, solo la marcan.
    """
    warnings: list[str] = []

    if not rows:
        warnings.append("La consulta no devolvió filas.")

    if len(rows) == row_limit:
        warnings.append(
            f"El resultado se truncó a {row_limit} filas; puede haber más datos."
        )

    if rows:
        for metric in optimized_query.metrics:
            column = _find_metric_column(metric, rows[0])

            # Si ninguna columna se parece al nombre del metric, no hay
            # suficiente evidencia para advertir: puede ser que el SQL
            # simplemente la haya nombrado de forma irreconocible, no que
            # el dato venga vacío.
            if column is None:
                continue

            if all(row.get(column) is None for row in rows):
                warnings.append(f"La métrica '{metric}' vino vacía en todas las filas.")

    return ResultCheck(ok=not warnings, warnings=warnings)


def check_groundedness(
    answer: str,
    rows: list[dict],
    tolerance: float = 0.01,
) -> GroundednessCheck:
    """Verifica que cada número del texto exista entre los valores de las filas.

    Extrae números del texto con una regex y los compara contra los
    valores numéricos de `rows`, con tolerancia de redondeo. Tiene falsos
    positivos esperables (números de fila, conteos, porcentajes derivados
    de dos columnas): por diseño es una señal de sospecha para disparar
    una única regeneración de la redacción, no un bloqueo automático.

    Args:
        answer: texto redactado por `synthesize_answer`.
        rows: filas reales que `answer` debería estar describiendo.
        tolerance: fracción de tolerancia relativa al comparar (0.01 = 1%).

    Returns:
        `GroundednessCheck` con los números del texto que no encontraron
        respaldo en `rows`.
    """
    numbers_in_answer = re.findall(r"\d+[.,]?\d*", answer)
    row_values = {
        round(float(value), 2)
        for row in rows
        for value in row.values()
        # Decimal es lo que devuelven las columnas NUMERIC de Postgres via
        # psycopg2/SQLAlchemy para execute_sql, no un float nativo.
        if isinstance(value, (int, float, Decimal))
    }

    unsupported = []
    for raw in numbers_in_answer:
        candidate = float(raw.replace(",", ""))
        if not any(
            abs(candidate - v) <= tolerance * max(abs(v), 1) for v in row_values
        ):
            unsupported.append(raw)

    return GroundednessCheck(ok=not unsupported, unsupported_numbers=unsupported)

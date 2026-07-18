"""Optimizador híbrido para preguntas Text-to-SQL.

Primero intenta normalizar la pregunta con Gemini. Si Gemini no está disponible
o falla, usa un fallback por reglas. Este módulo no ejecuta SQL ni modifica
el flujo RAG existente.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, Field

ALLOWED_INTENTS = {
    "ranking",
    "count",
    "aggregation",
    "temporal_trend",
    "comparison",
    "detail",
}


class _OptimizerFilterPayload(BaseModel):
    field: str
    operator: str = "="
    value: str


class _OptimizerPayload(BaseModel):
    """Esquema de salida estructurada para el optimizer vía LangChain."""

    normalized_question: str
    intent: str
    metrics: list[str] = Field(default_factory=list)
    filters: list[_OptimizerFilterPayload] = Field(default_factory=list)
    date_range: dict[str, str] | None = None
    group_by: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    suggested_tables: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class QueryFilter:
    field: str
    operator: str
    value: str


@dataclass(frozen=True)
class OptimizedQuery:
    original_question: str
    normalized_question: str
    intent: str
    operation: str
    metrics: list[str]
    filters: list[QueryFilter]
    date_range: dict[str, str] | None
    group_by: list[str]
    context: list[str]
    suggested_tables: list[str]
    optimizer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "normalized_question": self.normalized_question,
            "intent": self.intent,
            "operation": self.operation,
            "metrics": self.metrics,
            "filters": [asdict(query_filter) for query_filter in self.filters],
            "date_range": self.date_range,
            "group_by": self.group_by,
            "context": self.context,
            "suggested_tables": self.suggested_tables,
            "optimizer": self.optimizer,
        }


def optimize_query(
    question: str,
    *,
    llm: Any | None = None,
    use_llm: bool = True,
) -> OptimizedQuery:
    """Optimiza una pregunta usando un LLM (LangChain) y fallback por reglas."""
    cleaned_question = _clean_question(question)

    if not cleaned_question:
        raise ValueError("La pregunta no puede estar vacía.")

    if use_llm and llm is not None:
        try:
            return _optimize_query_with_llm(question=cleaned_question, llm=llm)
        except Exception:
            return optimize_query_rule_based(cleaned_question)

    return optimize_query_rule_based(cleaned_question)


def optimize_query_rule_based(question: str) -> OptimizedQuery:
    """Optimización determinística por reglas."""
    cleaned_question = _clean_question(question)

    if not cleaned_question:
        raise ValueError("La pregunta no puede estar vacía.")

    normalized_text = _normalize_for_matching(cleaned_question)

    intent = _detect_intent(normalized_text)
    operation = _detect_operation(
        normalized_text,
        intent=intent,
    )
    metrics = _detect_metrics(normalized_text)
    filters = _detect_filters(cleaned_question, normalized_text)
    date_range = _detect_date_range(normalized_text)
    group_by = _detect_group_by(normalized_text)
    context, suggested_tables = _detect_context_and_tables(
        normalized_text=normalized_text,
        metrics=metrics,
        group_by=group_by,
    )
    normalized_question = _build_normalized_question(
        question=cleaned_question,
        intent=intent,
        metrics=metrics,
        group_by=group_by,
        context=context,
    )

    return OptimizedQuery(
        original_question=cleaned_question,
        normalized_question=normalized_question,
        intent=intent,
        operation=operation,
        metrics=metrics,
        filters=filters,
        date_range=date_range,
        group_by=group_by,
        context=context,
        suggested_tables=suggested_tables,
        optimizer="rule_based",
    )


def _optimize_query_with_llm(
    *,
    question: str,
    llm: Any,
) -> OptimizedQuery:
    prompt = _build_optimizer_prompt(question)

    structured_llm = llm.with_structured_output(_OptimizerPayload)
    payload: _OptimizerPayload = structured_llm.invoke(prompt)

    return _optimized_query_from_payload(
        original_question=question,
        payload=payload.model_dump(),
    )


def _build_optimizer_prompt(question: str) -> str:
    return f"""
You are a query optimizer for a Text-to-SQL system.

Normalize the user's business question before SQL generation.
The DDL schema descriptions used for retrieval are written in Spanish, so
"normalized_question" must always be written in Spanish, regardless of the
language of the user's original question.

Return only valid JSON with this structure:
{{
  "normalized_question": "clear rewritten question, always in Spanish",
  "intent": "ranking | count | aggregation | temporal_trend | comparison | detail",
  "metrics": ["metric names"],
  "filters": [
    {{"field": "field_name", "operator": "=", "value": "value"}}
  ],
  "date_range": {{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}},
  "group_by": ["month", "state", "category", "seller", "carrier"],
  "context": ["business context labels"],
  "suggested_tables": ["table names"]
}}

Use these metric names when applicable:
revenue, order_count, on_time_rate, freight_value, review_score,
stock_qty, reorder_point, returns_count, refund_amount,
incidents_count, compensation_value, ticket_count,
resolution_time_hr, satisfaction_score, price.

Use these table names when applicable:
olist_orders_dataset, olist_order_items_dataset, olist_customers_dataset,
olist_products_dataset, olist_order_reviews_dataset, carriers,
warehouse_inventory, product_returns, delivery_incidents,
customer_support_tickets, product_price_history,
product_category_name_translation.

Important rules:
- Add a filter only when the user states an explicit business constraint.
- Never infer Brazilian state codes from common words or substrings.
- If the question has no explicit filter, return an empty filters list.
- Write context labels in Spanish and keep them short and stable.
- Do not translate context labels to English.

User question:
{question}
"""


def _optimized_query_from_payload(
    *,
    original_question: str,
    payload: dict[str, Any],
) -> OptimizedQuery:
    fallback = optimize_query_rule_based(original_question)

    intent = str(payload.get("intent") or fallback.intent)

    if intent not in ALLOWED_INTENTS:
        intent = fallback.intent

    # Las consultas explícitamente temporales detectadas por las reglas
    # deben conservar una intención estable aunque el LLM las clasifique
    # como una agregación genérica. Esto evita que paráfrasis equivalentes
    # como suma mensual y promedio mensual cambien de intent.
    if fallback.intent == "temporal_trend":
        intent = "temporal_trend"

    # La operación participa en la compatibilidad de Query Memory.
    # Se recalcula de forma determinística usando la intención final
    # validada y canonicalizada.
    operation = _detect_operation(
        _normalize_for_matching(original_question),
        intent=intent,
    )

    # Las métricas canónicas detectadas por reglas deben prevalecer
    # sobre etiquetas variables o incorrectas generadas por el LLM.
    metrics = fallback.metrics or _ensure_text_list(payload.get("metrics"))
    group_by = _ensure_text_list(payload.get("group_by")) or fallback.group_by

    # Estos campos forman parte de la compatibilidad estructural de Query
    # Memory. Cuando las reglas reconocen el dominio, sus valores canónicos
    # son más estables que etiquetas libres generadas por el LLM.
    context = fallback.context or _ensure_text_list(payload.get("context"))
    suggested_tables = (
        fallback.suggested_tables
        or _ensure_text_list(payload.get("suggested_tables"))
    )

    # Un filtro incorrecto cambia el significado de la consulta y puede
    # impedir una recuperación válida. Solo se conservan filtros detectados
    # explícitamente por las reglas determinísticas.
    filters = fallback.filters
    date_range = _build_date_range_from_payload(
        payload.get("date_range"),
        fallback.date_range,
    )

    normalized_question = str(
        payload.get("normalized_question") or fallback.normalized_question
    ).strip()

    if not normalized_question:
        normalized_question = fallback.normalized_question

    return OptimizedQuery(
        original_question=original_question,
        normalized_question=normalized_question,
        intent=intent,
        operation=operation,
        metrics=_unique(metrics),
        filters=filters,
        date_range=date_range,
        group_by=_unique(group_by),
        context=_unique(context),
        suggested_tables=_unique(suggested_tables),
        optimizer="gemini",
    )


def _build_filters_from_payload(value: Any) -> list[QueryFilter]:
    if not isinstance(value, list):
        return []

    filters = []

    for item in value:
        if not isinstance(item, dict):
            continue

        field = str(item.get("field") or "").strip()
        operator = str(item.get("operator") or "=").strip()
        filter_value = str(item.get("value") or "").strip()

        if field and filter_value:
            filters.append(
                QueryFilter(
                    field=field,
                    operator=operator,
                    value=filter_value,
                )
            )

    return filters


def _build_date_range_from_payload(
    value: Any,
    fallback: dict[str, str] | None,
) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return fallback

    start_date = str(value.get("start_date") or "").strip()
    end_date = str(value.get("end_date") or "").strip()

    if not start_date or not end_date:
        return fallback

    return {
        "start_date": start_date,
        "end_date": end_date,
    }


def _ensure_text_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []

    if not isinstance(value, list):
        return []

    items = []

    for item in value:
        if item is None:
            continue

        cleaned_item = str(item).strip()

        if cleaned_item:
            items.append(cleaned_item)

    return items


def _clean_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_for_matching(value: str) -> str:
    return _strip_accents(value).lower()


def _contains_any(value: str, keywords: list[str]) -> bool:
    return any(keyword in value for keyword in keywords)


def _unique(values: list[str]) -> list[str]:
    unique_values = []

    for value in values:
        if value not in unique_values:
            unique_values.append(value)

    return unique_values


def _detect_operation(
    normalized_text: str,
    *,
    intent: str,
) -> str:
    """Detecta la operación de negocio de forma determinística."""
    # Una consulta cuya intención canónica es contar siempre representa
    # COUNT, aunque contenga expresiones como "número total".
    if intent == "count":
        return "count"

    has_average_reference = "promedio" in normalized_text
    has_proximity_reference = any(
        token in normalized_text
        for token in (
            "cerca",
            "cercan",
            "proxim",
        )
    )

    if has_average_reference and has_proximity_reference:
        return "rank_nearest_average"

    if (
        intent == "comparison"
        or any(
            token in normalized_text
            for token in (
                "comparar",
                "comparacion",
                "versus",
                " vs ",
            )
        )
    ):
        return "compare"

    if intent == "ranking" and _contains_any(
        normalized_text,
        [
            "mayor",
            "mejor",
            "mas alto",
            "mas alta",
            "mas cumplido",
            "mas cumplida",
            "mas puntual",
            "de mayor a menor",
            "top",
        ],
    ):
        return "rank_desc"

    if intent == "ranking" and _contains_any(
        normalized_text,
        [
            "menor",
            "peor",
            "mas bajo",
            "mas baja",
            "menos cumplido",
            "menos cumplida",
            "menos puntual",
            "de menor a mayor",
        ],
    ):
        return "rank_asc"

    if "mediana" in normalized_text:
        return "median"

    if _contains_any(
        normalized_text,
        [
            "promedio",
            "media aritmetica",
            "valor medio",
            "average",
        ],
    ):
        return "average"

    if _contains_any(
        normalized_text,
        [
            "cuantos",
            "cuantas",
            "cantidad",
            "numero de",
            "contar",
            "conteo",
        ],
    ):
        return "count"

    if _contains_any(
        normalized_text,
        [
            "total",
            "suma",
            "sumar",
            "acumulado",
        ],
    ):
        return "sum"

    return "detail"


def _detect_intent(normalized_text: str) -> str:
    temporal_tokens = set(normalized_text.split())

    if (
        _contains_any(
            normalized_text,
            [
                "por mes",
                "mensual",
                "mes a mes",
                "tendencia",
            ],
        )
        or "evolucion" in temporal_tokens
        or "evoluciones" in temporal_tokens
    ):
        return "temporal_trend"

    if _contains_any(
        normalized_text,
        ["mayor", "menor", "mejor", "peor", "top", "ranking", "mas ", "menos "],
    ):
        return "ranking"

    if any(
        token in normalized_text
        for token in (
            "comparar",
            "comparacion",
            "versus",
            " vs ",
        )
    ):
        return "comparison"

    if _contains_any(
        normalized_text,
        [
            "cuantos",
            "cuantas",
            "cantidad",
            "numero de",
            "numero total de",
        ],
    ):
        return "count"

    if _contains_any(normalized_text, ["total", "promedio", "suma", "monto"]):
        return "aggregation"

    return "detail"


def _detect_metrics(normalized_text: str) -> list[str]:
    metrics = []

    if _contains_any(normalized_text, ["venta", "vendido", "ingreso", "facturacion"]):
        metrics.append("revenue")

    if _contains_any(normalized_text, ["orden", "pedido", "compra"]):
        metrics.append("order_count")

    if _contains_any(
        normalized_text,
        ["cumplimiento", "puntualidad", "a tiempo", "entrega puntual"],
    ):
        metrics.append("on_time_rate")

    if _contains_any(normalized_text, ["flete", "freight", "envio"]):
        metrics.append("freight_value")

    if _contains_any(normalized_text, ["resena", "calificacion", "puntaje"]):
        metrics.append("review_score")

    if _contains_any(normalized_text, ["stock", "inventario", "reorden"]):
        metrics.extend(["stock_qty", "reorder_point"])

    if _contains_any(normalized_text, ["devolucion", "devoluciones", "reembolso"]):
        metrics.extend(["returns_count", "refund_amount"])

    if _contains_any(normalized_text, ["incidencia", "incidente", "compensacion"]):
        metrics.extend(["incidents_count", "compensation_value"])

    is_support_query = _contains_any(
        normalized_text,
        [
            "ticket",
            "soporte",
            "atencion al cliente",
            "reclamo",
        ],
    )

    if is_support_query:
        metrics.append("ticket_count")

    if _contains_any(
        normalized_text,
        [
            "tiempo de resolucion",
            "tiempo de respuesta",
            "demora en resolver",
        ],
    ):
        metrics.append("resolution_time_hr")

    if "satisfaccion" in normalized_text:
        metrics.append("satisfaction_score")

    if _contains_any(normalized_text, ["precio", "price"]):
        metrics.append("price")

    return _unique(metrics)


def _detect_group_by(normalized_text: str) -> list[str]:
    group_by = []

    if _contains_any(normalized_text, ["por mes", "mensual", "mes a mes"]):
        group_by.append("month")

    if "por estado" in normalized_text:
        group_by.append("state")

    if "por categoria" in normalized_text:
        group_by.append("category")

    if "por vendedor" in normalized_text:
        group_by.append("seller")

    if _contains_any(normalized_text, ["por transportista", "por carrier"]):
        group_by.append("carrier")

    return group_by


def _detect_date_range(normalized_text: str) -> dict[str, str] | None:
    years = [int(year) for year in re.findall(r"\b(20\d{2})\b", normalized_text)]

    if not years:
        return None

    start_year = min(years)
    end_year = max(years)

    return {
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
    }


def _detect_filters(original_question: str, normalized_text: str) -> list[QueryFilter]:
    filters = []

    state_codes = [
        "AC",
        "AL",
        "AP",
        "AM",
        "BA",
        "CE",
        "DF",
        "ES",
        "GO",
        "MA",
        "MT",
        "MS",
        "MG",
        "PA",
        "PB",
        "PR",
        "PE",
        "PI",
        "RJ",
        "RN",
        "RS",
        "RO",
        "RR",
        "SC",
        "SP",
        "SE",
        "TO",
    ]

    # Algunos códigos de estado coinciden con palabras habituales:
    # AL, ES y SE. No debemos convertir palabras españolas en filtros.
    ambiguous_state_codes = {"AL", "ES", "SE"}

    for state_code in state_codes:
        explicit_state_reference = re.search(
            (
                rf"\b(?:"
                rf"en\s+(?:el\s+)?estado(?:\s+de)?"
                rf"|estado(?:\s+de)?"
                rf"|uf"
                rf"|en"
                rf"|de"
                rf"|para"
                rf")\s+{state_code}\b"
            ),
            original_question,
            flags=re.IGNORECASE,
        )

        standalone_uppercase_code = (
            state_code not in ambiguous_state_codes
            and re.search(
                rf"\b{state_code}\b",
                original_question,
            )
        )

        if explicit_state_reference or standalone_uppercase_code:
            filters.append(
                QueryFilter(
                    "state",
                    "=",
                    state_code,
                )
            )

    if "cancelad" in normalized_text:
        filters.append(QueryFilter("order_status", "=", "canceled"))

    if "entregad" in normalized_text:
        filters.append(QueryFilter("order_status", "=", "delivered"))

    if "tarjeta" in normalized_text or "credit_card" in normalized_text:
        filters.append(QueryFilter("payment_type", "=", "credit_card"))

    if "boleto" in normalized_text:
        filters.append(QueryFilter("payment_type", "=", "boleto"))

    if "voucher" in normalized_text:
        filters.append(QueryFilter("payment_type", "=", "voucher"))

    if "critica" in normalized_text or "critico" in normalized_text:
        filters.append(QueryFilter("priority", "=", "critica"))

    is_support_query = _contains_any(
        normalized_text,
        [
            "ticket",
            "soporte",
            "atencion al cliente",
            "reclamo",
        ],
    )

    unresolved_support_status = _contains_any(
        normalized_text,
        [
            "sin resolver",
            "no resuelt",
            "abiert",
            "pendient",
        ],
    )
    resolved_support_status = _contains_any(
        normalized_text,
        [
            "resuelt",
            "cerrad",
        ],
    )

    if is_support_query:
        if unresolved_support_status:
            filters.append(
                QueryFilter(
                    "resolved",
                    "=",
                    "false",
                )
            )
        elif resolved_support_status:
            filters.append(
                QueryFilter(
                    "resolved",
                    "=",
                    "true",
                )
            )

    return filters


def _detect_context_and_tables(
    *,
    normalized_text: str,
    metrics: list[str],
    group_by: list[str],
) -> tuple[list[str], list[str]]:
    context = []
    tables = []

    if (
        "on_time_rate" in metrics
        or _contains_any(normalized_text, ["transportista", "carrier"])
    ):
        context.extend(["logistica", "transportistas"])
        tables.extend(["carriers", "olist_order_items_dataset"])

    if "revenue" in metrics or "order_count" in metrics:
        context.append("ventas")
        tables.extend(["olist_orders_dataset", "olist_order_items_dataset"])

    if "freight_value" in metrics:
        context.append("logistica")
        tables.extend(["olist_order_items_dataset", "carriers"])

    if "review_score" in metrics:
        context.append("resenas")
        tables.append("olist_order_reviews_dataset")

    if "stock_qty" in metrics or "reorder_point" in metrics:
        context.append("inventario")
        tables.extend(["warehouse_inventory", "olist_products_dataset"])

    if "returns_count" in metrics or "refund_amount" in metrics:
        context.append("devoluciones")
        tables.append("product_returns")

    if "incidents_count" in metrics or "compensation_value" in metrics:
        context.append("incidencias")
        tables.append("delivery_incidents")

    support_metrics = {
        "ticket_count",
        "resolution_time_hr",
        "satisfaction_score",
    }

    if support_metrics.intersection(metrics):
        context.append("soporte")
        tables.append("customer_support_tickets")

    if "price" in metrics:
        context.append("precios")
        tables.append("product_price_history")

    if "category" in group_by:
        tables.extend(
            [
                "olist_products_dataset",
                "product_category_name_translation",
            ]
        )

    if "state" in group_by:
        tables.append("olist_customers_dataset")

    return _unique(context), _unique(tables)


def _build_normalized_question(
    *,
    question: str,
    intent: str,
    metrics: list[str],
    group_by: list[str],
    context: list[str],
) -> str:
    normalized = _normalize_for_matching(question)

    replacements = {
        "empresa de transporte": "transportista",
        "empresas de transporte": "transportistas",
        "mejor cumplimiento": "mayor tasa de cumplimiento",
        "cumplen mejor": "mayor tasa de cumplimiento",
        "total vendido": "ventas totales",
        "quiero ver": "",
        "muestrame": "",
        "dame": "",
        "necesito saber": "",
    }

    for old_value, new_value in replacements.items():
        normalized = normalized.replace(old_value, new_value)

    normalized = _clean_question(normalized)

    if intent == "ranking" and "on_time_rate" in metrics:
        return "Listar transportistas ordenados por mayor tasa de cumplimiento de entrega."

    if "revenue" in metrics and "month" in group_by:
        return "Calcular ventas totales agrupadas por mes."

    if "inventario" in context and "reorder_point" in metrics:
        return "Listar productos con stock por debajo del punto de reorden."

    return normalized
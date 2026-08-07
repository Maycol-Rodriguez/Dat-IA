import pytest

from app.context.business_rules import match_business_rules
from app.optimizer.query_optimizer import (
    _build_optimizer_prompt,
    _optimized_query_from_payload,
    optimize_query,
    optimize_query_rule_based,
)


class _BoundFakeOptimizerLlm:
    """Simula el runnable devuelto por llm.with_structured_output(schema)."""

    def __init__(self, payload: dict, schema) -> None:
        self.payload = payload
        self.schema = schema

    def invoke(self, prompt: str):
        _ = prompt
        return self.schema(**self.payload)


class FakeOptimizerLlm:
    """Simula ChatGoogleGenerativeAI() antes de aplicar with_structured_output."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def with_structured_output(self, schema):
        return _BoundFakeOptimizerLlm(self.payload, schema)


class BrokenOptimizerLlm:
    """Simula un LLM que falla al invocarse (indisponible, error de red, etc.)."""

    def with_structured_output(self, schema):
        _ = schema
        return self

    def invoke(self, prompt: str):
        _ = prompt
        raise RuntimeError("LLM no disponible")


def test_optimizer_prompt_lists_the_sixteen_official_tables() -> None:
    prompt = _build_optimizer_prompt("Pregunta de prueba")
    catalog_text = prompt.split(
        "Use these table names when applicable:",
        maxsplit=1,
    )[1].split("Important rules:", maxsplit=1)[0]
    catalog = {
        table.strip().removesuffix(".")
        for table in catalog_text.replace("\n", " ").split(",")
        if table.strip()
    }

    assert catalog == {
        "carriers",
        "customer_support_tickets",
        "delivery_incidents",
        "olist_customers_dataset",
        "olist_geolocation_dataset",
        "olist_order_items_dataset",
        "olist_order_payments_dataset",
        "olist_order_reviews_dataset",
        "olist_orders_dataset",
        "olist_products_dataset",
        "olist_sellers_dataset",
        "product_category_name_translation",
        "product_price_history",
        "product_returns",
        "seller_promotions",
        "warehouse_inventory",
    }


def test_rule_based_optimizer_detects_carrier_ranking_query() -> None:
    result = optimize_query_rule_based(
        "Que transportista tiene la mayor tasa de cumplimiento?"
    )

    assert result.intent == "ranking"
    assert result.metrics == ["on_time_rate"]
    assert "logistica" in result.context
    assert "transportistas" in result.context
    assert "carriers" in result.suggested_tables
    assert result.optimizer == "rule_based"
    assert result.normalized_question == (
        "Listar transportistas ordenados por mayor tasa de cumplimiento de entrega."
    )


def test_rule_based_optimizer_detects_monthly_sales_query_with_year() -> None:
    result = optimize_query_rule_based("Cual fue el total vendido por mes en 2018?")

    assert result.intent == "temporal_trend"
    assert "revenue" in result.metrics
    assert result.group_by == ["month"]
    assert result.date_range == {
        "start_date": "2018-01-01",
        "end_date": "2018-12-31",
    }
    assert "olist_orders_dataset" in result.suggested_tables
    assert "olist_order_items_dataset" in result.suggested_tables


def test_rule_based_optimizer_detects_inventory_reorder_query() -> None:
    result = optimize_query_rule_based(
        "Que productos tienen stock critico bajo el punto de reorden?"
    )

    assert "stock_qty" in result.metrics
    assert "reorder_point" in result.metrics
    assert "inventario" in result.context
    assert "warehouse_inventory" in result.suggested_tables
    assert result.normalized_question == (
        "Listar productos con stock por debajo del punto de reorden."
    )


def test_rule_based_optimizer_detects_filters() -> None:
    result = optimize_query_rule_based(
        "Cuantas ordenes canceladas hubo en SP con tarjeta?"
    )

    filters = result.to_dict()["filters"]

    assert {"field": "state", "operator": "=", "value": "SP"} in filters
    assert {"field": "order_status", "operator": "=", "value": "canceled"} in filters
    assert {"field": "payment_type", "operator": "=", "value": "credit_card"} in filters


def test_optimizer_uses_llm_when_available() -> None:
    payload = {
        "normalized_question": "Calcular ventas totales agrupadas por estado.",
        "intent": "aggregation",
        "metrics": ["revenue"],
        "filters": [],
        "date_range": None,
        "group_by": ["state"],
        "context": ["ventas"],
        "suggested_tables": [
            "olist_orders_dataset",
            "olist_order_items_dataset",
            "olist_customers_dataset",
        ],
    }
    llm = FakeOptimizerLlm(payload)

    result = optimize_query("Dame ventas por estado", llm=llm)

    assert result.optimizer == "gemini"
    assert result.normalized_question == "Calcular ventas totales agrupadas por estado."
    assert result.intent == "aggregation"
    assert result.metrics == ["revenue"]
    assert result.group_by == ["state"]


def test_llm_optimizer_preserves_temporal_intent_for_monthly_average() -> None:
    payload = {
        "normalized_question": (
            "Calcula el promedio de ingresos "
            "mensuales durante el año 2018."
        ),
        "intent": "aggregation",
        "metrics": ["revenue"],
        "filters": [],
        "date_range": {
            "start_date": "2018-01-01",
            "end_date": "2018-12-31",
        },
        "group_by": ["month"],
        "context": ["ventas"],
        "suggested_tables": [
            "olist_orders_dataset",
            "olist_order_items_dataset",
        ],
    }

    result = optimize_query(
        (
            "¿Cuál fue el promedio vendido "
            "por mes en 2018?"
        ),
        llm=FakeOptimizerLlm(payload),
    )

    assert result.optimizer == "gemini"
    assert result.intent == "temporal_trend"
    assert result.operation == "average"
    assert result.metrics == ["revenue"]
    assert result.group_by == ["month"]
    assert result.date_range == {
        "start_date": "2018-01-01",
        "end_date": "2018-12-31",
    }


def test_optimizer_includes_category_tables_when_llm_corrects_group_by() -> None:
    """La pregunta no contiene el literal "por categoria" que exige
    `_detect_group_by`, así que las reglas solas dejan `group_by=[]` y
    `suggested_tables=['product_price_history']` (sin las tablas de
    categoría). Si el LLM corrige el group_by a "category", las tablas
    sugeridas deben recalcularse con ese group_by final, no quedarse con
    las que el fallback aislado ya había fijado antes de la corrección.
    """
    payload = {
        "normalized_question": (
            "Identificar la categoria de producto con el precio "
            "promedio mas bajo."
        ),
        "intent": "ranking",
        "operation": "rank_asc",
        "metrics": ["price"],
        "filters": [],
        "date_range": None,
        "group_by": ["category"],
        "context": [],
        "suggested_tables": [],
    }

    result = optimize_query(
        "¿Qué categoría de producto tiene el precio promedio más bajo?",
        llm=FakeOptimizerLlm(payload),
    )

    assert result.group_by == ["category"]
    assert result.suggested_tables == [
        "product_price_history",
        "olist_products_dataset",
        "product_category_name_translation",
    ]


def test_optimizer_falls_back_to_rules_when_llm_fails() -> None:
    result = optimize_query(
        "Que transportista tiene la mayor tasa de cumplimiento?",
        llm=BrokenOptimizerLlm(),
    )

    assert result.optimizer == "rule_based"
    assert result.intent == "ranking"
    assert result.metrics == ["on_time_rate"]


def test_optimizer_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="pregunta no puede estar vacía"):
        optimize_query("   ")


def test_optimizer_discards_ungrounded_llm_filter() -> None:
    payload = {
        "normalized_question": (
            "Identifica la empresa transportista con el mayor "
            "índice de entregas a tiempo."
        ),
        "intent": "ranking",
        "metrics": ["on_time_rate"],
        "filters": [
            {
                "field": "state",
                "operator": "=",
                "value": "es",
            }
        ],
        "date_range": None,
        "group_by": ["carrier"],
        "context": [
            "logistics",
            "performance",
        ],
        "suggested_tables": ["carriers"],
    }

    result = optimize_query(
        (
            "Indica cuál es la empresa transportista con el "
            "mejor índice de entregas a tiempo."
        ),
        llm=FakeOptimizerLlm(payload),
    )

    assert result.filters == []
    assert result.context == [
        "logistica",
        "transportistas",
    ]
    assert result.suggested_tables == [
        "carriers",
        "olist_order_items_dataset",
    ]


def test_optimizer_preserves_only_explicit_rule_based_filters() -> None:
    payload = {
        "normalized_question": (
            "Contar las órdenes canceladas pagadas con tarjeta "
            "en el estado indicado."
        ),
        "intent": "count",
        "metrics": ["order_count"],
        "filters": [
            {
                "field": "state",
                "operator": "=",
                "value": "ES",
            }
        ],
        "date_range": None,
        "group_by": [],
        "context": ["orders"],
        "suggested_tables": ["olist_orders_dataset"],
    }

    result = optimize_query(
        "Cuántas órdenes canceladas hubo en SP con tarjeta?",
        llm=FakeOptimizerLlm(payload),
    )

    assert result.to_dict()["filters"] == [
        {
            "field": "state",
            "operator": "=",
            "value": "SP",
        },
        {
            "field": "order_status",
            "operator": "=",
            "value": "canceled",
        },
        {
            "field": "payment_type",
            "operator": "=",
            "value": "credit_card",
        },
    ]


@pytest.mark.parametrize(
    ("question", "expected_resolved"),
    [
        (
            "¿Cuántos tickets de soporte están sin resolver?",
            "false",
        ),
        (
            "¿Cuántos reclamos de soporte siguen sin resolver?",
            "false",
        ),
        (
            (
                "¿Cuántos reclamos de atención al cliente "
                "siguen abiertos?"
            ),
            "false",
        ),
        (
            "¿Cuántos tickets de soporte ya fueron resueltos?",
            "true",
        ),
    ],
)
def test_rule_based_optimizer_canonicalizes_support_status(
    question: str,
    expected_resolved: str,
) -> None:
    result = optimize_query_rule_based(question)

    assert result.intent == "count"
    assert result.metrics == ["ticket_count"]
    assert result.to_dict()["filters"] == [
        {
            "field": "resolved",
            "operator": "=",
            "value": expected_resolved,
        }
    ]
    assert result.context == ["soporte"]
    assert result.suggested_tables == [
        "customer_support_tickets",
    ]


def test_optimizer_prefers_llm_metric_over_rules_when_valid() -> None:
    """El LLM gana sobre las reglas cuando propone un metric del catálogo
    canónico (ALLOWED_METRICS), aunque las reglas hubieran elegido otro.
    Antes las reglas ganaban siempre; se invirtió la precedencia porque el
    detector por reglas tiene colisiones conocidas (ver query_optimizer.py)
    y el LLM es más confiable para desambiguar vocabulario de negocio,
    siempre que el valor propuesto esté dentro del catálogo cerrado.
    """
    payload = {
        "normalized_question": (
            "Contar los tickets de soporte sin resolver."
        ),
        "intent": "count",
        "metrics": ["incidents_count"],
        "filters": [],
        "date_range": None,
        "group_by": [],
        "context": [
            "customer_service",
            "ticket_status",
        ],
        "suggested_tables": [
            "customer_support_tickets",
        ],
    }

    result = optimize_query(
        "¿Cuántos tickets de soporte están sin resolver?",
        llm=FakeOptimizerLlm(payload),
    )

    assert result.metrics == ["incidents_count"]
    assert result.to_dict()["filters"] == [
        {
            "field": "resolved",
            "operator": "=",
            "value": "false",
        }
    ]
    # context/suggested_tables se recalculan a partir del metric final
    # ("incidents_count"), no del texto crudo de la pregunta: por eso
    # apuntan al dominio de incidencias y no al de soporte, aunque la
    # pregunta hable de "tickets". Es el comportamiento correcto: si el
    # metric ganador es de otro dominio, las tablas deben seguir a la
    # métrica, no al fallback aislado.
    assert result.context == ["incidencias"]
    # Pero el filtro "resolved" sigue viniendo de las reglas (ver el
    # comentario de CLAUDE.md sobre filters) y `delivery_incidents` no
    # tiene una columna `resolved` — solo `resolved_date`/`resolution_type`.
    # Sin `customer_support_tickets`, el filtro quedaría sin ninguna tabla
    # donde aplicarse. _tables_required_by_filters agrega la tabla del
    # filtro sin desplazar la que ya trajo el metric ganador.
    assert result.suggested_tables == [
        "delivery_incidents",
        "customer_support_tickets",
    ]


def test_optimizer_falls_back_to_rules_when_llm_metric_is_not_canonical() -> None:
    """Si el LLM propone un metric fuera de ALLOWED_METRICS (alucinado o
    con nombre libre), las reglas siguen siendo el respaldo real.
    """
    payload = {
        "normalized_question": (
            "Contar los tickets de soporte sin resolver."
        ),
        "intent": "count",
        "metrics": ["support_ticket_backlog"],
        "filters": [],
        "date_range": None,
        "group_by": [],
        "context": [],
        "suggested_tables": [],
    }

    result = optimize_query(
        "¿Cuántos tickets de soporte están sin resolver?",
        llm=FakeOptimizerLlm(payload),
    )

    assert result.metrics == ["ticket_count"]


def test_pending_orders_are_not_support_tickets() -> None:
    result = optimize_query_rule_based(
        "¿Cuántos pedidos pendientes existen?"
    )

    assert result.metrics == ["order_count"]
    assert result.filters == []


@pytest.mark.parametrize(
    ("question", "expected_operation"),
    [
        (
            "\u00bfCu\u00e1ntos tickets de soporte "
            "est\u00e1n sin resolver?",
            "count",
        ),
        (
            "\u00bfQu\u00e9 transportista tiene "
            "mayor cumplimiento?",
            "rank_desc",
        ),
        (
            "\u00bfQu\u00e9 transportista tiene "
            "menor cumplimiento?",
            "rank_asc",
        ),
        (
            (
                "\u00bfQu\u00e9 transportistas est\u00e1n "
                "m\u00e1s cerca del cumplimiento promedio?"
            ),
            "rank_nearest_average",
        ),
        (
            (
                "\u00bfCu\u00e1l fue la facturaci\u00f3n total "
                "mensual en SP durante 2018?"
            ),
            "sum",
        ),
        (
            (
                "\u00bfCu\u00e1l fue la facturaci\u00f3n promedio "
                "mensual en SP durante 2018?"
            ),
            "average",
        ),
        (
            (
                "\u00bfCu\u00e1l fue la mediana mensual "
                "de ventas en SP durante 2018?"
            ),
            "median",
        ),
        (
            "Comparar ventas versus devoluciones.",
            "compare",
        ),
        (
            "Listar productos disponibles.",
            "detail",
        ),
    ],
)
def test_rule_based_optimizer_detects_canonical_operation(
    question: str,
    expected_operation: str,
) -> None:
    result = optimize_query_rule_based(question)

    assert result.operation == expected_operation
    assert result.to_dict()["operation"] == expected_operation


def test_optimizer_uses_deterministic_operation_with_llm() -> None:
    payload = {
        "normalized_question": (
            "Listar transportistas por cumplimiento."
        ),
        "intent": "ranking",
        "metrics": ["on_time_rate"],
        "filters": [],
        "date_range": None,
        "group_by": [],
        "context": ["logistica"],
        "suggested_tables": ["carriers"],
    }

    result = optimize_query(
        "\u00bfQu\u00e9 transportista tiene "
        "menor cumplimiento?",
        llm=FakeOptimizerLlm(payload),
    )

    assert result.operation == "rank_asc"


def test_rule_based_optimizer_distinguishes_explicit_ranking_direction() -> None:
    descending = optimize_query_rule_based(
        "Lista los transportistas de mayor a menor cumplimiento."
    )
    ascending = optimize_query_rule_based(
        "Lista los transportistas de menor a mayor cumplimiento."
    )

    assert descending.intent == "ranking"
    assert descending.operation == "rank_desc"

    assert ascending.intent == "ranking"
    assert ascending.operation == "rank_asc"

    assert descending.metrics == ascending.metrics
    assert descending.context == ascending.context
    assert descending.suggested_tables == ascending.suggested_tables


def test_rule_based_optimizer_detects_comparison_intent() -> None:
    result = optimize_query_rule_based(
        "Comparar ventas versus devoluciones."
    )

    assert result.intent == "comparison"
    assert result.operation == "compare"


def test_devoluciones_does_not_trigger_temporal_intent() -> None:
    result = optimize_query_rule_based(
        "\u00bfCu\u00e1ntas devoluciones se registraron?"
    )

    assert result.intent == "count"
    assert result.operation == "count"


def test_count_intent_takes_precedence_over_total_wording() -> None:
    result = optimize_query_rule_based(
        "\u00bfCu\u00e1l es el n\u00famero total de tickets de "
        "atenci\u00f3n al cliente que tienen un estado abierto?"
    )

    assert result.intent == "count"
    assert result.operation == "count"
    assert result.metrics == ["ticket_count"]
    assert len(result.filters) == 1
    assert result.filters[0].field == "resolved"
    assert result.filters[0].operator == "="
    assert result.filters[0].value == "false"


def test_payload_optimizer_prefers_llm_operation_when_valid() -> None:
    """El LLM gana sobre `_detect_operation` cuando propone un valor dentro
    de ALLOWED_OPERATIONS, aunque contradiga lo que las reglas habrían
    calculado a partir del intent final. Antes `operation` siempre se
    recalculaba por reglas y el LLM nunca podía influir en este campo.
    """
    result = _optimized_query_from_payload(
        original_question="Tickets de soporte abiertos.",
        payload={
            "intent": "count",
            "normalized_question": (
                "Contar los tickets de soporte abiertos."
            ),
            "operation": "sum",
        },
    )

    assert result.optimizer == "gemini"
    assert result.intent == "count"
    assert result.operation == "sum"
    assert result.metrics == ["ticket_count"]
    assert len(result.filters) == 1
    assert result.filters[0].field == "resolved"
    assert result.filters[0].operator == "="
    assert result.filters[0].value == "false"


def test_payload_optimizer_falls_back_to_rules_when_llm_operation_invalid() -> None:
    """Si el LLM no propone `operation` o propone un valor fuera del
    catálogo, `_detect_operation` sigue siendo el respaldo real.
    """
    result = _optimized_query_from_payload(
        original_question="Tickets de soporte abiertos.",
        payload={
            "intent": "count",
            "normalized_question": (
                "Contar los tickets de soporte abiertos."
            ),
            "operation": "not_a_real_operation",
        },
    )

    assert result.operation == "count"


# ---------------------------------------------------------------------------
# Regresión: tablas que las reglas globales de negocio (business_rules.json)
# deben forzar en `suggested_tables` aunque el vocabulario de la pregunta
# nunca las mencione. Reproduce, sin LLM ni Chroma, el hueco de recuperación
# medido en reports/dat_ia_golden_set_v2_latest.md para golden_013/014/016/
# 021/024/028: la búsqueda semántica sola nunca las trae porque no hay
# nada en la pregunta que se les parezca (nadie dice "traducción" al
# preguntar por categorías).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("question", "required_table"),
    [
        (
            "¿Cuáles son los 5 estados con más órdenes entregadas?",
            "olist_customers_dataset",
        ),
        (
            "¿Cuáles son las 5 categorías con más ítems vendidos en órdenes entregadas?",
            "olist_products_dataset",
        ),
        (
            "¿Cuáles son las 5 categorías con más ítems vendidos en órdenes entregadas?",
            "product_category_name_translation",
        ),
        (
            "¿Cuáles son los 5 estados con mayor flete promedio por ítem entregado?",
            "olist_customers_dataset",
        ),
        (
            "¿Cuántos compradores realizaron más de una orden entregada?",
            "olist_customers_dataset",
        ),
        (
            "¿Cuáles son las 5 categorías de producto con más devoluciones?",
            "olist_products_dataset",
        ),
        (
            "¿Cuáles son las 5 categorías de producto con más devoluciones?",
            "product_category_name_translation",
        ),
        (
            "¿Cuáles son los 5 estados de vendedores con más registros de "
            "inventario bajo el punto de reorden?",
            "olist_sellers_dataset",
        ),
    ],
)
def test_business_rules_force_required_tables_missed_by_keyword_rules(
    question: str,
    required_table: str,
) -> None:
    result = optimize_query_rule_based(question)

    assert required_table in result.suggested_tables


def test_business_rules_do_not_add_customers_when_seller_state_applies() -> None:
    """golden_028: "estados de vendedores" debe traer sellers, no customers
    (tablas_alternativas reemplaza, no suma, cuando el patrón coincide)."""
    result = optimize_query_rule_based(
        "¿Cuáles son los 5 estados de vendedores con más registros de "
        "inventario bajo el punto de reorden?"
    )

    assert "olist_sellers_dataset" in result.suggested_tables
    assert "olist_customers_dataset" not in result.suggested_tables


def test_business_rules_do_not_trigger_on_support_ticket_categories() -> None:
    """golden_018: "categoría" de tickets de soporte no debe activar la
    regla `categorias_producto` (falso positivo detectado durante el
    diseño de la regla). Se verifica contra `match_business_rules`
    directamente: la vía preexistente y no relacionada que también
    arrastraba `product_category_name_translation` (`_detect_group_by`
    matcheando el literal "por categoria" dentro de "tickets hay por
    categoría") se corrigió aparte en
    `test_category_group_by_does_not_pull_product_tables_for_support_tickets`,
    más abajo — acá solo se comprueba el módulo de reglas de negocio en
    aislamiento."""
    matched = match_business_rules(
        "¿Cuántos tickets hay por categoría y qué porcentaje fue resuelto?"
    )

    assert not any(rule.id == "categorias_producto" for rule in matched)


# ---------------------------------------------------------------------------
# Punto 2: pregunta original vs. normalizada para generación, unión (no
# override) de suggested_tables en el camino LLM, tablas requeridas por
# filtro, y el bug de "category" en group_by que se descubrió al validar
# el punto anterior en la misma función.
# ---------------------------------------------------------------------------


def test_category_group_by_does_not_pull_product_tables_for_support_tickets() -> None:
    """golden_018: el literal "por categoria" dentro de "tickets hay por
    categoría" seguía disparando group_by=["category"], y
    `_detect_context_and_tables` traducía eso en
    `product_category_name_translation` sin que la pregunta tuviera nada
    que ver con productos. El group_by se conserva (es una señal válida de
    agrupación sobre la columna `category` de customer_support_tickets);
    solo se corrige qué tablas dispara."""
    result = optimize_query_rule_based(
        "¿Cuántos tickets hay por categoría y qué porcentaje fue resuelto?"
    )

    assert result.group_by == ["category"]
    assert result.suggested_tables == ["customer_support_tickets"]
    assert "product_category_name_translation" not in result.suggested_tables
    assert "olist_products_dataset" not in result.suggested_tables


def test_rule_based_optimizer_still_pulls_product_tables_for_real_categories() -> None:
    """Contraprueba de la anterior: una pregunta que sí habla de categorías
    de producto (sin mención de tickets/soporte) debe seguir trayendo las
    tablas de categoría — el guard no debe apagar el caso legítimo."""
    result = optimize_query_rule_based(
        "¿Cuál fue el ingreso total por categoría de producto?"
    )

    assert "olist_products_dataset" in result.suggested_tables
    assert "product_category_name_translation" in result.suggested_tables


def test_llm_path_unions_suggested_tables_instead_of_overriding() -> None:
    """golden_022: antes, si las reglas encontraban CUALQUIER tabla (aunque
    incompleta), las sugerencias del LLM se descartaban enteras vía `or`.
    Acá las reglas locales solo encuentran las tablas de reseñas/categoría
    (por las métricas "review_score" detectadas); la tabla de ítems —que
    conecta reseñas con categorías— solo la propuso el LLM."""
    result = _optimized_query_from_payload(
        original_question=(
            "¿Cuáles son las 5 categorías con mejor calificación promedio, "
            "considerando al menos 100 reseñas?"
        ),
        payload={
            "intent": "ranking",
            "operation": "rank_desc",
            "normalized_question": (
                "categorias con mejor calificacion promedio con al menos "
                "100 resenas"
            ),
            "suggested_tables": [
                "olist_order_items_dataset",
                "olist_products_dataset",
                "olist_order_reviews_dataset",
            ],
        },
    )

    assert "olist_order_items_dataset" in result.suggested_tables
    assert "olist_products_dataset" in result.suggested_tables
    assert "olist_order_reviews_dataset" in result.suggested_tables


def test_optimized_query_from_payload_adds_table_required_by_filter() -> None:
    """Mismo mecanismo que en el camino por reglas, pero pasando por
    `_optimized_query_from_payload`: `filters` siempre viene de
    `fallback.filters` (las reglas determinísticas), nunca del payload del
    LLM, así que la tabla que ese filtro requiere debe sumarse igual aunque
    el LLM no la haya propuesto."""
    result = _optimized_query_from_payload(
        original_question="¿Cuántas órdenes hay en SP?",
        payload={
            "intent": "count",
            "normalized_question": "cuantas ordenes hay en sp",
            "suggested_tables": ["olist_orders_dataset"],
        },
    )

    assert result.filters and result.filters[0].field == "state"
    assert "olist_customers_dataset" in result.suggested_tables


@pytest.mark.parametrize(
    ("question", "required_table"),
    [
        ("¿Cuántas órdenes hay en SP?", "olist_customers_dataset"),
        ("¿Cuántas órdenes fueron canceladas?", "olist_orders_dataset"),
        (
            "¿Cuántas órdenes se pagaron con tarjeta de crédito?",
            "olist_order_payments_dataset",
        ),
        (
            "¿Cuántos tickets de soporte permanecen sin resolver?",
            "customer_support_tickets",
        ),
        (
            "¿Cuántos tickets tienen prioridad crítica?",
            "customer_support_tickets",
        ),
    ],
)
def test_rule_based_optimizer_adds_table_required_by_filter(
    question: str,
    required_table: str,
) -> None:
    """`_detect_filters` ya detecta estos filtros; lo nuevo es que la tabla
    donde se aplican queda garantizada en `suggested_tables` aunque el
    vocabulario de la pregunta no la sugiera por otra vía (ej. "en SP" sin
    la palabra "estado" no activa la regla global `geografia_estado`)."""
    result = optimize_query_rule_based(question)

    assert result.filters, "la pregunta debería producir al menos un filtro"
    assert required_table in result.suggested_tables


def test_generate_validated_sql_receives_original_question_not_normalized(
    monkeypatch,
) -> None:
    """query_answer/query_json deben pasar `original_question` al generador,
    no `normalized_question`: una paráfrasis del optimizer puede desviar la
    intención (golden_008). Se prueba a nivel de API en test_api.py; acá se
    deja constancia de que `OptimizedQuery` distingue ambos campos y que
    nada dentro del optimizer los confunde."""
    result = optimize_query_rule_based("¿Cuántos tickets de soporte permanecen sin resolver?")

    assert result.original_question == (
        "¿Cuántos tickets de soporte permanecen sin resolver?"
    )
    assert result.normalized_question != result.original_question

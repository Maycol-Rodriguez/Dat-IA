import pytest

from app.optimizer.query_optimizer import (
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


def test_optimizer_prefers_canonical_support_metric() -> None:
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

    assert result.metrics == ["ticket_count"]
    assert result.to_dict()["filters"] == [
        {
            "field": "resolved",
            "operator": "=",
            "value": "false",
        }
    ]
    assert result.context == ["soporte"]
    assert result.suggested_tables == [
        "customer_support_tickets",
    ]


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


def test_payload_optimizer_recomputes_operation_from_final_intent() -> None:
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
    assert result.operation == "count"
    assert result.metrics == ["ticket_count"]
    assert len(result.filters) == 1
    assert result.filters[0].field == "resolved"
    assert result.filters[0].operator == "="
    assert result.filters[0].value == "false"

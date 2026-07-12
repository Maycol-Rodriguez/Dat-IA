import pytest

from app.optimizer.query_optimizer import optimize_query, optimize_query_rule_based


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

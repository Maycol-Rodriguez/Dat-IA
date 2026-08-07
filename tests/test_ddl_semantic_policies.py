from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_DDL_PATH = REPOSITORY_ROOT / "data" / "ddl.json"
DEPRECATED_DDL_PATH = REPOSITORY_ROOT / "data" / "ddl_old.json"


def _load_by_table(path: Path) -> dict[str, dict]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {entry["nombre"]: entry for entry in entries}


def _load_entries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_semantic_ddl_preserves_the_sixteen_official_structures() -> None:
    active = _load_by_table(ACTIVE_DDL_PATH)
    deprecated = _load_by_table(DEPRECATED_DDL_PATH)

    assert len(active) == 16
    assert set(active) == set(deprecated)
    assert {
        table: entry["ddl"]
        for table, entry in active.items()
    } == {
        table: entry["ddl"]
        for table, entry in deprecated.items()
    }


def test_active_and_deprecated_ddl_use_contiguous_identifiers() -> None:
    expected_ids = [f"tabla_{index}" for index in range(1, 17)]

    assert [
        entry["id"] for entry in _load_entries(ACTIVE_DDL_PATH)
    ] == expected_ids
    assert [
        entry["id"] for entry in _load_entries(DEPRECATED_DDL_PATH)
    ] == expected_ids


def test_active_ddl_descripcion_is_free_of_policy_text() -> None:
    """`descripcion` es lo único que se vectoriza: las políticas viven en
    `politicas` desde que se separaron, para no mover los embeddings de
    retrieval cada vez que se agrega una regla de negocio nueva."""
    active = _load_by_table(ACTIVE_DDL_PATH)

    assert all(
        "POLÍTICA SEMÁNTICA" not in entry["descripcion"]
        for entry in active.values()
    )


def test_active_ddl_contains_the_required_business_policies() -> None:
    active = _load_by_table(ACTIVE_DDL_PATH)

    def policies_text(table: str) -> str:
        return " ".join(active[table]["politicas"])

    assert "order_purchase_timestamp como fecha predeterminada" in policies_text(
        "olist_orders_dataset"
    )
    assert "COUNT(DISTINCT order_id)" in policies_text("olist_orders_dataset")
    assert "SUM(price)" in policies_text("olist_order_items_dataset")
    assert "excluye freight_value" in policies_text("olist_order_items_dataset")
    assert "customer_unique_id" in policies_text("olist_customers_dataset")
    assert "sumar units_sold_during" in policies_text("seller_promotions")
    # Corregida respecto a la redacción original ("solo cuando se
    # solicite"): el golden set espera traducción por defecto cuando existe.
    assert "mostrar product_category_name_english cuando exista" in policies_text(
        "olist_products_dataset"
    )


def test_active_ddl_contains_the_new_business_policies() -> None:
    """Políticas agregadas para cubrir vacíos de conocimiento que el LLM no
    puede deducir del DDL (golden_008, golden_012, golden_028)."""
    active = _load_by_table(ACTIVE_DDL_PATH)

    def policies_text(table: str) -> str:
        return " ".join(active[table]["politicas"])

    assert "resolved = FALSE" in policies_text("customer_support_tickets")
    assert "no tiene una columna status" in policies_text(
        "customer_support_tickets"
    )
    assert (
        "order_delivered_customer_date > order_estimated_delivery_date"
        in policies_text("olist_orders_dataset")
    )
    assert "stock_qty < reorder_point" in policies_text("warehouse_inventory")


def test_deprecated_ddl_remains_a_policy_free_snapshot() -> None:
    deprecated = _load_by_table(DEPRECATED_DDL_PATH)

    assert all(
        "POLÍTICA SEMÁNTICA" not in entry["descripcion"]
        for entry in deprecated.values()
    )

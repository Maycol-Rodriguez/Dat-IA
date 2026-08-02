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


def test_active_ddl_contains_the_required_business_policies() -> None:
    active = _load_by_table(ACTIVE_DDL_PATH)

    assert "order_purchase_timestamp como fecha predeterminada" in active[
        "olist_orders_dataset"
    ]["descripcion"]
    assert "COUNT(DISTINCT order_id)" in active[
        "olist_orders_dataset"
    ]["descripcion"]
    assert "SUM(price)" in active["olist_order_items_dataset"]["descripcion"]
    assert "excluye freight_value" in active[
        "olist_order_items_dataset"
    ]["descripcion"]
    assert "customer_unique_id" in active[
        "olist_customers_dataset"
    ]["descripcion"]
    assert "sumar units_sold_during" in active[
        "seller_promotions"
    ]["descripcion"]
    assert "salida requiera categorías en inglés" in active[
        "olist_products_dataset"
    ]["descripcion"]


def test_deprecated_ddl_remains_a_policy_free_snapshot() -> None:
    deprecated = _load_by_table(DEPRECATED_DDL_PATH)

    assert all(
        "POLÍTICA SEMÁNTICA" not in entry["descripcion"]
        for entry in deprecated.values()
    )

from __future__ import annotations

from scripts.refresh_ddl_from_database import (
    DEFAULT_REPORT_PATH,
    REPOSITORY_ROOT,
    _build_corrected_entries,
)


def test_build_corrected_entries_reassigns_contiguous_ids() -> None:
    legacy_entries = [
        {
            "id": "tabla_1",
            "nombre": "first_table",
            "descripcion": "Primera tabla.",
            "ddl": "CREATE TABLE first_table (\nid integer\n);",
        },
        {
            "id": "tabla_3",
            "nombre": "second_table",
            "descripcion": "Segunda tabla.",
            "ddl": "CREATE TABLE second_table (\nid integer\n);",
        },
    ]
    database_schema = {
        "first_table": {"columns": [{"name": "id", "type": "integer"}]},
        "second_table": {"columns": [{"name": "id", "type": "integer"}]},
    }

    corrected = _build_corrected_entries(
        legacy_entries=legacy_entries,
        database_schema=database_schema,
        catalogs={},
    )

    assert [entry["id"] for entry in corrected] == ["tabla_1", "tabla_2"]
    assert [entry["nombre"] for entry in corrected] == [
        "first_table",
        "second_table",
    ]


def test_ddl_audit_defaults_to_the_ignored_archive() -> None:
    assert DEFAULT_REPORT_PATH == (
        REPOSITORY_ROOT
        / "reports"
        / "archive"
        / "dat_ia_ddl_validation.json"
    )

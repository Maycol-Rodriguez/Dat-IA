"""Audita el DDL y genera una versión compatible con la base PostgreSQL actual."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text

from app.db.connect_db import create_db_engine

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DDL_PATH = REPOSITORY_ROOT / "data" / "ddl.json"
DEFAULT_BACKUP_PATH = REPOSITORY_ROOT / "data" / "ddl_old.json"
DEFAULT_REPORT_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "archive"
    / "dat_ia_ddl_validation.json"
)

_COLUMN_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+(.+?)(?:,\s*)?(?:\s+--\s*(.*))?$"
)
_TYPE_PREFIX_PATTERN = re.compile(
    r"^(.+?)(?=\s+(?:DEFAULT|GENERATED|NOT NULL|NULL|PRIMARY KEY|REFERENCES|"
    r"UNIQUE|CHECK)\b|$)",
    flags=re.IGNORECASE,
)
_CATALOG_COLUMNS = {
    "delivery_incidents": ("incident_type", "resolution_type"),
    "customer_support_tickets": ("category", "priority"),
    "carriers": ("carrier_type",),
    "product_price_history": ("change_reason",),
    "seller_promotions": ("promo_type",),
    "product_returns": ("return_reason", "refund_method"),
    "olist_orders_dataset": ("order_status",),
    "olist_order_payments_dataset": ("payment_type",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara data/ddl.json con PostgreSQL, conserva ddl_old.json y "
            "regenera el DDL usando la estructura y catálogos actuales."
        )
    )
    parser.add_argument("--ddl-path", type=Path, default=DEFAULT_DDL_PATH)
    parser.add_argument("--backup-path", type=Path, default=DEFAULT_BACKUP_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Genera solamente el reporte, sin copiar ni reemplazar el DDL.",
    )
    return parser.parse_args()


def refresh_ddl(
    database_url: str,
    *,
    ddl_path: Path = DEFAULT_DDL_PATH,
    backup_path: Path = DEFAULT_BACKUP_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    audit_only: bool = False,
) -> dict[str, Any]:
    legacy_entries = json.loads(ddl_path.read_text(encoding="utf-8"))
    legacy_by_table = {entry["nombre"]: entry for entry in legacy_entries}
    legacy_schema = {
        table: _parse_legacy_ddl(entry["ddl"])
        for table, entry in legacy_by_table.items()
    }
    engine = create_db_engine(database_url)

    try:
        inspector = inspect(engine)
        database_tables = sorted(inspector.get_table_names(schema="public"))
        database_schema = {
            table: _inspect_table(inspector, table)
            for table in database_tables
        }

        with engine.connect() as connection:
            with connection.begin():
                if connection.dialect.name == "postgresql":
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                catalogs = _read_catalog_values(connection, database_tables)
    finally:
        engine.dispose()

    report = _build_report(
        legacy_schema=legacy_schema,
        database_schema=database_schema,
        catalogs=catalogs,
    )
    _write_json(report_path, report)

    if audit_only:
        return report

    if not backup_path.exists():
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ddl_path, backup_path)
    corrected_entries = _build_corrected_entries(
        legacy_entries=legacy_entries,
        database_schema=database_schema,
        catalogs=catalogs,
    )
    _write_json(ddl_path, corrected_entries)
    return report


def _inspect_table(inspector: Any, table: str) -> dict[str, Any]:
    columns = []

    for column in inspector.get_columns(table, schema="public"):
        columns.append(
            {
                "name": column["name"],
                "type": str(column["type"]),
                "nullable": bool(column.get("nullable", True)),
                "default": column.get("default"),
                "identity": column.get("identity"),
            }
        )

    return {
        "columns": columns,
        "primary_key": list(
            inspector.get_pk_constraint(table, schema="public").get(
                "constrained_columns"
            )
            or []
        ),
        "foreign_keys": [
            {
                "columns": list(foreign_key.get("constrained_columns") or []),
                "referred_table": foreign_key.get("referred_table"),
                "referred_columns": list(
                    foreign_key.get("referred_columns") or []
                ),
            }
            for foreign_key in inspector.get_foreign_keys(
                table,
                schema="public",
            )
        ],
        "unique_constraints": [
            list(constraint.get("column_names") or [])
            for constraint in inspector.get_unique_constraints(
                table,
                schema="public",
            )
            if constraint.get("column_names")
        ],
    }


def _parse_legacy_ddl(ddl: str) -> dict[str, Any]:
    columns: dict[str, dict[str, str | None]] = {}
    body = ddl.partition("(")[2].rpartition(")")[0]

    for raw_line in body.splitlines():
        line = raw_line.strip().rstrip(",")

        if not line or line.upper().startswith(
            ("PRIMARY KEY", "FOREIGN KEY", "UNIQUE", "CONSTRAINT", "CHECK")
        ):
            continue

        match = _COLUMN_PATTERN.match(line)

        if match is None:
            continue

        declaration = match.group(2).strip()
        type_match = _TYPE_PREFIX_PATTERN.match(declaration)
        columns[match.group(1)] = {
            "type": type_match.group(1).strip() if type_match else declaration,
            "comment": match.group(3),
        }

    return {"columns": columns}


def _read_catalog_values(
    connection: Any,
    database_tables: list[str],
) -> dict[str, dict[str, list[Any]]]:
    catalogs: dict[str, dict[str, list[Any]]] = {}
    available = set(database_tables)

    for table, columns in _CATALOG_COLUMNS.items():
        if table not in available:
            continue

        table_catalogs: dict[str, list[Any]] = {}

        for column in columns:
            statement = text(
                f'SELECT DISTINCT "{column}" AS value '
                f'FROM "public"."{table}" '
                f'WHERE "{column}" IS NOT NULL ORDER BY value'
            )
            table_catalogs[column] = [
                row.value
                for row in connection.execute(statement)
            ]

        catalogs[table] = table_catalogs

    return catalogs


def _build_report(
    *,
    legacy_schema: Mapping[str, Any],
    database_schema: Mapping[str, Any],
    catalogs: Mapping[str, Any],
) -> dict[str, Any]:
    legacy_tables = set(legacy_schema)
    database_tables = set(database_schema)
    table_reports = []

    for table in sorted(legacy_tables & database_tables):
        documented_columns = set(legacy_schema[table]["columns"])
        actual_columns = {
            column["name"]
            for column in database_schema[table]["columns"]
        }
        type_differences = []
        catalog_value_differences = []

        for column in sorted(documented_columns & actual_columns):
            documented_type = legacy_schema[table]["columns"][column]["type"]
            actual_type = next(
                item["type"]
                for item in database_schema[table]["columns"]
                if item["name"] == column
            )

            if _normalize_type(str(documented_type)) != _normalize_type(actual_type):
                type_differences.append(
                    {
                        "column": column,
                        "documented": documented_type,
                        "actual": actual_type,
                    }
                )

        for column, actual_values in catalogs.get(table, {}).items():
            legacy_comment = (
                legacy_schema[table]["columns"]
                .get(column, {})
                .get("comment")
            )
            documented_values = _extract_documented_catalog_values(
                legacy_comment
            )
            normalized_actual = sorted(str(value) for value in actual_values)

            if documented_values and documented_values != normalized_actual:
                catalog_value_differences.append(
                    {
                        "column": column,
                        "documented": documented_values,
                        "actual": normalized_actual,
                    }
                )

        table_reports.append(
            {
                "table": table,
                "missing_columns_in_database": sorted(
                    documented_columns - actual_columns
                ),
                "undocumented_columns_in_ddl": sorted(
                    actual_columns - documented_columns
                ),
                "type_differences": type_differences,
                "catalog_value_differences": catalog_value_differences,
                "observed_catalog_values": catalogs.get(table, {}),
            }
        )

    return {
        "database_dialect": "postgresql",
        "legacy_table_count": len(legacy_tables),
        "database_table_count": len(database_tables),
        "missing_tables_in_database": sorted(legacy_tables - database_tables),
        "undocumented_tables_in_ddl": sorted(database_tables - legacy_tables),
        "tables": table_reports,
    }


def _build_corrected_entries(
    *,
    legacy_entries: list[dict[str, Any]],
    database_schema: Mapping[str, Any],
    catalogs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    legacy_by_table = {entry["nombre"]: entry for entry in legacy_entries}
    legacy_comments = {
        table: _parse_legacy_ddl(entry["ddl"])["columns"]
        for table, entry in legacy_by_table.items()
    }
    legacy_schema = {
        table: _parse_legacy_ddl(entry["ddl"])
        for table, entry in legacy_by_table.items()
    }
    corrected = []

    ordered_tables = [
        entry["nombre"]
        for entry in legacy_entries
        if entry["nombre"] in database_schema
    ]
    ordered_tables.extend(
        sorted(set(database_schema) - set(ordered_tables))
    )

    for table_index, table in enumerate(ordered_tables, start=1):
        legacy = legacy_by_table.get(table)
        entry_id = f"tabla_{table_index}"

        if legacy is None:
            description = (
                f"Tabla {table}. Su estructura fue obtenida directamente "
                "de la base PostgreSQL."
            )
        else:
            description = legacy["descripcion"].partition(
                " CATÁLOGOS REALES OBSERVADOS EN LA BASE ACTUAL"
            )[0]

        table_catalogs = catalogs.get(table, {})

        if table_catalogs:
            catalog_text = "; ".join(
                f"{column}={values}"
                for column, values in table_catalogs.items()
            )
            description = (
                f"{description} CATÁLOGOS REALES OBSERVADOS EN LA BASE ACTUAL "
                f"(usar estos valores al filtrar): {catalog_text}."
            )

        if legacy is not None and _same_columns_and_types(
            legacy_schema[table],
            database_schema[table],
        ):
            corrected_ddl = _refresh_catalog_comments(
                legacy["ddl"],
                table_catalogs,
            )
        else:
            corrected_ddl = _render_create_table(
                table,
                database_schema[table],
                comments=legacy_comments.get(table, {}),
                catalogs=table_catalogs,
            )

        corrected.append(
            {
                "id": entry_id,
                "nombre": table,
                "descripcion": description,
                "ddl": corrected_ddl,
            }
        )

    return corrected


def _same_columns_and_types(
    legacy_schema: Mapping[str, Any],
    database_schema: Mapping[str, Any],
) -> bool:
    legacy_columns = legacy_schema["columns"]
    actual_columns = {
        column["name"]: column
        for column in database_schema["columns"]
    }

    if set(legacy_columns) != set(actual_columns):
        return False

    return all(
        _normalize_type(str(legacy_columns[name]["type"]))
        == _normalize_type(str(actual_columns[name]["type"]))
        for name in legacy_columns
    )


def _refresh_catalog_comments(
    ddl: str,
    catalogs: Mapping[str, list[Any]],
) -> str:
    if not catalogs:
        return ddl

    refreshed_lines = []

    for raw_line in ddl.splitlines():
        stripped = raw_line.strip()
        column = next(
            (
                candidate
                for candidate in catalogs
                if re.match(rf"^{re.escape(candidate)}\s+", stripped)
            ),
            None,
        )

        if column is None:
            refreshed_lines.append(raw_line)
            continue

        declaration = raw_line.split("--", maxsplit=1)[0].rstrip()
        values = " | ".join(str(value) for value in catalogs[column])
        refreshed_lines.append(
            f"{declaration} -- Valores reales observados en la base actual: "
            f"{values}"
        )

    return "\n".join(refreshed_lines)


def _render_create_table(
    table: str,
    schema: Mapping[str, Any],
    *,
    comments: Mapping[str, Any],
    catalogs: Mapping[str, list[Any]],
) -> str:
    primary_key = schema["primary_key"]
    single_primary_key = primary_key[0] if len(primary_key) == 1 else None
    foreign_keys = {
        foreign_key["columns"][0]: foreign_key
        for foreign_key in schema["foreign_keys"]
        if len(foreign_key["columns"]) == 1
    }
    unique_single_columns = {
        columns[0]
        for columns in schema["unique_constraints"]
        if len(columns) == 1
    }
    declarations = []

    for column in schema["columns"]:
        name = column["name"]
        declaration = f"  {name} {_render_type(column)}"

        if not column["nullable"]:
            declaration += " NOT NULL"

        if column["default"] is not None and not column["identity"]:
            declaration += f" DEFAULT {column['default']}"

        if name == single_primary_key:
            declaration += " PRIMARY KEY"

        if name in unique_single_columns:
            declaration += " UNIQUE"

        foreign_key = foreign_keys.get(name)

        if foreign_key is not None:
            referred_columns = ", ".join(foreign_key["referred_columns"])
            declaration += (
                f" REFERENCES {foreign_key['referred_table']} "
                f"({referred_columns})"
            )

        comment = _column_comment(
            name,
            legacy_comment=(comments.get(name) or {}).get("comment"),
            catalog_values=catalogs.get(name),
        )

        if comment:
            declaration += f" -- {comment}"

        declarations.append(declaration)

    if len(primary_key) > 1:
        declarations.append(f"  PRIMARY KEY ({', '.join(primary_key)})")

    for columns in schema["unique_constraints"]:
        if len(columns) > 1:
            declarations.append(f"  UNIQUE ({', '.join(columns)})")

    return (
        f"CREATE TABLE {table} (\n"
        + ",\n".join(declarations)
        + "\n);"
    )


def _render_type(column: Mapping[str, Any]) -> str:
    identity = column.get("identity")

    if identity:
        generation = str(identity.get("generation") or "BY DEFAULT").upper()
        return f"{column['type']} GENERATED {generation} AS IDENTITY"

    return str(column["type"])


def _column_comment(
    column: str,
    *,
    legacy_comment: str | None,
    catalog_values: list[Any] | None,
) -> str:
    if catalog_values is not None:
        values = " | ".join(str(value) for value in catalog_values)
        return f"Valores reales observados en la base actual: {values}"

    if legacy_comment:
        return legacy_comment.strip()

    return f"Columna {column} según el esquema PostgreSQL actual"


def _normalize_type(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().casefold())
    normalized = re.sub(r"\s*,\s*", ",", normalized)
    normalized = re.sub(r"\(\s*", "(", normalized)
    normalized = re.sub(r"\s*\)", ")", normalized)
    aliases = {
        "timestamp": "timestamp without time zone",
        "decimal": "numeric",
        "int": "integer",
        "int4": "integer",
        "int8": "bigint",
        "float8": "double precision",
        "bool": "boolean",
    }
    return aliases.get(normalized, normalized)


def _extract_documented_catalog_values(
    comment: str | None,
) -> list[str]:
    if not comment or ":" not in comment or "|" not in comment:
        return []

    catalog_text = comment.split(":", maxsplit=1)[1]
    values = []

    for item in catalog_text.split("|"):
        match = re.match(r"\s*([A-Za-z0-9_]+)", item)

        if match is not None:
            values.append(match.group(1))

    return sorted(set(values))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise SystemExit(
            "DATABASE_URL no está definida. Usa `uv run --env-file .env "
            "python -m scripts.refresh_ddl_from_database`."
        )

    report = refresh_ddl(
        database_url,
        ddl_path=args.ddl_path,
        backup_path=args.backup_path,
        report_path=args.report_path,
        audit_only=args.audit_only,
    )
    structural_issues = sum(
        bool(item["missing_columns_in_database"])
        + bool(item["undocumented_columns_in_ddl"])
        + len(item["type_differences"])
        for item in report["tables"]
    )
    print(
        json.dumps(
            {
                "missing_tables_in_database": report[
                    "missing_tables_in_database"
                ],
                "undocumented_tables_in_ddl": report[
                    "undocumented_tables_in_ddl"
                ],
                "structural_issues": structural_issues,
                "audit_only": args.audit_only,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Reporte: {args.report_path}")


if __name__ == "__main__":
    main()

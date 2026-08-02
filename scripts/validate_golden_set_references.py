"""Audita los SQL del golden set sin modificar el dataset canónico."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import uuid
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.db.connect_db import create_db_engine
from app.evaluation import (
    compare_result_facts,
    golden_set_content_hash,
    is_read_only_sql,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "evaluation"
    / "datasets"
    / "dat_ia_golden_set_v2.jsonl"
)
DEFAULT_REPORT_PATH = (
    REPOSITORY_ROOT
    / "reports"
    / "archive"
    / "dat_ia_golden_v2_reference_validation.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta en modo lectura los SQL de referencia del único golden "
            "set y genera un reporte para revisión, sin cambiar el dataset."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def validate_references(
    database_url: str,
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    cases = _load_jsonl(dataset_path)
    engine = create_db_engine(database_url)
    validations: list[dict[str, Any]] = []

    try:
        with engine.connect() as connection:
            with connection.begin():
                if connection.dialect.name == "postgresql":
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")

                for case in cases:
                    validations.append(_validate_case(connection, case))
    finally:
        engine.dispose()

    counts = {
        classification: sum(
            item["classification"] == classification
            for item in validations
        )
        for classification in (
            "correct",
            "golden_set_outdated",
            "reference_sql_error",
        )
    }
    report = {
        "dataset_version": cases[0]["metadata"]["dataset_version"],
        "dataset_content_sha256": golden_set_content_hash(cases),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "validation_scope": f"{len(cases)} reference SQL queries",
        "dataset_path": _path_for_report(dataset_path),
        "database_dialect": engine.dialect.name,
        "counts": counts,
        "correct_case_ids": [
            item["case_id"]
            for item in validations
            if item["classification"] == "correct"
        ],
        "review_case_ids": [
            item["case_id"]
            for item in validations
            if item["classification"] != "correct"
        ],
        "cases": validations,
    }

    _write_json(report_path, report)
    return report


def _path_for_report(path: Path) -> str:
    """Usa una ruta relativa al repo cuando sea posible, o absoluta si no."""
    resolved_path = path.resolve()

    try:
        return str(resolved_path.relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(resolved_path)


def _validate_case(connection: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    case_id = str(case["case_id"])
    reference = case["reference_outputs"]
    sql = str(reference["reference_sql"])
    expected = reference["expected_result"]
    context = {
        "case_id": case_id,
        "question": case["inputs"]["question"],
        "reference_sql": sql,
        "expected_sources": reference["expected_sources"],
    }

    if not is_read_only_sql(sql):
        return {
            **context,
            "classification": "reference_sql_error",
            "reason": "El SQL no cumple la política de solo lectura.",
            "expected_result": expected,
            "actual_result": None,
        }

    try:
        with connection.begin_nested():
            result = connection.execute(text(sql))
            actual_rows = [
                {key: _json_value(value) for key, value in row.items()}
                for row in result.mappings()
            ]
    except Exception as exc:  # El reporte debe conservar el error de PostgreSQL.
        return {
            **context,
            "classification": "reference_sql_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "expected_result": expected,
            "actual_result": None,
        }

    actual_result = {
        "row_count": len(actual_rows),
        "rows": actual_rows,
    }

    if compare_result_facts(actual_rows, expected):
        return {
            **context,
            "classification": "correct",
            "reason": "El SQL se ejecutó y el resultado completo coincide.",
            "expected_result": expected,
            "actual_result": actual_result,
        }

    return {
        **context,
        "classification": "golden_set_outdated",
        "reason": _mismatch_reason(actual_rows, expected),
        "expected_result": expected,
        "actual_result": actual_result,
    }


def _mismatch_reason(actual_rows: list[dict[str, Any]], expected: Mapping[str, Any]) -> str:
    expected_rows = expected.get("rows", [])

    if len(actual_rows) != len(expected_rows):
        return (
            "El SQL se ejecutó, pero la cantidad de filas difiere: "
            f"esperadas={len(expected_rows)}, actuales={len(actual_rows)}."
        )

    for index, (actual, expected_row) in enumerate(
        zip(actual_rows, expected_rows, strict=True),
        start=1,
    ):
        if actual != expected_row:
            return (
                f"El SQL se ejecutó, pero la fila {index} difiere. "
                f"Esperada={expected_row}; actual={actual}."
            )

    return "El SQL se ejecutó, pero los valores exceden la tolerancia configurada."


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)

    if isinstance(value, (dt.date, dt.datetime, dt.time)):
        return value.isoformat()

    if isinstance(value, uuid.UUID):
        return value.hex

    if isinstance(value, bytes):
        return value.hex()

    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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
            "DATABASE_URL no está definida. Ejecuta con `uv run --env-file "
            ".env python -m scripts.validate_golden_set_references` o "
            "expórtala en la consola."
        )

    report = validate_references(
        database_url,
        dataset_path=args.dataset_path,
        report_path=args.report_path,
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"Reporte: {args.report_path}")


if __name__ == "__main__":
    main()

"""Recalcula expected_result contra la BD y escribe una copia del golden set.

Nunca modifica `tests/evaluation/datasets/dat_ia_golden_set_v2.jsonl`: lee el
archivo canónico, ejecuta cada `reference_sql` en una transacción de solo
lectura y escribe el resultado en una copia aparte para que el equipo la
revise antes de reemplazar el archivo canónico a mano. La copia queda marcada
como candidata pendiente de promoción, aunque el baseline de origen ya hubiese
sido verificado.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import text

from app.db.connect_db import create_db_engine
from app.evaluation import is_read_only_sql
from scripts.validate_golden_set_references import _json_value

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "evaluation"
    / "datasets"
    / "dat_ia_golden_set_v2.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "evaluation"
    / "datasets"
    / "dat_ia_golden_set_v2_refreshed.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta cada reference_sql del golden set contra la BD y "
            "escribe una copia con expected_result recalculado."
        )
    )
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def refresh_expected_results(
    database_url: str,
    *,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    raw_lines = _load_raw_lines(dataset_path)
    engine = create_db_engine(database_url)
    refreshed_case_ids: list[str] = []
    skipped_cases: list[dict[str, str]] = []
    output_lines: list[str] = []

    try:
        with engine.connect() as connection:
            with connection.begin():
                if connection.dialect.name == "postgresql":
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")

                for raw_line in raw_lines:
                    case = json.loads(raw_line)
                    case_id = str(case["case_id"])
                    reference = case["reference_outputs"]
                    sql = str(reference["reference_sql"])

                    if not is_read_only_sql(sql):
                        skipped_cases.append(
                            {
                                "case_id": case_id,
                                "reason": "El SQL no cumple la política de solo lectura.",
                            }
                        )
                        output_lines.append(raw_line)
                        continue

                    try:
                        with connection.begin_nested():
                            result = connection.execute(text(sql))
                            actual_rows = [
                                {
                                    key: _json_value(value)
                                    for key, value in row.items()
                                }
                                for row in result.mappings()
                            ]
                    except Exception as exc:
                        skipped_cases.append(
                            {
                                "case_id": case_id,
                                "reason": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        output_lines.append(raw_line)
                        continue

                    reference["expected_result"]["row_count"] = len(actual_rows)
                    reference["expected_result"]["rows"] = actual_rows
                    case["metadata"]["quality_status"] = (
                        "candidate_pending_promotion"
                    )
                    case["metadata"]["quality_notes"] = [
                        "expected_result recalculado automáticamente contra "
                        "PostgreSQL; requiere revisión antes de promoverse."
                    ]
                    refreshed_case_ids.append(case_id)
                    output_lines.append(
                        json.dumps(case, ensure_ascii=False, separators=(",", ":"))
                    )
    finally:
        engine.dispose()

    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    return {
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "refreshed": refreshed_case_ids,
        "skipped_due_to_error": skipped_cases,
    }


def _load_raw_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as jsonl_file:
        return [line.rstrip("\n") for line in jsonl_file if line.strip()]


def main() -> None:
    args = parse_args()
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise SystemExit(
            "DATABASE_URL no está definida. Ejecuta con `uv run --env-file "
            ".env python -m scripts.refresh_golden_set_expected_results` o "
            "expórtala en la consola."
        )

    summary = refresh_expected_results(
        database_url,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
    )
    print(f"Copia escrita en: {summary['output_path']}")
    print(f"Casos recalculados: {len(summary['refreshed'])}")

    if summary["skipped_due_to_error"]:
        print("Casos NO actualizados en la copia (revisar reference_sql a mano):")
        for skipped in summary["skipped_due_to_error"]:
            print(f"  - {skipped['case_id']}: {skipped['reason']}")


if __name__ == "__main__":
    main()

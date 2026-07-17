from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.memory.query_memory_v2 import (
    QUERY_MEMORY_V2_DISTANCE_THRESHOLD,
    QueryMemoryV2Record,
    build_query_memory_v2_document,
    create_query_memory_v2_record,
    get_or_create_query_memory_v2_collection,
    search_query_memory_v2_for_record,
    upsert_query_memory_v2,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BANK = (
    ROOT
    / "tests"
    / "evaluation"
    / "query_memory_cases.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "query_memory_v2_real_benchmark.json"
)
DEFAULT_THRESHOLDS = (
    0.00,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.12,
    0.15,
    0.20,
    0.30,
    0.50,
    1.00,
    2.00,
)


def _load_google_api_key() -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()

    if api_key:
        return api_key

    env_path = ROOT / ".env"

    if env_path.exists():
        for raw_line in env_path.read_text(
            encoding="utf-8-sig",
        ).splitlines():
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line.removeprefix("export ").strip()

            if not line.startswith("GOOGLE_API_KEY="):
                continue

            api_key = line.split("=", 1)[1].strip()
            api_key = api_key.strip('"').strip("'")

            if api_key:
                return api_key

    raise SystemExit(
        "GOOGLE_API_KEY no está definida en el entorno "
        "ni en el archivo .env."
    )


def _load_bank(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def _record_from_memory(
    memory: dict[str, Any],
) -> QueryMemoryV2Record:
    return create_query_memory_v2_record(
        original_question=memory["original_question"],
        normalized_question=memory["normalized_question"],
        intent=memory["intent"],
        operation=memory["operation"],
        metrics=memory["metrics"],
        filters=memory["filters"],
        date_range=memory["date_range"],
        group_by=memory.get("group_by", []),
        context=memory["context"],
        sql=memory["sql"],
        sources=memory["sources"],
        status=memory.get("status", "success"),
        validated=memory["validated"],
        execution_status=memory["execution_status"],
        model="real-embedding-benchmark",
    )


def _record_from_case(
    case: dict[str, Any],
) -> QueryMemoryV2Record:
    return create_query_memory_v2_record(
        original_question=case["question"],
        normalized_question=case["normalized_question"],
        intent=case["intent"],
        operation=case["operation"],
        metrics=case["metrics"],
        filters=case["filters"],
        date_range=case["date_range"],
        group_by=case.get("group_by", []),
        context=case["context"],
        sql="",
        sources="",
        status="candidate",
        validated=False,
        execution_status="not_executed",
        model="real-embedding-benchmark-query",
    )


def _candidate_key(
    metadata: dict[str, Any],
    memory_key_by_id: dict[str, str],
) -> str | None:
    memory_id = str(
        metadata.get("memory_id") or ""
    ).strip()

    return memory_key_by_id.get(memory_id)


def _evaluate_threshold(
    case_reports: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    positive_count = sum(
        report["expected_memory"] is not None
        for report in case_reports
    )
    negative_count = len(case_reports) - positive_count

    hits_at_1 = 0
    hits_at_3 = 0
    false_positives = 0
    correct_candidates = 0
    wrong_candidates = 0
    cases_with_wrong_candidates = 0

    for report in case_reports:
        returned_keys = [
            candidate["key"]
            for candidate in report[
                "structured_candidates"
            ]
            if candidate["distance"] <= threshold
        ]

        expected_memory = report["expected_memory"]

        if expected_memory is None:
            if returned_keys:
                false_positives += 1
                wrong_candidates += len(returned_keys)
                cases_with_wrong_candidates += 1
            continue

        case_wrong_candidates = sum(
            key != expected_memory
            for key in returned_keys
        )
        correct_candidates += sum(
            key == expected_memory
            for key in returned_keys
        )
        wrong_candidates += case_wrong_candidates

        if case_wrong_candidates:
            cases_with_wrong_candidates += 1

        if (
            returned_keys
            and returned_keys[0] == expected_memory
        ):
            hits_at_1 += 1

        if expected_memory in returned_keys[:3]:
            hits_at_3 += 1

    recall_at_1 = (
        hits_at_1 / positive_count
        if positive_count
        else 0.0
    )
    recall_at_3 = (
        hits_at_3 / positive_count
        if positive_count
        else 0.0
    )
    false_positive_rate = (
        false_positives / negative_count
        if negative_count
        else 0.0
    )
    candidate_total = (
        correct_candidates + wrong_candidates
    )
    candidate_precision = (
        correct_candidates / candidate_total
        if candidate_total
        else 1.0
    )

    return {
        "threshold": threshold,
        "positive_cases": positive_count,
        "negative_cases": negative_count,
        "hits_at_1": hits_at_1,
        "hits_at_3": hits_at_3,
        "false_positives": false_positives,
        "correct_candidates": correct_candidates,
        "wrong_candidates": wrong_candidates,
        "cases_with_wrong_candidates": (
            cases_with_wrong_candidates
        ),
        "recall_at_1": recall_at_1,
        "recall_at_3": recall_at_3,
        "candidate_precision": candidate_precision,
        "false_positive_rate": false_positive_rate,
    }


def _calculate_semantic_margin(
    case_reports: list[dict[str, Any]],
) -> dict[str, float | None]:
    positive_distances: list[float] = []
    wrong_distances: list[float] = []

    for report in case_reports:
        expected_memory = report["expected_memory"]

        for candidate in report["structured_candidates"]:
            distance = float(candidate["distance"])

            if (
                expected_memory is not None
                and candidate["key"] == expected_memory
            ):
                positive_distances.append(distance)
            else:
                wrong_distances.append(distance)

    maximum_positive = (
        max(positive_distances)
        if positive_distances
        else None
    )
    minimum_wrong = (
        min(wrong_distances)
        if wrong_distances
        else None
    )

    if (
        maximum_positive is not None
        and minimum_wrong is not None
    ):
        margin = minimum_wrong - maximum_positive
        midpoint = (
            maximum_positive + minimum_wrong
        ) / 2
    else:
        margin = None
        midpoint = None

    return {
        "maximum_positive_distance": maximum_positive,
        "minimum_wrong_distance": minimum_wrong,
        "semantic_margin": margin,
        "midpoint_threshold": midpoint,
    }


def _recommend_threshold(
    threshold_metrics: list[dict[str, Any]],
    case_reports: list[dict[str, Any]],
) -> float:
    calibration = _calculate_semantic_margin(
        case_reports,
    )
    margin = calibration["semantic_margin"]
    midpoint = calibration["midpoint_threshold"]

    if (
        margin is not None
        and margin > 0
        and midpoint is not None
    ):
        return round(midpoint, 3)

    best_score = max(
        (
            metric["recall_at_1"],
            metric["candidate_precision"],
            metric["recall_at_3"],
            -metric["false_positive_rate"],
        )
        for metric in threshold_metrics
    )

    best_thresholds = [
        metric["threshold"]
        for metric in threshold_metrics
        if (
            metric["recall_at_1"],
            metric["candidate_precision"],
            metric["recall_at_3"],
            -metric["false_positive_rate"],
        )
        == best_score
    ]

    return min(best_thresholds)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa Query Memory V2 con embeddings "
            "reales de Gemini."
        )
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=DEFAULT_BANK,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--model",
        default="gemini-embedding-2",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_arguments()
    api_key = _load_google_api_key()
    bank = _load_bank(args.bank)

    thresholds = sorted(set(args.thresholds))

    if not thresholds:
        raise SystemExit(
            "Debe proporcionarse al menos un umbral."
        )

    if any(threshold < 0 for threshold in thresholds):
        raise SystemExit(
            "Los umbrales no pueden ser negativos."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=args.model,
        google_api_key=api_key,
    )

    with tempfile.TemporaryDirectory(
        prefix="datia-query-memory-v2-",
        ignore_cleanup_errors=True,
    ) as temporary_directory:
        chroma_client = chromadb.PersistentClient(
            path=temporary_directory,
        )
        collection = (
            get_or_create_query_memory_v2_collection(
                chroma_client,
                embeddings,
            )
        )

        memory_key_by_id: dict[str, str] = {}

        print(
            "Indexando memorias con "
            f"{args.model}..."
        )

        for memory in bank["memories"]:
            record = _record_from_memory(memory)
            stored_record = upsert_query_memory_v2(
                collection,
                record,
            )
            memory_key_by_id[
                stored_record.memory_id
            ] = memory["key"]

        print(
            "Memorias indexadas: "
            f"{collection._collection.count()}"
        )

        maximum_threshold = max(thresholds)
        raw_candidate_count = max(
            len(bank["memories"]),
            1,
        )
        case_reports: list[dict[str, Any]] = []

        for case in bank["cases"]:
            print(f"Evaluando caso: {case['id']}")

            query_record = _record_from_case(case)
            query_document = (
                build_query_memory_v2_document(
                    query_record,
                )
            )

            raw_results = (
                collection.similarity_search_with_score(
                    query_document,
                    k=raw_candidate_count,
                )
            )
            structured_results = (
                search_query_memory_v2_for_record(
                    collection,
                    query_record,
                    n_results=3,
                    distance_threshold=maximum_threshold,
                )
            )

            raw_candidates = [
                {
                    "key": _candidate_key(
                        document.metadata,
                        memory_key_by_id,
                    ),
                    "memory_id": str(
                        document.metadata.get(
                            "memory_id",
                            "",
                        )
                    ),
                    "validated": bool(
                        document.metadata.get(
                            "validated",
                            False,
                        )
                    ),
                    "distance": float(distance),
                }
                for document, distance in raw_results[:5]
            ]

            structured_candidates = [
                {
                    "key": _candidate_key(
                        result["metadata"],
                        memory_key_by_id,
                    ),
                    "memory_id": str(
                        result["metadata"].get(
                            "memory_id",
                            "",
                        )
                    ),
                    "distance": float(
                        result["distance"]
                    ),
                }
                for result in structured_results
            ]

            case_reports.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "normalized_question": (
                        case["normalized_question"]
                    ),
                    "expected_memory": (
                        case["expected_memory"]
                    ),
                    "raw_candidates": raw_candidates,
                    "structured_candidates": (
                        structured_candidates
                    ),
                }
            )

    threshold_metrics = [
        _evaluate_threshold(
            case_reports,
            threshold,
        )
        for threshold in thresholds
    ]
    calibration = _calculate_semantic_margin(
        case_reports,
    )
    recommended_threshold = _recommend_threshold(
        threshold_metrics,
        case_reports,
    )

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "scope": "post_optimizer_structured_retrieval",
        "model": args.model,
        "bank_version": bank.get("version"),
        "memory_count": len(bank["memories"]),
        "case_count": len(bank["cases"]),
        "current_production_threshold": (
            QUERY_MEMORY_V2_DISTANCE_THRESHOLD
        ),
        "recommended_threshold_for_this_bank": (
            recommended_threshold
        ),
        "semantic_calibration": calibration,
        "threshold_metrics": threshold_metrics,
        "cases": case_reports,
        "notes": [
            (
                "Este benchmark no evalúa la calidad "
                "del optimizer."
            ),
            (
                "Las búsquedas estructurales solo "
                "reutilizan memorias validadas."
            ),
            (
                "No modifica el umbral productivo "
                "automáticamente."
            ),
        ],
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Resultados por umbral")
    print(
        "umbral | recall@1 | recall@3 | precisión | "
        "candidatos incorrectos | falsos positivos"
    )

    for metric in threshold_metrics:
        print(
            f"{metric['threshold']:>7.3f} | "
            f"{metric['recall_at_1']:>8.3f} | "
            f"{metric['recall_at_3']:>8.3f} | "
            f"{metric['candidate_precision']:>9.3f} | "
            f"{metric['wrong_candidates']:>21} | "
            f"{metric['false_positive_rate']:>16.3f}"
        )

    print()
    print(
        "Umbral sugerido para este banco: "
        f"{recommended_threshold:.2f}"
    )
    print(f"Reporte generado: {args.output}")


if __name__ == "__main__":
    main()

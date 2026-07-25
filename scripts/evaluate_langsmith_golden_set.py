"""Ejecuta el golden set contra la API y publica el experimento en LangSmith."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.evaluation import (
    DEFAULT_GOLDEN_SET_PATH,
    GOLDEN_SET_DATASET_NAME,
    GOLDEN_SET_VERSION,
    answer_contains_expected_facts,
    generated_sql_is_read_only,
    golden_set_content_hash,
    load_golden_set,
    reported_source_tables_match_expected,
    response_status_matches_expected,
    result_facts_match_expected,
    sync_golden_set,
)
from app.observability import (
    get_langsmith_client,
    langsmith_project_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa end-to-end la API Dat-IA con LangSmith."
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="URL base de la API Dat-IA.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_GOLDEN_SET_PATH,
    )
    parser.add_argument(
        "--dataset-name",
        default=GOLDEN_SET_DATASET_NAME,
    )
    parser.add_argument(
        "--experiment-prefix",
        default=None,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Tiempo máximo por pregunta, en segundos.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Usa el dataset remoto existente sin sincronizarlo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida los 30 casos y /ready sin ejecutar las preguntas.",
    )
    return parser.parse_args()


def request_json(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return {
            "status": "http_error",
            "http_status": exc.code,
            "answer": response_body,
            "sql": "",
            "data": [],
            "sources": "",
        }


def check_api_ready(api_url: str, *, timeout: float) -> dict[str, Any]:
    ready = request_json(
        f"{api_url.rstrip('/')}/ready",
        timeout=timeout,
    )

    if ready.get("status") != "ok":
        raise RuntimeError(f"La API no está lista: {ready}")

    if ready.get("database") != "connected":
        raise RuntimeError(
            "La evaluación end-to-end requiere database='connected' en /ready."
        )

    return ready


def build_target(api_url: str, *, timeout: float):
    endpoint = f"{api_url.rstrip('/')}/query/answer"

    def target(inputs: dict[str, Any]) -> dict[str, Any]:
        started_at = time.perf_counter()
        output = request_json(
            endpoint,
            payload={"question": inputs["question"]},
            timeout=timeout,
        )
        output["latency_ms"] = round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )
        return output

    return target


def main() -> None:
    args = parse_args()

    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency debe ser mayor o igual que 1.")

    cases = load_golden_set(args.dataset_path)
    content_hash = golden_set_content_hash(cases)
    ready = check_api_ready(args.api_url, timeout=min(args.timeout, 10.0))
    print(
        f"Golden set válido: {len(cases)} casos. "
        f"API lista, base de datos: {ready['database']}."
    )

    if args.dry_run:
        return

    client = get_langsmith_client()

    if client is None:
        raise SystemExit(
            "LangSmith no está configurado. Define USE_LANGSMITH_TRACING=true "
            "y LANGSMITH_API_KEY."
        )

    if not args.skip_sync:
        sync_summary = sync_golden_set(
            client,
            cases=cases,
            dataset_name=args.dataset_name,
        )
        print(
            "Dataset sincronizado: "
            f"{sync_summary['created']} creados, "
            f"{sync_summary['updated']} actualizados, "
            f"{sync_summary['unchanged']} sin cambios."
        )

        if sync_summary["remote_only"]:
            raise SystemExit(
                "El dataset remoto contiene ejemplos que ya no están en Git. "
                "Revísalos con el script de sincronización y --prune antes "
                "de evaluar."
            )

    experiment_prefix = args.experiment_prefix or (
        f"{langsmith_project_name()}-golden-v2"
    )

    results = client.evaluate(
        build_target(args.api_url, timeout=args.timeout),
        data=args.dataset_name,
        evaluators=[
            response_status_matches_expected,
            reported_source_tables_match_expected,
            generated_sql_is_read_only,
            result_facts_match_expected,
            answer_contains_expected_facts,
        ],
        experiment_prefix=experiment_prefix,
        description=(
            "Evaluación end-to-end de /query/answer con el golden set "
            "versionado de Dat-IA."
        ),
        metadata={
            "solution": "dat_ia_test",
            "dataset_version": GOLDEN_SET_VERSION,
            "dataset_content_sha256": content_hash,
            "api_url": args.api_url,
        },
        max_concurrency=args.max_concurrency,
    )
    print(results)


if __name__ == "__main__":
    main()

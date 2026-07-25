"""Herramientas para evaluar Dat-IA con el golden set versionado."""

from app.evaluation.golden_set import (
    DEFAULT_GOLDEN_SET_PATH,
    GOLDEN_SET_DATASET_NAME,
    GOLDEN_SET_VERSION,
    GoldenSetValidationError,
    answer_contains_expected_facts,
    compare_result_facts,
    generated_sql_is_read_only,
    golden_set_content_hash,
    is_read_only_sql,
    load_golden_set,
    reported_source_tables_match_expected,
    response_status_matches_expected,
    result_facts_match_expected,
    sync_golden_set,
)

__all__ = [
    "DEFAULT_GOLDEN_SET_PATH",
    "GOLDEN_SET_DATASET_NAME",
    "GOLDEN_SET_VERSION",
    "GoldenSetValidationError",
    "answer_contains_expected_facts",
    "compare_result_facts",
    "generated_sql_is_read_only",
    "golden_set_content_hash",
    "is_read_only_sql",
    "load_golden_set",
    "reported_source_tables_match_expected",
    "response_status_matches_expected",
    "result_facts_match_expected",
    "sync_golden_set",
]

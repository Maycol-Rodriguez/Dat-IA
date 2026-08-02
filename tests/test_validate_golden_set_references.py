from pathlib import Path

from scripts.validate_golden_set_references import (
    DEFAULT_REPORT_PATH,
    REPOSITORY_ROOT,
    _path_for_report,
)


def test_path_for_report_accepts_a_relative_repository_path(
    monkeypatch,
) -> None:
    monkeypatch.chdir(REPOSITORY_ROOT)
    dataset_path = Path(
        "tests/evaluation/datasets/dat_ia_golden_set_v2.jsonl"
    )

    assert _path_for_report(dataset_path) == str(dataset_path)


def test_path_for_report_keeps_an_external_path_absolute() -> None:
    dataset_path = REPOSITORY_ROOT.parent / "external_golden.jsonl"

    assert _path_for_report(dataset_path) == str(dataset_path.resolve())


def test_reference_audit_defaults_to_the_ignored_archive() -> None:
    assert DEFAULT_REPORT_PATH == (
        REPOSITORY_ROOT
        / "reports"
        / "archive"
        / "dat_ia_golden_v2_reference_validation.json"
    )

from app.optimizer.query_optimizer import OptimizedQuery, QueryFilter
from app.validation.result_guardrail import check_groundedness, check_result


def _optimized_query(**overrides) -> OptimizedQuery:
    defaults = dict(
        original_question="Que transportista tiene el mayor promedio de retraso?",
        normalized_question="promedio de retraso por transportista",
        intent="aggregation",
        operation="average",
        metrics=["delay_days"],
        filters=[QueryFilter(field="state", operator="=", value="SP")],
        date_range=None,
        group_by=["carrier_name"],
        context=["logistica"],
        suggested_tables=["carriers"],
        optimizer="rule_based",
    )
    defaults.update(overrides)
    return OptimizedQuery(**defaults)


def test_check_result_ok_when_rows_present_and_under_limit() -> None:
    rows = [{"carrier_name": "DHL", "delay_days": 1.2}]

    result = check_result(rows, _optimized_query(), row_limit=200)

    assert result.ok is True
    assert result.warnings == []


def test_check_result_warns_when_rows_empty() -> None:
    result = check_result([], _optimized_query(), row_limit=200)

    assert result.ok is False
    assert "no devolvió filas" in result.warnings[0]


def test_check_result_warns_when_metric_all_null() -> None:
    rows = [
        {"carrier_name": "DHL", "delay_days": None},
        {"carrier_name": "FedEx", "delay_days": None},
    ]

    result = check_result(rows, _optimized_query(metrics=["delay_days"]), row_limit=200)

    assert result.ok is False
    assert "delay_days" in result.warnings[0]


def test_check_result_ignores_partial_nulls_in_metric() -> None:
    rows = [
        {"carrier_name": "DHL", "delay_days": None},
        {"carrier_name": "FedEx", "delay_days": 2.0},
    ]

    result = check_result(rows, _optimized_query(metrics=["delay_days"]), row_limit=200)

    assert result.ok is True


def test_check_result_warns_when_truncated_to_row_limit() -> None:
    rows = [{"carrier_name": f"carrier_{i}", "delay_days": 1.0} for i in range(3)]

    result = check_result(rows, _optimized_query(), row_limit=3)

    assert result.ok is False
    assert "truncó a 3 filas" in result.warnings[0]


def test_check_groundedness_ok_when_numbers_match_rows() -> None:
    rows = [{"carrier_name": "DHL", "on_time_rate": 0.97}]
    answer = "El transportista con mejor cumplimiento es DHL con 0.97."

    result = check_groundedness(answer, rows)

    assert result.ok is True
    assert result.unsupported_numbers == []


def test_check_groundedness_flags_unsupported_number() -> None:
    rows = [{"carrier_name": "DHL", "on_time_rate": 0.97}]
    answer = "El transportista con mejor cumplimiento es DHL con 452 pedidos."

    result = check_groundedness(answer, rows)

    assert result.ok is False
    assert "452" in result.unsupported_numbers


def test_check_groundedness_respects_rounding_tolerance() -> None:
    rows = [{"carrier_name": "DHL", "on_time_rate": 0.9701}]
    answer = "La tasa de cumplimiento es 0.97."

    result = check_groundedness(answer, rows, tolerance=0.01)

    assert result.ok is True

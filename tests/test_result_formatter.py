from datetime import date, datetime
from decimal import Decimal

from app.formatting.result_formatter import (
    format_result_table,
    humanize_column_name,
)


def _column_type(
    table: dict,
    key: str,
) -> str:
    return next(
        column["type"]
        for column in table["columns"]
        if column["key"] == key
    )


def test_humanize_column_name() -> None:
    assert (
        humanize_column_name(
            "carrier_name"
        )
        == "Transportista"
    )
    assert (
        humanize_column_name(
            "order_id"
        )
        == "Order ID"
    )


def test_format_result_table_empty() -> None:
    result = format_result_table([])

    assert result == {
        "columns": [],
        "rows": [],
        "row_count": 0,
        "locale": "es_PE",
    }


def test_format_result_table_numbers_and_nulls() -> None:
    rows = [
        {
            "order_count": 1946,
            "average_revenue": Decimal(
                "140.7401589162433242"
            ),
            "comment": None,
        }
    ]

    result = format_result_table(rows)

    assert (
        result["rows"][0][
            "order_count"
        ]
        == "1,946"
    )

    assert (
        result["rows"][0][
            "average_revenue"
        ]
        == "140.74"
    )

    assert (
        result["rows"][0][
            "comment"
        ]
        == "—"
    )


def test_format_result_table_percentage() -> None:
    result = format_result_table(
        [
            {
                "carrier_name": "DHL",
                "on_time_rate": 0.97,
            }
        ]
    )

    assert (
        result["rows"][0][
            "on_time_rate"
        ]
        == "97.00 %"
    )

    assert (
        _column_type(
            result,
            "on_time_rate",
        )
        == "percentage"
    )


def test_format_result_table_boolean() -> None:
    result = format_result_table(
        [
            {
                "active": True,
                "resolved": False,
            }
        ]
    )

    assert (
        result["rows"][0]["active"]
        == "Sí"
    )
    assert (
        result["rows"][0]["resolved"]
        == "No"
    )


def test_format_result_table_date() -> None:
    result = format_result_table(
        [
            {
                "order_date": date(
                    2026,
                    7,
                    20,
                )
            }
        ]
    )

    assert (
        result["rows"][0][
            "order_date"
        ]
        == "20/07/2026"
    )

    assert (
        _column_type(
            result,
            "order_date",
        )
        == "date"
    )


def test_format_result_table_datetime() -> None:
    result = format_result_table(
        [
            {
                "created_at": datetime(
                    2026,
                    7,
                    20,
                    14,
                    35,
                    10,
                )
            }
        ]
    )

    assert (
        result["rows"][0][
            "created_at"
        ]
        == "20/07/2026 14:35:10"
    )

    assert (
        _column_type(
            result,
            "created_at",
        )
        == "datetime"
    )


def test_format_result_table_iso_date_string() -> None:
    result = format_result_table(
        [
            {
                "order_date": (
                    "2026-07-20"
                )
            }
        ]
    )

    assert (
        result["rows"][0][
            "order_date"
        ]
        == "20/07/2026"
    )


def test_identifier_is_not_grouped() -> None:
    result = format_result_table(
        [
            {
                "customer_id": 123456789,
            }
        ]
    )

    assert (
        result["rows"][0][
            "customer_id"
        ]
        == "123456789"
    )


def test_year_is_not_grouped() -> None:
    result = format_result_table(
        [
            {
                "year": 2026,
            }
        ]
    )

    assert (
        result["rows"][0]["year"]
        == "2026"
    )


def test_original_rows_are_not_modified() -> None:
    rows = [
        {
            "order_count": 1946,
            "on_time_rate": 0.97,
        }
    ]

    original = [
        row.copy()
        for row in rows
    ]

    format_result_table(rows)

    assert rows == original


def test_percentage_already_in_percentage_points() -> None:
    result = format_result_table(
        [
            {
                "completion_rate": 97,
            }
        ]
    )

    assert (
        result["rows"][0][
            "completion_rate"
        ]
        == "97.00 %"
    )

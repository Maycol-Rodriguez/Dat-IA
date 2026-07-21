"""Formateo determinista de resultados SQL para presentación al usuario.

Este módulo no modifica los datos originales devueltos por la base de datos.
Construye una representación adicional orientada exclusivamente a presentación.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal
from math import isfinite
from typing import Any, Literal

from babel.dates import format_date, format_datetime
from babel.numbers import format_decimal, format_percent

from .column_labels import get_column_label


DEFAULT_LOCALE = "es_PE"
NULL_DISPLAY = "—"


ResultValueType = Literal[
    "text",
    "integer",
    "decimal",
    "percentage",
    "date",
    "datetime",
    "boolean",
    "null",
]


_PERCENT_HINTS = {
    "pct",
    "percent",
    "percentage",
    "porcentaje",
    "rate",
    "ratio",
    "tasa",
}

_DATE_HINTS = {
    "date",
    "fecha",
}

_DATETIME_HINTS = {
    "datetime",
    "timestamp",
}

_PLAIN_INTEGER_HINTS = {
    "year",
    "anio",
    "año",
    "month",
    "mes",
    "day",
    "dia",
    "día",
    "month_number",
    "mes_nro",
    "mes_num",
}


def _key_tokens(key: str) -> set[str]:
    normalized = re.sub(
        r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ]+",
        " ",
        key,
    )

    return {
        token.casefold()
        for token in normalized.split()
        if token
    }


def _is_identifier_key(key: str) -> bool:
    normalized = key.strip().casefold()

    return (
        normalized == "id"
        or normalized.endswith("_id")
        or normalized.endswith(" id")
    )


def _is_plain_integer_key(key: str) -> bool:
    normalized = key.strip().casefold()

    if normalized in _PLAIN_INTEGER_HINTS:
        return True

    return bool(
        _key_tokens(normalized)
        & _PLAIN_INTEGER_HINTS
    )


def _looks_like_percentage_key(key: str) -> bool:
    return bool(
        _key_tokens(key)
        & _PERCENT_HINTS
    )


def _looks_like_datetime_key(key: str) -> bool:
    normalized = key.strip().casefold()

    if normalized.endswith("_at"):
        return True

    return bool(
        _key_tokens(normalized)
        & _DATETIME_HINTS
    )


def _looks_like_date_key(key: str) -> bool:
    return (
        _looks_like_datetime_key(key)
        or bool(
            _key_tokens(key)
            & _DATE_HINTS
        )
    )


def humanize_column_name(key: str) -> str:
    """Convierte nombres técnicos en etiquetas legibles.

    Ejemplos:
        carrier_name -> Carrier name
        order_id -> Order ID
        on_time_rate -> On time rate
    """
    registered_label = get_column_label(
        key
    )

    if registered_label is not None:
        return registered_label

    label = re.sub(
        r"[_\-\s]+",
        " ",
        str(key).strip(),
    )

    label = re.sub(
        r"\bid\b",
        "ID",
        label,
        flags=re.IGNORECASE,
    )

    if not label:
        return ""

    return (
        label[0].upper()
        + label[1:]
    )


def _is_numeric(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(
            value,
            (int, float, Decimal),
        )
    )


def _is_finite_numeric(value: Any) -> bool:
    if isinstance(value, Decimal):
        return value.is_finite()

    if isinstance(value, float):
        return isfinite(value)

    return True


def _is_integral_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False

    if isinstance(value, int):
        return True

    if isinstance(value, Decimal):
        return (
            value
            == value.to_integral_value()
        )

    if isinstance(value, float):
        return (
            isfinite(value)
            and value.is_integer()
        )

    return False


def _parse_iso_temporal(
    value: str,
) -> date | datetime | None:
    text = value.strip()

    if not text:
        return None

    iso_text = (
        text[:-1] + "+00:00"
        if text.endswith("Z")
        else text
    )

    try:
        if (
            "T" in iso_text
            or " " in iso_text
        ):
            return datetime.fromisoformat(
                iso_text
            )

        if re.fullmatch(
            r"\d{4}-\d{2}-\d{2}",
            iso_text,
        ):
            return date.fromisoformat(
                iso_text
            )
    except ValueError:
        return None

    return None


def _coerce_temporal(
    value: Any,
    key: str,
) -> date | datetime | None:
    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return value

    if (
        isinstance(value, str)
        and _looks_like_date_key(key)
    ):
        return _parse_iso_temporal(value)

    return None


def infer_result_type(
    key: str,
    values: list[Any],
) -> ResultValueType:
    """Infiere el tipo de presentación de una columna."""
    non_null = [
        value
        for value in values
        if value is not None
    ]

    if not non_null:
        return "null"

    if _is_identifier_key(key):
        return "text"

    if all(
        isinstance(value, bool)
        for value in non_null
    ):
        return "boolean"

    temporal_values = [
        _coerce_temporal(
            value,
            key,
        )
        for value in non_null
    ]

    if all(
        value is not None
        for value in temporal_values
    ):
        if any(
            isinstance(value, datetime)
            for value in temporal_values
        ):
            return "datetime"

        return "date"

    if all(
        _is_numeric(value)
        for value in non_null
    ):
        if _looks_like_percentage_key(
            key
        ):
            return "percentage"

        if all(
            _is_integral_numeric(value)
            for value in non_null
        ):
            return "integer"

        return "decimal"

    return "text"


def format_result_value(
    value: Any,
    *,
    column_key: str,
    value_type: ResultValueType,
    locale: str = DEFAULT_LOCALE,
) -> str:
    """Formatea un único valor para presentación."""
    if value is None:
        return NULL_DISPLAY

    if value_type == "null":
        return NULL_DISPLAY

    if value_type == "boolean":
        return (
            "Sí"
            if bool(value)
            else "No"
        )

    if value_type in {
        "date",
        "datetime",
    }:
        temporal = _coerce_temporal(
            value,
            column_key,
        )

        if temporal is None:
            return str(value)

        if (
            value_type == "datetime"
            and isinstance(
                temporal,
                datetime,
            )
        ):
            return format_datetime(
                temporal,
                format="dd/MM/yyyy HH:mm:ss",
                locale=locale,
            )

        return format_date(
            temporal,
            format="dd/MM/yyyy",
            locale=locale,
        )

    if value_type == "text":
        return str(value)

    if not _is_numeric(value):
        return str(value)

    if not _is_finite_numeric(value):
        return NULL_DISPLAY

    if value_type == "percentage":
        numeric_value = (
            value
            if isinstance(value, Decimal)
            else Decimal(str(value))
        )

        if abs(numeric_value) <= Decimal("1"):
            return format_percent(
                numeric_value,
                format="#,##0.00 %",
                locale=locale,
            )

        return (
            format_decimal(
                numeric_value,
                format="#,##0.00",
                locale=locale,
                decimal_quantization=True,
                group_separator=True,
            )
            + " %"
        )

    if value_type == "integer":
        if (
            _is_identifier_key(
                column_key
            )
            or _is_plain_integer_key(
                column_key
            )
        ):
            return str(int(value))

        return format_decimal(
            value,
            format="#,##0",
            locale=locale,
            decimal_quantization=True,
            group_separator=True,
        )

    return format_decimal(
        value,
        format="#,##0.00",
        locale=locale,
        decimal_quantization=True,
        group_separator=True,
    )


def format_result_table(
    rows: list[dict[str, Any]],
    *,
    locale: str = DEFAULT_LOCALE,
) -> dict[str, Any]:
    """Construye una tabla estandarizada sin modificar las filas originales."""
    if not rows:
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "locale": locale,
        }

    keys: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for raw_key in row:
            key = str(raw_key)

            if key in seen:
                continue

            seen.add(key)
            keys.append(key)

    column_types: dict[
        str,
        ResultValueType,
    ] = {}

    columns = []

    for key in keys:
        values = [
            row.get(key)
            for row in rows
        ]

        value_type = infer_result_type(
            key,
            values,
        )

        column_types[key] = value_type

        columns.append(
            {
                "key": key,
                "label": (
                    humanize_column_name(
                        key
                    )
                ),
                "type": value_type,
            }
        )

    formatted_rows = []

    for row in rows:
        formatted_row = {}

        for key in keys:
            formatted_row[key] = (
                format_result_value(
                    row.get(key),
                    column_key=key,
                    value_type=(
                        column_types[key]
                    ),
                    locale=locale,
                )
            )

        formatted_rows.append(
            formatted_row
        )

    return {
        "columns": columns,
        "rows": formatted_rows,
        "row_count": len(rows),
        "locale": locale,
    }

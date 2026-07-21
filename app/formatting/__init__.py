"""Utilidades para estandarizar resultados SQL para presentación."""

from .result_formatter import (
    DEFAULT_LOCALE,
    format_result_table,
    format_result_value,
    humanize_column_name,
)

__all__ = [
    "DEFAULT_LOCALE",
    "format_result_table",
    "format_result_value",
    "humanize_column_name",
]

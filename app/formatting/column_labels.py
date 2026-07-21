"""Etiquetas de presentación para columnas conocidas de Dat-IA.

El registro es deliberadamente independiente del formatter:
permite ampliar las etiquetas de negocio sin modificar las reglas
de inferencia y formateo de tipos.
"""

from __future__ import annotations


COLUMN_LABELS_ES: dict[str, str] = {
    # Logística / transportistas
    "carrier_name": "Transportista",
    "carrier_type": "Tipo de transportista",
    "avg_delivery_days": "Tiempo promedio de entrega (días)",
    "coverage_regions": "Regiones de cobertura",
    "cost_per_kg": "Costo por kg",
    "on_time_rate": "Tasa de puntualidad",

    # Historial de precios
    "old_price": "Precio anterior",
    "new_price": "Precio nuevo",
    "change_at": "Fecha de cambio",
    "change_reason": "Motivo del cambio",

    # Soporte al cliente
    "created_at": "Fecha de creación",
    "resolution_time_hr": "Tiempo de resolución (h)",
    "satisfaction_score": "Puntaje de satisfacción",
    "resolved": "Resuelto",

    # Fechas comunes
    "start_date": "Fecha de inicio",
    "end_date": "Fecha de fin",
    "reported_date": "Fecha de reporte",
    "resolved_date": "Fecha de resolución",
    "return_date": "Fecha de devolución",
    "last_restocked_date": "Fecha del último reabastecimiento",
}


def get_column_label(key: str) -> str | None:
    """Devuelve la etiqueta española registrada para una columna."""
    normalized_key = str(key).strip().casefold()

    return COLUMN_LABELS_ES.get(normalized_key)

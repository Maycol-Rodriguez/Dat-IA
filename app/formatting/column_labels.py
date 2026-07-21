"""Etiquetas de presentación para columnas conocidas de Dat-IA.

El registro mantiene las etiquetas semánticas del esquema separadas
de las reglas de inferencia y formateo de valores.

Para columnas o aliases no registrados se utiliza un fallback
determinista basado en tokens comunes.
"""

from __future__ import annotations

import re


COLUMN_LABELS_ES: dict[str, str] = {
    # ------------------------------------------------------------------
    # Transportistas
    # ------------------------------------------------------------------
    "carrier_id": "ID del transportista",
    "carrier_name": "Transportista",
    "carrier_type": "Tipo de transportista",
    "avg_delivery_days": "Tiempo promedio de entrega (días)",
    "coverage_regions": "Regiones de cobertura",
    "cost_per_kg": "Costo por kg",
    "on_time_rate": "Tasa de puntualidad",

    # ------------------------------------------------------------------
    # Incidentes de entrega
    # ------------------------------------------------------------------
    "incident_id": "ID del incidente",
    "incident_type": "Tipo de incidente",
    "reported_date": "Fecha de reporte",
    "resolved_date": "Fecha de resolución",
    "resolution_type": "Tipo de resolución",
    "compensation_value": "Valor de compensación",

    # ------------------------------------------------------------------
    # Soporte al cliente
    # ------------------------------------------------------------------
    "ticket_id": "ID del ticket",
    "created_at": "Fecha de creación",
    "category": "Categoría",
    "priority": "Prioridad",
    "resolution_time_hr": "Tiempo de resolución (h)",
    "satisfaction_score": "Puntaje de satisfacción",
    "resolved": "Resuelto",

    # ------------------------------------------------------------------
    # Clientes
    # ------------------------------------------------------------------
    "customer_id": "ID del cliente",
    "customer_unique_id": "ID único del cliente",
    "customer_zip_code_prefix": "Código postal del cliente",
    "customer_city": "Ciudad del cliente",
    "customer_state": "Estado del cliente",

    # ------------------------------------------------------------------
    # Geolocalización
    # ------------------------------------------------------------------
    "geolocation_zip_code_prefix": "Código postal",
    "geolocation_lat": "Latitud",
    "geolocation_lng": "Longitud",
    "geolocation_city": "Ciudad",
    "geolocation_state": "Estado",

    # ------------------------------------------------------------------
    # Órdenes
    # ------------------------------------------------------------------
    "order_id": "ID del pedido",
    "order_status": "Estado del pedido",
    "order_purchase_timestamp": "Fecha de compra",
    "order_approved_at": "Fecha de aprobación",
    "order_delivered_carrier_date": "Fecha de entrega al transportista",
    "order_delivered_customer_date": "Fecha de entrega al cliente",
    "order_estimated_delivery_date": "Fecha estimada de entrega",

    # ------------------------------------------------------------------
    # Ítems de órdenes
    # ------------------------------------------------------------------
    "order_item_id": "ID del ítem del pedido",
    "shipping_limit_date": "Fecha límite de envío",
    "price": "Precio",
    "freight_value": "Valor del flete",

    # ------------------------------------------------------------------
    # Pagos
    # ------------------------------------------------------------------
    "payment_sequential": "Secuencia de pago",
    "payment_type": "Tipo de pago",
    "payment_installments": "Número de cuotas",
    "payment_value": "Valor del pago",

    # ------------------------------------------------------------------
    # Productos
    # ------------------------------------------------------------------
    "product_id": "ID del producto",
    "product_category_name": "Categoría del producto",
    "product_category_name_english": "Categoría del producto en inglés",
    "product_name_lenght": "Longitud del nombre del producto",
    "product_description_lenght": "Longitud de la descripción",
    "product_photos_qty": "Cantidad de fotos",
    "product_weight_g": "Peso del producto (g)",
    "product_length_cm": "Largo del producto (cm)",
    "product_height_cm": "Alto del producto (cm)",
    "product_width_cm": "Ancho del producto (cm)",

    # ------------------------------------------------------------------
    # Historial de precios
    # ------------------------------------------------------------------
    "price_event_id": "ID del cambio de precio",
    "old_price": "Precio anterior",
    "new_price": "Precio nuevo",
    "change_at": "Fecha de cambio",
    "change_reason": "Motivo del cambio",

    # ------------------------------------------------------------------
    # Promociones
    # ------------------------------------------------------------------
    "promotion_id": "ID de la promoción",
    "discount_pct": "Porcentaje de descuento",
    "start_date": "Fecha de inicio",
    "end_date": "Fecha de fin",
    "promo_type": "Tipo de promoción",
    "units_sold_during": "Unidades vendidas durante la promoción",

    # ------------------------------------------------------------------
    # Inventario
    # ------------------------------------------------------------------
    "warehouse_id": "ID del almacén",
    "stock_qty": "Cantidad en stock",
    "reorder_point": "Punto de reorden",
    "last_restocked_date": "Fecha del último reabastecimiento",

    # ------------------------------------------------------------------
    # Devoluciones
    # ------------------------------------------------------------------
    "return_id": "ID de la devolución",
    "return_reason": "Motivo de devolución",
    "return_date": "Fecha de devolución",
    "refund_amount": "Monto reembolsado",
    "refund_method": "Método de reembolso",
    "reestocked": "Reincorporado al inventario",

    # ------------------------------------------------------------------
    # Reseñas
    # ------------------------------------------------------------------
    "review_id": "ID de la reseña",
    "review_score": "Puntaje de la reseña",
    "review_comment_title": "Título del comentario",
    "review_comment_message": "Comentario de la reseña",
    "review_creation_date": "Fecha de creación de la reseña",
    "review_answer_timestamp": "Fecha de respuesta a la reseña",

    # ------------------------------------------------------------------
    # Vendedores
    # ------------------------------------------------------------------
    "seller_id": "ID del vendedor",
    "seller_zip_code_prefix": "Código postal del vendedor",
    "seller_city": "Ciudad del vendedor",
    "seller_state": "Estado del vendedor",

    # ------------------------------------------------------------------
    # Genéricos
    # ------------------------------------------------------------------
    "id": "ID",
}


_TOKEN_LABELS_ES: dict[str, str] = {
    "total": "Total",
    "count": "cantidad",
    "number": "número",
    "average": "promedio",
    "avg": "promedio",
    "sum": "total",
    "orders": "pedidos",
    "order": "pedido",
    "customers": "clientes",
    "customer": "cliente",
    "products": "productos",
    "product": "producto",
    "sellers": "vendedores",
    "seller": "vendedor",
    "returns": "devoluciones",
    "return": "devolución",
    "payments": "pagos",
    "payment": "pago",
    "price": "precio",
    "revenue": "ingresos",
    "value": "valor",
    "amount": "monto",
    "quantity": "cantidad",
    "qty": "cantidad",
    "units": "unidades",
    "days": "días",
    "delivery": "entrega",
    "rate": "tasa",
    "score": "puntaje",
    "status": "estado",
    "date": "fecha",
}


def get_column_label(key: str) -> str | None:
    """Devuelve una etiqueta española para una columna o alias."""
    normalized_key = str(key).strip().casefold()

    registered = COLUMN_LABELS_ES.get(
        normalized_key
    )

    if registered is not None:
        return registered

    tokens = [
        token
        for token in re.split(
            r"[_\-\s]+",
            normalized_key,
        )
        if token
    ]

    if not tokens:
        return None

    translated = [
        _TOKEN_LABELS_ES.get(token)
        for token in tokens
    ]

    if any(
        token is None
        for token in translated
    ):
        return None

    label = " ".join(
        token
        for token in translated
        if token is not None
    )

    if not label:
        return None

    return (
        label[0].upper()
        + label[1:]
    )

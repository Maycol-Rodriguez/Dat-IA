# DDL de Dat-IA

- `ddl.json`: DDL operativo. Contiene las 16 tablas oficiales, sus catálogos
  observados y políticas semánticas de negocio. Sus identificadores son
  consecutivos, desde `tabla_1` hasta `tabla_16`.
- `ddl_old.json`: copia deprecada del DDL operativo anterior a la incorporación
  de políticas semánticas. Se conserva solo como trazabilidad y no se ingiere
  al iniciar la API.

Las políticas añadidas al DDL activo definen:

- fecha de compra como dimensión temporal predeterminada;
- conteo distinto de órdenes al unir tablas con granularidad de ítem;
- ingreso de productos como `SUM(price)`, sin flete salvo solicitud explícita;
- identidad de compradores únicos y recurrentes mediante `customer_unique_id`;
- uso de `seller_promotions` y `units_sold_during` para promociones;
- traducción de categorías únicamente cuando la salida requiera inglés.

PostgreSQL sigue siendo la fuente oficial de estructura y datos. Estas políticas
son reglas de interpretación del negocio y no modifican la base de datos.

## Uso actual en el pipeline

Al iniciar la API, `descripcion` se vectoriza como contenido del documento y
`ddl` se almacena como metadata. La descripción ayuda a seleccionar tablas,
pero el generador SQL recibe actualmente solo el campo `ddl` de las tablas
recuperadas. Por ello, una política escrita únicamente en `descripcion` todavía
no actúa como instrucción obligatoria de generación. Esta limitación está
registrada en el reporte vigente del golden set.

## Actualización desde PostgreSQL

El script `scripts/refresh_ddl_from_database.py` vuelve a asignar IDs
consecutivos después de ordenar las tablas, evitando huecos cuando cambia el
catálogo:

```powershell
uv run --env-file .env python -m scripts.refresh_ddl_from_database
```

La auditoría detallada se escribe en
`reports/archive/dat_ia_ddl_validation.json`. `archive/` está ignorado por Git,
por lo que ese JSON es evidencia local regenerable y no un artefacto que deba
acompañar cada commit.

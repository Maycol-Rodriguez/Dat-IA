# Últimos hallazgos del golden set v2

- Fecha de corte: 2026-08-01 (America/Lima)
- Golden set canónico: `tests/evaluation/datasets/dat_ia_golden_set_v2.jsonl`
- Versión lógica: `2.1.0`
- Dataset remoto: `dat_ia_test_golden_v2`

## Alcance del reporte

Este es el único reporte vigente del golden set versionado en Git. Analiza la
última ejecución completa y válida disponible:

- experimento: `dat_ia_test-golden-v2-6fea79af`;
- ID: `867f96c5-8989-4be8-b7a8-ae06e1e98166`;
- inicio: 2026-08-02 01:54:15 UTC;
- revisión ejecutada: `f21ea0c-dirty`;
- SHA-256 evaluado: `3137a95ae22e0e394d8d37e5c00e5547df90edd2a5e600d62948b94d42e40266`.

La ejecución posterior que muestra `0%` no se considera una medición de
calidad: se agotó la cuota de tokens y el pipeline no pudo producir respuestas
evaluables. Debe tratarse como un fallo de infraestructura, no como una
regresión de Dat-IA.

## Resultado observado

| Métrica | Resultado |
|---|---:|
| Hechos del resultado correctos | 56.67% (17/30) |
| Hechos presentes en la respuesta | 53.33% (16/30) |
| Tablas fuente correctas | 80.00% (24/30) |
| Estado funcional correcto | 80.00% (24/30) |
| SQL de solo lectura | 96.67% (29/30) |

La métrica principal permaneció en 17 de 30, pero hubo cambios internos:

- mejoraron `golden_015`, `golden_025`, `golden_026` y `golden_029`;
- retrocedieron `golden_008`, `golden_013`, `golden_016` y `golden_028`;
- la simplificación de columnas opcionales funcionó en `015`, `025` y `029`.

## Ajuste posterior de `golden_010`

La ejecución válida todavía comparó `golden_010` contra un porcentaje de
`96.0`. PostgreSQL devuelve oficialmente la tasa cruda `0.960`, por lo que el
caso canónico ahora espera:

```json
{"carrier_name": "InterEstadual Cargo", "on_time_rate": 0.96}
```

Su SQL de referencia también devuelve `on_time_rate` sin multiplicar por 100.
Después del cambio, los 30 SQL de referencia fueron ejecutados nuevamente
contra PostgreSQL: `30 correct`, `0 golden_set_outdated` y
`0 reference_sql_error`. El SHA-256 canónico actual es:

```text
b326cc291513a1ec9d5262c57d0d86535e320b4648e37ca7914c12dbb4683c3e
```

Este ajuste aún no está medido en un experimento completo válido. La próxima
ejecución debe realizarse sin `--skip-sync` para actualizar el ejemplo remoto
antes de evaluar.

## Fallos funcionales pendientes

Excluyendo el contrato ya corregido de `golden_010`, quedan 12 casos que deben
volver a medirse:

| Caso | Hallazgo principal | Capa probable |
|---|---|---|
| `008` | La normalización amplió “sin resolver” e indujo una columna `status` inexistente. | Optimizer/generación |
| `011` | Agrupó por fecha de entrega en lugar de fecha de compra. | Política semántica/generación |
| `012` | PromptShield bloqueó una pregunta analítica válida. | Seguridad |
| `013` | `customer_state` no activó la regla de tabla de clientes y el SQL quedó fuera del contexto permitido. | Optimizer/recuperación |
| `014` | Devolvió categorías en portugués aunque la salida requería traducción. | Generación |
| `016` | La tabla de órdenes quedó a distancia `0.7013`, apenas fuera del umbral `0.7`. | Recuperación |
| `021` | Usó `customer_id` en vez de `customer_unique_id`. | Política semántica/generación |
| `022` | El optimizer omitió la tabla de ítems necesaria para el SQL. | Optimizer/recuperación |
| `024` | Omitió la traducción de categorías aun cuando la tabla fue recuperada. | Generación |
| `027` | Contó filas de ítems en vez de órdenes distintas. | Política semántica/generación |
| `028` | El juez LLM rechazó un SQL que había superado el `EXPLAIN`. | Juez SQL |
| `030` | Calculó ingreso por mes de entrega en vez de mes de compra. | Política semántica/generación |

## Causas transversales

1. Las políticas añadidas a `descripcion` en `data/ddl.json` influyen en la
   búsqueda vectorial, pero el generador recibe únicamente el campo `ddl`.
2. El generador y el juez trabajan actualmente con la pregunta normalizada;
   una normalización más amplia puede alterar la intención original.
3. El optimizer puede omitir tablas requeridas por filtros o por variantes de
   nombres como `state` frente a `customer_state`.
4. Un umbral global rígido puede excluir una tabla necesaria por diferencias
   mínimas de distancia.
5. Query Memory reutiliza ejemplos dentro del prompt, pero no evita que el
   flujo vuelva a generar, validar, juzgar y ejecutar SQL. No explica por sí
   sola la mayoría de las regresiones observadas.

## Prioridad recomendada

1. Propagar al generador las políticas semánticas de las tablas recuperadas.
2. Conservar la pregunta original para generación SQL y usar la normalizada
   para recuperación y memoria.
3. Hacer que el optimizer combine reglas deterministas y sugerencias del LLM,
   incluyendo tablas exigidas por filtros y agrupaciones.
4. Añadir regresiones específicas para el juez SQL y revisar el falso bloqueo
   de PromptShield por separado.
5. Sincronizar el golden set actual y ejecutar otra línea base completa cuando
   exista cuota suficiente.

Los JSON detallados producidos por las auditorías manuales se guardan en
`reports/archive/`. Esa carpeta está ignorada por Git; sus archivos son apoyo
local y pueden regenerarse desde los scripts documentados.

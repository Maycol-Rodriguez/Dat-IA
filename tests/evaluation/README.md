# Evaluación end-to-end de Dat-IA

Este directorio contiene el flujo manual para evaluar las 30 preguntas del
golden set contra `POST /query/answer` y publicar los resultados en LangSmith.
No forma parte del arranque normal de la API ni de `pytest`.

## Qué se versiona

Existe un único golden set canónico:

[`datasets/dat_ia_golden_v2.jsonl`](datasets/dat_ia_golden_v2.jsonl)

Contiene las 30 preguntas, sus SQL de referencia, tablas esperadas, reglas de
negocio y todas las filas esperadas. Está codificado en UTF-8 y tiene un objeto
JSON por línea.

Se usa JSONL en lugar de CSV porque cada ejemplo contiene listas, objetos
anidados y varias filas. JSONL mantiene los tipos numéricos, se puede revisar
fácilmente en Git y no obliga a serializar JSON dentro de celdas CSV.

No existen datasets separados para casos correctos e incorrectos. Los 30 casos
se sincronizan en LangSmith bajo el nombre:

```text
dat_ia_test_golden_v2
```

El nombre del archivo local y el nombre del dataset remoto son identificadores
distintos, pero se mantienen alineados con el sufijo `golden_v2` para evitar
confusión. Los experimentos usan el prefijo `dat_ia_test-golden-v2`.

LangSmith guarda el dataset, las trazas, las métricas y los experimentos en su
servicio remoto. No crea archivos locales de resultados. Las antiguas carpetas
`candidates` y `quarantine` ya no existen y ningún script actual las genera.

## Archivos de auditoría

- `reports/dat_ia_golden_v2_reference_validation.json`: ejecuta los 30 SQL de
  referencia contra PostgreSQL y muestra, por caso, el resultado esperado, el
  resultado actual y el error. Sirve para que el equipo de Base de Datos
  determine si debe corregirse el golden set, la carga de datos o el SQL.
- `reports/dat_ia_ddl_validation.json`: compara `data/ddl_old.json` con la
  estructura y los valores categóricos de PostgreSQL. Sirve como evidencia del
  cambio de DDL dentro del PR.

Estos reportes pueden actualizarse manualmente, pero nunca modifican el golden
set ni afectan el funcionamiento de Dat-IA.

## Contrato de un caso

Cada línea del JSONL tiene esta forma:

```json
{
  "case_id": "golden_001",
  "enabled": true,
  "split": "core",
  "inputs": {
    "question": "¿Cuántas órdenes se registraron en total?"
  },
  "reference_outputs": {
    "expected_status": "success",
    "expected_sources": ["olist_orders_dataset"],
    "reference_sql": "SELECT COUNT(*) AS total_orders FROM olist_orders_dataset;",
    "expected_result": {
      "row_count": 1,
      "rows": [{"total_orders": 99441}],
      "numeric_tolerance": 0.01
    }
  },
  "metadata": {
    "dataset_version": "2.0.0",
    "domain": "Órdenes",
    "result_type": "scalar",
    "business_rule": "Contar cada order_id una sola vez.",
    "quality_status": "candidate_pending_db_validation",
    "quality_notes": [],
    "omitted_optional_columns": [],
    "source_row": 1
  }
}
```

`expected_result.rows` siempre contiene todas las filas conocidas y
`row_count` debe coincidir con su longitud. No existe `comparison_mode`.

La comparación principal:

- exige la cantidad completa de filas;
- compara todos los hechos esperados;
- acepta alias de columnas diferentes;
- normaliza números, UUID y periodos mensuales;
- usa `numeric_tolerance`;
- no depende del orden accidental en que PostgreSQL devuelve las filas.

## Independencia de LangSmith

LangSmith es opcional para la API:

| Escenario | Variables LangSmith | Resultado |
|---|---|---|
| Ejecutar únicamente la API | Ninguna | La API funciona y `/ready` muestra `langsmith: not_connected`. |
| Registrar una sola consulta | `USE_LANGSMITH_TRACING=true` y `LANGSMITH_API_KEY` | La consulta y sus etapas aparecen en el proyecto de LangSmith. |
| Ejecutar el golden set | Las mismas dos variables | El script sincroniza los 30 ejemplos y crea un experimento. |

`LANGSMITH_PROJECT` es opcional y usa `dat_ia_test` de forma predeterminada.
`LANGSMITH_TRACING_SAMPLING_RATE` también es opcional y usa `1.0`.

Los decoradores de observabilidad se deshabilitan cuando falta la bandera o la
API key. En ese estado no se crea un cliente, no se envían trazas y un fallo de
configuración de LangSmith no impide arrancar la API.

Las variables se leen al importar la aplicación. Si se activa el tracing
después de iniciar Uvicorn, hay que reiniciar la API.

## Requisitos para `/query/answer`

La ejecución end-to-end necesita:

- `GOOGLE_API_KEY`;
- `DATABASE_URL`;
- API Dat-IA iniciada;
- LangSmith solamente si se desea guardar la traza o ejecutar el experimento.

Sin `DATABASE_URL`, los endpoints que solo generan SQL pueden seguir
funcionando, pero `/query/answer` no puede ejecutar la consulta.

## 1. Iniciar la API

Con un `.env`:

```cmd
uv run --env-file .env uvicorn app.main:app --reload
```

Comprobar disponibilidad:

```cmd
curl http://127.0.0.1:8000/ready
```

Para la evaluación, la respuesta debe contener:

```json
{
  "status": "ok",
  "database": "connected"
}
```

## 2. Probar la API sin LangSmith

Dejar estas variables sin definir o desactivadas:

```text
USE_LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

Después se puede llamar normalmente a la API:

```cmd
curl -X POST http://127.0.0.1:8000/query/answer -H "Content-Type: application/json; charset=utf-8" -d "{\"question\":\"¿Cuántas órdenes se registraron en total?\"}"
```

No se ejecuta ningún script de evaluación y no se crea ninguna traza.

## 3. Registrar una sola respuesta en LangSmith

Configurar antes de iniciar la API:

```text
USE_LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
```

Reiniciar Uvicorn y llamar una sola vez a `/query/answer`:

```cmd
curl -X POST http://127.0.0.1:8000/query/answer -H "Content-Type: application/json; charset=utf-8" -d "{\"question\":\"¿Cuántas órdenes se registraron en total?\"}"
```

Esta operación registra una traza normal en el proyecto `dat_ia_test`. No
sincroniza el golden set ni crea un experimento de 30 preguntas.

## 4. Validar el flujo sin consumir 30 consultas

Con la API iniciada:

```cmd
uv run --env-file .env python -m scripts.evaluate_langsmith_golden_set --dry-run
```

Este comando:

1. abre y valida el JSONL;
2. confirma que existen 30 casos habilitados;
3. verifica `/ready`;
4. confirma que PostgreSQL está conectado.

No ejecuta las 30 preguntas, no llama al modelo para evaluarlas y no sincroniza
el dataset con LangSmith.

## 5. Ejecutar las 30 preguntas

Con la API iniciada y LangSmith configurado:

```cmd
uv run --env-file .env python -m scripts.evaluate_langsmith_golden_set
```

El comando realiza, en orden:

1. valida el golden set local;
2. verifica `/ready`;
3. crea o actualiza `dat_ia_test_golden_v2`;
4. llama a `/query/answer` una vez por cada pregunta;
5. ejecuta los evaluadores deterministas;
6. crea un nuevo experimento con prefijo
   `dat_ia_test-golden-v2`.

Ejecutarlo varias veces no duplica los ejemplos del dataset porque sus UUID son
deterministas. Cada repetición sí crea un experimento nuevo, lo cual permite
comparar estabilidad entre ejecuciones.

Opciones útiles:

```cmd
uv run --env-file .env python -m scripts.evaluate_langsmith_golden_set --timeout 180 --max-concurrency 1
```

`--max-concurrency 1` es la opción más segura para evitar límites del proveedor
y facilitar el diagnóstico por pregunta.

## Métricas publicadas

El experimento registra cinco métricas:

- `result_facts_match_expected`: métrica principal. Comprueba que `data`
  contiene todos los hechos y filas esperados.
- `answer_contains_expected_facts`: comprueba que la respuesta redactada
  mencione los hechos esperados. Puede ser más estricta con traducciones de
  categorías.
- `reported_source_tables_match_expected`: compara las tablas reportadas por
  Dat-IA con las tablas esperadas.
- `response_status_matches_expected`: comprueba que el estado funcional sea el
  esperado.
- `generated_sql_is_read_only`: rechaza SQL que no sea de solo lectura.

Una puntuación `0` indica que esa dimensión falló; no significa necesariamente
que toda la respuesta sea incorrecta. Para diagnosticar un caso hay que revisar
conjuntamente SQL, `data`, `answer`, `sources` y `status`.

## Auditar las respuestas de referencia

Este flujo es independiente del experimento end-to-end:

```cmd
uv run --env-file .env python -m scripts.validate_golden_set_references
```

Ejecuta los 30 SQL de referencia dentro de una transacción de solo lectura y
actualiza únicamente:

```text
reports/dat_ia_golden_v2_reference_validation.json
```

Clasifica cada caso como:

- `correct`: el SQL se ejecutó y coincide con la respuesta esperada;
- `golden_set_outdated`: el SQL se ejecutó, pero los datos difieren;
- `reference_sql_error`: el SQL de referencia no se pudo ejecutar.

El comando no elimina preguntas, no crea datasets parciales y no sustituye
automáticamente las respuestas esperadas.

## Auditar el DDL

Esta operación es opcional y está destinada al equipo de Base de Datos:

```cmd
uv run --env-file .env python -m scripts.refresh_ddl_from_database --audit-only --ddl-path data/ddl_old.json
```

Compara el DDL anterior con PostgreSQL y actualiza:

```text
reports/dat_ia_ddl_validation.json
```

Para refrescar los catálogos documentados en el DDL activo:

```cmd
uv run --env-file .env python -m scripts.refresh_ddl_from_database
```

`data/ddl_old.json` se conserva y nunca se sobrescribe.

## Pruebas locales

Las pruebas unitarias no necesitan LangSmith, PostgreSQL ni llamadas al modelo:

```cmd
uv run pytest
uv run ruff check app scripts tests
```

El experimento de 30 preguntas es manual porque consume el modelo, consulta la
base externa y escribe resultados en LangSmith.

## Problemas comunes

`LangSmith no está configurado`

: El experimento requiere `USE_LANGSMITH_TRACING=true` y
  `LANGSMITH_API_KEY`. La API normal no los requiere.

`database='not_configured'`

: Falta `DATABASE_URL` o falló la conexión al iniciar la API.

El experimento remoto tiene ejemplos antiguos

: El script se detiene si detecta ejemplos remotos que ya no están en el JSONL.
  Se debe revisar la diferencia antes de eliminar datos en LangSmith.

Una métrica falla aunque los números parezcan correctos

: Revisar si cambió la categoría, la regla temporal, las tablas utilizadas o
  la cantidad total de filas. La métrica principal ignora alias y orden, pero
  no ignora hechos faltantes.

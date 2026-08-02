# Evaluación end-to-end de Dat-IA

Este directorio contiene el flujo manual que ejecuta las 30 preguntas del
golden set contra `POST /query/answer` y publica las métricas en LangSmith. No
se ejecuta al iniciar FastAPI, durante `pytest` ni al consultar un endpoint de
la API.

## Fuente canónica

El único golden set operativo es:

[`datasets/dat_ia_golden_set_v2.jsonl`](datasets/dat_ia_golden_set_v2.jsonl)

Características:

- 30 casos habilitados;
- formato JSONL y codificación UTF-8;
- versión lógica `2.1.0`;
- SQL de referencia de solo lectura;
- resultados completos validados contra la PostgreSQL oficial;
- reglas de negocio y tablas esperadas por pregunta.

JSONL es preferible a CSV porque cada caso contiene listas, metadata, objetos
anidados y varias filas. Mantiene sus tipos numéricos y permite revisar un caso
por línea en Git. No se versionan copias `candidates`, `quarantine`,
`refreshed` ni baselines parciales; Git y los experimentos de LangSmith ya
conservan el historial.

El dataset remoto estable se llama:

```text
dat_ia_test_golden_v2
```

El nombre local y el remoto son identificadores distintos. Los UUID de los 30
ejemplos se derivan de `dataset + case_id`, por lo que sincronizar cambios
actualiza los casos existentes y no crea duplicados.

## Contrato de cada caso

Cada línea contiene esta estructura:

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
    "dataset_version": "2.1.0",
    "domain": "Órdenes",
    "result_type": "scalar",
    "business_rule": "Contar cada order_id una sola vez.",
    "quality_status": "verified_against_official_postgresql",
    "quality_notes": [],
    "omitted_optional_columns": [],
    "source_row": 1
  }
}
```

La comparación de resultados es exacta en cantidad de filas y hechos
esperados, sin usar `comparison_mode`. El evaluador:

- exige que `row_count` coincida;
- compara todas las filas, sin depender de su orden;
- acepta alias de columnas porque compara los valores requeridos;
- permite columnas adicionales en la respuesta;
- normaliza números, UUID y periodos mensuales;
- aplica `numeric_tolerance`.

El golden set exige solo lo necesario para responder la pregunta. Las columnas
retiradas del contrato se registran en `metadata.omitted_optional_columns`.
Esto aplica a `golden_015`, `golden_022`, `golden_025` y `golden_029`.

## Independencia de LangSmith

LangSmith es opcional para el flujo normal:

| Escenario | Configuración | Resultado |
|---|---|---|
| Ejecutar la API | Sin variables LangSmith | La API funciona y `/ready` devuelve `langsmith: not_connected`. |
| Trazar una pregunta | Bandera y API key | La llamada aparece como una traza normal. |
| Evaluar el golden set | Bandera, API key y script manual | Se sincronizan 30 ejemplos y se crea un experimento. |

Variables mínimas:

```text
USE_LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
```

`LANGSMITH_PROJECT` es opcional y usa `dat_ia_test`. La tasa
`LANGSMITH_TRACING_SAMPLING_RATE` también es opcional y usa `1.0`. Las variables
se leen al importar la aplicación; hay que reiniciar Uvicorn después de
cambiarlas.

Si LangSmith está deshabilitado, no se crea el cliente, no se envían trazas y
la API continúa funcionando. `/ready` comprueba que la bandera y la API key
existan; no realiza un ping remoto adicional.

## Requisitos de la evaluación

- `GOOGLE_API_KEY` configurada;
- `DATABASE_URL` conectada a la PostgreSQL oficial;
- API Dat-IA iniciada;
- `USE_LANGSMITH_TRACING=true` y `LANGSMITH_API_KEY` para publicar;
- cuota suficiente del proveedor para 30 preguntas.

Cloudflare no elimina la dependencia de Gemini: solo sustituye al generador
SQL, mientras el optimizer, el juez, la síntesis y los embeddings siguen usando
Google.

## 1. Iniciar la API

Desde la raíz del repositorio:

```cmd
uv run --env-file .env uvicorn app.main:app --reload
```

Comprobarla desde otra consola:

```cmd
curl http://127.0.0.1:8000/ready
```

La evaluación requiere al menos:

```json
{
  "status": "ok",
  "database": "connected",
  "langsmith": "connected"
}
```

## 2. Validar sin consumir las 30 preguntas

```cmd
uv run --env-file .env python -m scripts.evaluate_langsmith_golden_set --dry-run
```

`--dry-run` realiza únicamente estas comprobaciones:

1. abre el JSONL y valida el contrato de cada caso;
2. confirma que hay 30 casos habilitados y sin IDs duplicados;
3. consulta `/ready`;
4. exige `database: connected`.

No llama a `/query/answer`, no consume 30 inferencias, no sincroniza el dataset
remoto y no crea un experimento.

## 3. Ejecutar el golden set completo

Con la API activa:

```cmd
uv run --env-file .env python -m scripts.evaluate_langsmith_golden_set --timeout 180 --max-concurrency 1
```

El script:

1. valida el archivo canónico;
2. comprueba `/ready`;
3. crea o actualiza `dat_ia_test_golden_v2`;
4. llama una vez a `/query/answer` por pregunta;
5. ejecuta cinco evaluadores deterministas;
6. crea un experimento con prefijo `dat_ia_test-golden-v2`.

`--max-concurrency 1` es la opción recomendada para reducir límites de cuota y
facilitar el diagnóstico. Cada ejecución crea un experimento nuevo, pero no
duplica los ejemplos del dataset.

### Cuándo usar `--skip-sync`

La ejecución normal sincroniza primero Git con LangSmith. Debe usarse después
de cualquier cambio en pregunta, SQL, respuesta esperada o metadata.

`--skip-sync` evalúa directamente la copia que ya existe en LangSmith. Solo es
seguro cuando se confirmó que el archivo local y el dataset remoto son
idénticos. Los experimentos históricos nunca se sobrescriben.

El caso `golden_010` fue actualizado después de la última ejecución válida para
esperar la tasa oficial `0.96`. Por ello, la próxima ejecución debe hacerse sin
`--skip-sync`.

## Métricas

| Nombre en LangSmith | Qué comprueba |
|---|---|
| `result_facts_match_expected` | Filas y hechos esperados en `data`; es la métrica principal. |
| `answer_contains_expected_facts` | Hechos esperados en la respuesta redactada. |
| `reported_source_tables_match_expected` | Coincidencia exacta de tablas reportadas. |
| `response_status_matches_expected` | Estado funcional esperado. |
| `generated_sql_is_read_only` | SQL limitado a `SELECT` o CTE sin mutaciones. |

Una puntuación `0` en una métrica no explica por sí sola todo el caso. Para el
diagnóstico deben revisarse conjuntamente `sql`, `data`, `answer`, `sources`,
`status`, el optimizer y las tablas recuperadas.

El reporte vigente está en
[../../reports/dat_ia_golden_set_v2_latest.md](../../reports/dat_ia_golden_set_v2_latest.md).
La última medición válida obtuvo 17/30 en la métrica principal. Una ejecución
posterior con `0%` se excluyó porque agotó los tokens antes de producir salidas
evaluables.

## Auditorías locales regenerables

Las auditorías detalladas escriben JSON en `reports/archive/`. Esa carpeta está
ignorada por Git: permite investigar sin acumular reportes generados en cada
commit.

Validar los 30 SQL de referencia contra PostgreSQL:

```cmd
uv run --env-file .env python -m scripts.validate_golden_set_references
```

Salida local:

```text
reports/archive/dat_ia_golden_v2_reference_validation.json
```

Clasificaciones posibles:

- `correct`: SQL ejecutado y resultado coincidente;
- `golden_set_outdated`: SQL ejecutado con datos diferentes;
- `reference_sql_error`: SQL inválido o no ejecutable.

Auditar únicamente el DDL anterior:

```cmd
uv run --env-file .env python -m scripts.refresh_ddl_from_database --audit-only --ddl-path data/ddl_old.json
```

Salida local:

```text
reports/archive/dat_ia_ddl_validation.json
```

Refrescar un candidato del golden set sin modificar el canónico:

```cmd
uv run --env-file .env python -m scripts.refresh_golden_set_expected_results
```

El candidato `dat_ia_golden_set_v2_refreshed.jsonl` está ignorado por Git. Solo
una revisión explícita del equipo puede promover sus datos al archivo canónico.

## Limpiar Chroma para una nueva línea base

`chroma_db` y `chroma_data` no contienen PostgreSQL: guardan el índice DDL y
Query Memory. Detener la API antes de borrar la carpeta correspondiente.

Ejecución local:

```powershell
Remove-Item -LiteralPath .\chroma_db -Recurse -Force
uv run --env-file .env uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Docker Compose:

```powershell
docker compose down
Remove-Item -LiteralPath .\chroma_data -Recurse -Force
docker compose --env-file .env up --build
```

No se deben borrar ambas carpetas; se elimina solo la utilizada por el modo de
ejecución actual. Si `CHROMA_HOST` apunta a un servidor externo, estas órdenes
no limpian ese servidor.

Después de recrear Chroma, comprobar Query Memory:

```cmd
curl http://127.0.0.1:8000/memory/v2/stats
```

Una evaluación completa volverá a poblar la memoria con las consultas que
lleguen a ejecución exitosa.

## Pruebas de código

```cmd
uv run pytest
uv run ruff check app scripts tests
```

Estas pruebas son locales, deterministas y no consumen LangSmith, PostgreSQL ni
el modelo. La evaluación de 30 preguntas permanece manual.

## Problemas comunes

`LangSmith no está configurado`

: Definir `USE_LANGSMITH_TRACING=true` y `LANGSMITH_API_KEY`, y reiniciar la
  API.

`database='not_configured'`

: Revisar `DATABASE_URL` y reiniciar Uvicorn.

El experimento remoto usa referencias anteriores

: Ejecutar sin `--skip-sync` para sincronizar el archivo canónico.

La corrida muestra `0%` y hay errores de cuota o tokens

: No usarla como baseline de calidad. Restablecer cuota, comprobar una pregunta
  individual y volver a ejecutar el experimento completo.

Hay 30 runs completed y 29 runs evaluated

: Revisar el caso sin feedback. Suele indicar que una salida, un evaluador o el
  proveedor falló antes de publicar todas las métricas; no debe imputarse como
  una respuesta incorrecta sin inspeccionar la traza.

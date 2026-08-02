# Dat-IA: agente analista de datos Text-to-SQL

Dat-IA convierte preguntas de negocio en lenguaje natural a SQL de solo
lectura, valida la consulta, la ejecuta sobre PostgreSQL y redacta una respuesta
trazable. El proyecto combina FastAPI, Gemini, LangChain, ChromaDB, guardrails
de seguridad y observabilidad opcional con LangSmith.

La base PostgreSQL conectada es la fuente oficial de estructura y datos. El
catálogo operativo contiene 16 tablas del dominio Olist y extensiones de
negocio para logística, soporte, devoluciones, precios y promociones.

## Flujo principal

`POST /query/answer` ejecuta el proceso completo:

1. clasifica la pregunta con SQLPromptShield;
2. normaliza la intención mediante el optimizer;
3. recupera tablas relevantes del índice DDL en Chroma;
4. recupera ejemplos compatibles de Query Memory V2;
5. genera, valida y juzga SQL de solo lectura;
6. ejecuta la consulta en PostgreSQL;
7. valida los resultados y redacta la respuesta final;
8. registra la traza en LangSmith cuando está habilitado.

La generación SQL usa Gemini de forma predeterminada. Cloudflare Workers AI
puede sustituir únicamente al generador SQL; el optimizer, el juez y la síntesis
siguen necesitando `GOOGLE_API_KEY`.

## Requisitos

- Python 3.11 o 3.12;
- [uv](https://docs.astral.sh/uv/);
- acceso a la PostgreSQL oficial mediante `DATABASE_URL`;
- una clave de Gemini en `GOOGLE_API_KEY`;
- Docker Desktop, solo si se usará Docker Compose.

Instalar las dependencias:

```powershell
uv sync
```

## Configuración

Crear el archivo local de variables a partir de la plantilla:

```powershell
Copy-Item .env.example .env
```

En CMD:

```cmd
copy .env.example .env
```

Variables principales:

| Variable | Requerida | Propósito |
|---|---|---|
| `DATABASE_URL` | Para ejecutar respuestas | Conexión de solo lectura a PostgreSQL/Supabase. |
| `GOOGLE_API_KEY` | Sí | Gemini y embeddings. |
| `APP_ENV` | No | Etiqueta de ambiente; el valor predeterminado es `test`. |
| `USE_CLOUDFLARE_LLM` | No | Activa Cloudflare solo para generar SQL. |
| `CLOUDFLARE_ACCOUNT_ID` | Con Cloudflare | Cuenta de Workers AI. |
| `CLOUDFLARE_API_KEY` | Con Cloudflare | Credencial de Workers AI. |
| `USE_LANGSMITH_TRACING` | No | Activa el envío de trazas. |
| `LANGSMITH_API_KEY` | Con tracing | Credencial de LangSmith. |
| `LANGSMITH_PROJECT` | No | Proyecto remoto; usa `dat_ia_test` por defecto. |
| `LANGSMITH_TRACING_SAMPLING_RATE` | No | Proporción entre `0.0` y `1.0`; usa `1.0` por defecto. |

La lista completa y segura para copiar está en `.env.example`. El archivo
`.env` contiene secretos, está ignorado por Git y no debe enviarse al
repositorio.

## Ejecutar localmente

Desde la raíz del repositorio:

```powershell
uv run --env-file .env uvicorn app.main:app --reload
```

Direcciones útiles:

- API: `http://127.0.0.1:8000/`
- interfaz web: `http://127.0.0.1:8000/ui/`
- Swagger: `http://127.0.0.1:8000/docs`
- disponibilidad: `http://127.0.0.1:8000/ready`

Prueba rápida en CMD:

```cmd
curl http://127.0.0.1:8000/ready
```

Para ejecutar una pregunta completa:

```cmd
curl -X POST http://127.0.0.1:8000/query/answer -H "Content-Type: application/json; charset=utf-8" -d "{\"question\":\"¿Cuál fue el total vendido por mes en 2018?\"}"
```

`/ready` informa `database: connected` cuando PostgreSQL está disponible y
`langsmith: connected` cuando la bandera y la API key de LangSmith están
configuradas. Este último valor comprueba configuración local; no realiza un
ping adicional al servicio remoto.

## Endpoints

| Método y ruta | Uso |
|---|---|
| `GET /` | Modelo, embedding y cantidad de documentos DDL indexados. |
| `GET /health` | Estado básico del proceso. |
| `GET /ready` | Disponibilidad de PostgreSQL y configuración de LangSmith. |
| `POST /query/optimize` | Intención, métricas, filtros, agrupaciones y tablas sugeridas. |
| `POST /query/json` | Generación de SQL sin ejecutar la consulta. |
| `POST /query/answer` | Flujo integral con ejecución y respuesta redactada. |
| `POST /query/shield` | Clasificación aislada de seguridad. |
| `POST /ingest` | Indexación manual de un catálogo DDL en JSON UTF-8. |
| `GET /memory/v2/stats` | Estadísticas locales de Query Memory V2. |
| `POST /memory/v2/search` | Inspección semántica de memorias sin registrar su uso. |

Los esquemas exactos de entrada y salida están disponibles en Swagger.

## Docker Compose

Docker Compose es la forma más sencilla de obtener una ejecución reproducible
si Docker Desktop ya está instalado. Lee `.env`, construye la imagen, publica
el puerto 8000 y persiste Chroma en `chroma_data/`:

```powershell
docker compose --env-file .env up --build
```

Para detener y eliminar el contenedor:

```powershell
docker compose down
```

La primera construcción puede tardar por las dependencias de PyTorch y el
modelo de SQLPromptShield. `chroma_data/` es estado local ignorado por Git; la
base PostgreSQL no se almacena allí.

## Pruebas

Las pruebas unitarias no requieren LangSmith ni ejecutan el golden set:

```powershell
uv run pytest
uv run ruff check app scripts tests
```

La evaluación end-to-end de 30 preguntas es manual porque consume el modelo,
consulta PostgreSQL y crea un experimento remoto. El procedimiento completo se
encuentra en [tests/evaluation/README.md](tests/evaluation/README.md).

El único reporte vigente de calidad está en
[reports/dat_ia_golden_set_v2_latest.md](reports/dat_ia_golden_set_v2_latest.md).
Una ejecución con `0%` provocada por falta de cuota o tokens no debe usarse como
medición funcional.

## Datos y persistencia local

- `data/ddl.json`: catálogo activo de 16 tablas.
- `data/ddl_old.json`: snapshot anterior conservado para trazabilidad.
- `chroma_db/`: DDL vectorizado y Query Memory en ejecución local.
- `chroma_data/`: la misma persistencia cuando se usa Docker Compose.

Chroma no contiene las filas de negocio de PostgreSQL. Ambas carpetas locales
están ignoradas por Git y se recrean a partir del DDL cuando corresponde. Los
detalles del catálogo están en [data/README.md](data/README.md).

## Estructura relevante

```text
app/
  evaluation/       Contrato y evaluadores del golden set
  formatting/       Tabla y etiquetas de resultados
  memory/           Query Memory V2
  observability/    Integración opcional con LangSmith
  optimizer/        Normalización de preguntas
  validation/       Validador, juez y guardrails
data/                DDL activo y snapshot anterior
scripts/             Evaluación y auditorías manuales
tests/evaluation/    Golden set canónico y documentación
reports/             Evidencia de calidad vigente y benchmarks estables
```

## CI

GitHub Actions valida Ruff, Pytest, la construcción de Docker y el endpoint
`/health`. La evaluación LangSmith no forma parte de CI para evitar consumo no
controlado de cuota y dependencia de servicios externos.

## Equipo

- Stefano Ñuflo Paucar — Ingeniero de Datos
- Rommel Paredes Banda — Ingeniero DevOps
- Rolando Maycol Rodriguez Mallqui — Científico de Datos
- Marcelo Sebastian Chavez Cisneros — Ingeniero MLOps
- Yobel Bañes — Científico de Datos

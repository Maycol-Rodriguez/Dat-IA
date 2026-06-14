# Dat-IA: Agente Analista de Datos (Text-to-SQL)

## Descripción del Proyecto
Las organizaciones generan grandes volúmenes de datos, pero existe una brecha estructural entre los tomadores de decisiones y los equipos técnicos. Para responder preguntas de negocio no rutinarias, se suele depender de analistas de datos, tableros estáticos u hojas de cálculo descentralizadas, lo que genera retrasos operativos y pérdida de confiabilidad.

**Dat-IA** es un agente de analítica de datos basado en Inteligencia Artificial Generativa y prácticas de MLOps. Su objetivo es democratizar el acceso a la información empresarial, permitiendo a los usuarios interactuar con bases de datos relacionales mediante lenguaje natural, sin necesidad de conocer SQL.

## Objetivos
* **Negocio:** Reducir el tiempo de respuesta entre la formulación de una pregunta de negocio y la obtención de información confiable, garantizando auditabilidad.
* **Técnico:** Diseñar, desplegar y evaluar un sistema modular (Agentes IA + RAG) que traduzca consultas en lenguaje natural a SQL, las ejecute de forma segura y devuelva respuestas explicadas al usuario, aplicando monitoreo continuo.

## Arquitectura de la Solución
La solución se compone de 7 módulos principales:
1. **Interfaz de Usuario:** Recepción de consultas en lenguaje natural y presentación de resultados y visualizaciones.
2. **Filtro de Seguridad:** Subagente responsable de identificar y bloquear prompts maliciosos (SQL injection, fugas de confidencialidad).
3. **Caché Semántico:** Almacenamiento y recuperación de consultas previas mediante búsqueda vectorial para reducir latencia.
4. **Buscador de Catálogo de Datos (RAG):** Identificación de las tablas relevantes en el catálogo de datos mediante bases vectoriales.
5. **Inferencia Text-to-SQL:** Core del sistema que traduce la intención del usuario a una consulta SQL ejecutable (basado en modelos como Defog/SQLCoder).
6. **Ejecución Controlada:** Conexión segura a la base de datos relacional para la extracción de la data requerida.
7. **Parseo y Evaluación:** Validación de resultados, creación de gráficos automáticos y redacción de la respuesta final explicada.

## Stack Tecnológico (Propuesto)
* **Lenguaje:** Python
* **Gestión de Entorno:** Docker & Docker Compose
* **Base de Datos Relacional:** PostgreSQL / MySQL (Dataset: Olist Brazilian E-Commerce)
* **Base de Datos Vectorial:** Chroma DB
* **Modelos IA:** 
* **Observabilidad y MLOps:** LangSmith, Grafana

## Prototipo API

El proyecto incluye una API inicial con FastAPI para validar la base del sistema Dat-IA.

### Ejecutar localmente

```powershell
uv sync
uv run uvicorn app.main:app --reload
```

Abrir la documentación interactiva:

```text
http://127.0.0.1:8000/docs
```

Probar healthcheck:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "service": "dat-ia-api",
  "version": "0.1.0"
}
```

### Ejecutar tests

```powershell
uv run pytest
uv run ruff check app tests
```

### Ejecutar con Docker

```powershell
docker build -t dat-ia-api:local .
docker run --rm -p 8000:8000 dat-ia-api:local
```

### Ejecutar con Docker Compose

```powershell
docker compose up --build
```

Para detener:

```powershell
docker compose down
```

### CI

El repositorio incluye un workflow de GitHub Actions que valida Ruff, Pytest, build de Docker y el endpoint `/health`.


## Dataset Utilizado
Se utilizará el dataset público **Olist Brazilian E-Commerce** (aprox. 100,000 órdenes de 2016-2018). Su estructura relacional de 9 tablas (clientes, órdenes, pagos, productos, etc.) simula a la perfección el ciclo de una transacción retail, permitiendo formular preguntas complejas que requieren múltiples uniones (JOINs) y análisis temporales o geográficos.

## Equipo de Trabajo
* **Stefano Ñuflo Paucar** - Ingeniero de Datos
* **Rommel Paredes Banda** - Ingeniero DevOps
* **Rolando Maycol Rodriguez Mallqui** - Científico de Datos
* **Marcelo Sebastian Chavez Cisneros** - Ingeniero MLOps
* **Yobel Bañes** - Científico de Datos

## Roadmap de Implementación (Gantt)
* **01/06/2026 - 05/06/2026:** Setup, Arquitectura, Dockerización y definición de esquemas.
* **04/06/2026 - 19/06/2026:** Creación de BD relacional, catálogo de datos y scripts de conexión.
* **22/06/2026 - 03/07/2026:** Vectorización, Embeddings y memoria semántica en Chroma DB.
* **22/06/2026 - 17/07/2026:** Desarrollo de Agentes IA (Filtro de seguridad, optimizador y Text-to-SQL).
* **15/07/2026 - 04/08/2026:** UI, formato de resultados y agente de visualización.
* **20/07/2026 - 03/08/2026:** Validación End-to-End e implementación de LangSmith.
* **04/08/2026 - 09/08/2026:** Pruebas integrales y Entrega Final.

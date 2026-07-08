from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "dat-ia-api"


def test_ready_returns_database_not_configured() -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["database"] == "not_configured"


def test_ask_returns_prototype_response() -> None:
    response = client.post(
        "/query/json",
        json={"question": "¿Cuál fue el total vendido por mes?"},
    )

    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "prototype"
    assert body["sql"] == "SELECT 1 AS prototype_result;"


def test_ask_rejects_empty_question() -> None:
    response = client.post("/query/json", json={"question": ""})

    assert response.status_code == 422

def test_memory_stats_returns_response() -> None:
    response = client.get("/memory/stats")

    assert response.status_code == 200
    assert response.json()["collection"] == "query_memory"


def test_memory_search_returns_503_when_memory_is_not_initialized() -> None:
    response = client.post(
        "/memory/search",
        json={
            "question": "Que empresa de transporte tiene mejor cumplimiento?",
            "n_results": 3,
        },
    )

    assert response.status_code == 503

def test_query_optimize_returns_normalized_response() -> None:
    response = client.post(
        "/query/optimize",
        json={
            "question": "Que empresa de transporte tiene mejor cumplimiento?",
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["intent"] == "ranking"
    assert "on_time_rate" in body["metrics"]
    assert "carriers" in body["suggested_tables"]
    assert body["optimizer"] == "rule_based"
    assert body["normalized_question"] == (
        "Listar transportistas ordenados por mayor tasa de cumplimiento de entrega."
    )


def test_query_optimize_rejects_blank_question() -> None:
    response = client.post(
        "/query/optimize",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 422

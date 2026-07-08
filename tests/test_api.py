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


def test_query_json_uses_optimized_question(monkeypatch) -> None:
    from app import main as main_module

    captured = {}

    class FakeCollection:
        def count(self) -> int:
            return 1

    def fake_query_embeddings(
        collection,
        query: str,
        distance_threshold: float = 0.9,
    ):
        _ = collection
        captured["retrieval_query"] = query
        captured["distance_threshold"] = distance_threshold

        return main_module.EmbeddingsResponse(
            tabla=["carriers"],
            descripcion=["Transportistas y tasa de cumplimiento."],
            distance=[0.1],
            ddl="CREATE TABLE carriers (carrier_name text, on_time_rate numeric);",
        )

    def fake_build_rag_response(question: str, ddl: str):
        captured["generation_question"] = question
        captured["ddl"] = ddl

        return main_module.RAGResponse(
            sql=(
                "SELECT carrier_name FROM carriers "
                "ORDER BY on_time_rate DESC LIMIT 1;"
            ),
            sources="carriers",
            confidence_note="Usa la métrica on_time_rate.",
            status="success",
        )

    monkeypatch.setattr(main_module, "text_collection", FakeCollection())
    monkeypatch.setattr(main_module, "query_memory_collection", None)
    monkeypatch.setattr(main_module, "query_embeddings", fake_query_embeddings)
    monkeypatch.setattr(main_module, "build_rag_response", fake_build_rag_response)

    response = client.post(
        "/query/json",
        json={
            "question": "Que empresa de transporte tiene mejor cumplimiento?",
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "success"
    assert captured["retrieval_query"] == (
        "Listar transportistas ordenados por mayor tasa de cumplimiento de entrega."
    )
    assert captured["generation_question"] == (
        "Listar transportistas ordenados por mayor tasa de cumplimiento de entrega."
    )
    assert captured["distance_threshold"] == 0.7

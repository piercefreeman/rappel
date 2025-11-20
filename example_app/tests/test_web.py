from __future__ import annotations

from fastapi.testclient import TestClient

from example_app.web import app


def test_run_task_endpoint_executes_workflow() -> None:
    client = TestClient(app)
    response = client.post("/api/tasks", json={"number": 5})
    assert response.status_code == 200
    payload = response.json()

    assert payload["factorial"] == 120
    assert payload["fibonacci"] == 5
    assert payload["summary"] == "5! is larger, but Fibonacci is 5"

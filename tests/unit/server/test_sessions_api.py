import asyncio

import pytest
from fastapi.testclient import TestClient

from src.server.session.dependencies import set_session_store
from src.server.session.store import SQLiteSessionStore


@pytest.fixture
def client(tmp_path):
    db_path = tmp_path / "sessions_api.db"
    store = SQLiteSessionStore(str(db_path))
    asyncio.run(store.init())
    set_session_store(store)

    from src.server.app import app

    with TestClient(app) as test_client:
        yield test_client

    asyncio.run(store.close())


def test_session_crud_flow(client: TestClient):
    response = client.post("/api/sessions", json={})
    assert response.status_code == 201
    session = response.json()["session"]
    session_id = session["id"]

    response = client.get("/api/sessions")
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id

    response = client.patch(
        f"/api/sessions/{session_id}", json={"title": "测试会话"}
    )
    assert response.status_code == 200
    assert response.json()["title"] == "测试会话"

    response = client.delete(f"/api/sessions/{session_id}")
    assert response.status_code == 200
    assert response.json()["success"] is True

    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert response.json()["sessions"] == []

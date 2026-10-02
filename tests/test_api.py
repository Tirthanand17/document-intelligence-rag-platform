from fastapi.testclient import TestClient
from app.api import app

def test_health_and_ask():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/ask", json={"question": "How long is express shipping?"})
    assert response.status_code == 200
    payload = response.json()
    assert "shipping.txt" in payload["citations"]

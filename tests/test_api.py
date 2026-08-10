import uuid

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/healt")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_signup_and_signin_and_me_flow():
    unique = uuid.uuid4().hex[:8]
    signup_payload = {
        "username": f"tester_{unique}",
        "password": "password123",
        "email": f"tester_{unique}@example.com",
    }
    signup_response = client.post("/api/auth/signup", json=signup_payload)
    assert signup_response.status_code == 201
    assert signup_response.json()["username"] == signup_payload["username"]
    assert signup_response.json()["email"] == signup_payload["email"]

    signin_payload = {"email": signup_payload["email"], "password": "password123"}
    signin_response = client.post("/api/auth/signin", json=signin_payload)
    assert signin_response.status_code == 200
    token_data = signin_response.json()
    assert token_data["token_type"] == "bearer"
    assert "access_token" in token_data

    auth_header = {"Authorization": f"Bearer {token_data['access_token']}"}
    me_response = client.get("/api/auth/me", headers=auth_header)
    assert me_response.status_code == 200
    assert me_response.json() == {"username": "tester", "email": "tester@example.com"}

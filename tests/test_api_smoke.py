from fastapi.testclient import TestClient
from app import app
import joblib

def fake_predict(*args, **kwargs):
    return [1.23]

def test_predict(monkeypatch):
    monkeypatch.setattr(
        joblib,
        "load",
        lambda _: type("M", (), {"predict": fake_predict})()
    )

    client = TestClient(app)
    response = client.post(
        "/predict",
        json={
            "MedInc": 8.3,
            "HouseAge": 21,
            "AveRooms": 6.5,
            "AveBedrms": 1.1,
            "Population": 2400,
            "AveOccup": 3.1,
            "Latitude": 34.2,
            "Longitude": -118.3,
        },
    )

    assert response.status_code == 200
    assert "prediction" in response.json()

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/")
    assert response.status_code == 200

def test_predict():
    payload = {
        "MedInc": 8.3,
        "HouseAge": 21,
        "AveRooms": 6.5,
        "AveBedrms": 1.1,
        "Population": 2400,
        "AveOccup": 3.1,
        "Latitude": 34.2,
        "Longitude": -118.3,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

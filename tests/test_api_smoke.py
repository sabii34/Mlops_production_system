import os
os.environ["SKIP_MODEL_LOAD"] = "1"

from fastapi.testclient import TestClient
import importlib
import joblib

def fake_predict(X):
    return [1.23]

def test_predict(monkeypatch):
    monkeypatch.setattr(
        joblib,
        "load",
        lambda *_: type("M", (), {"predict": lambda self, X: fake_predict(X)})()
    )

    import app as app_module
    importlib.reload(app_module)  # ensure patched load is used
    client = TestClient(app_module.app)

    r = client.post("/predict", json={
        "MedInc": 8.3, "HouseAge": 21, "AveRooms": 6.5, "AveBedrms": 1.1,
        "Population": 2400, "AveOccup": 3.1, "Latitude": 34.2, "Longitude": -118.3
    })
    assert r.status_code == 200
    assert "prediction" in r.json()


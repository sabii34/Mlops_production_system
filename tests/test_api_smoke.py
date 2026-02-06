import os
import importlib
import joblib
from fastapi.testclient import TestClient

def test_predict(monkeypatch):
    os.environ["SKIP_MODEL_LOAD"] = "1"

    class DummyModel:
        def predict(self, X):
            return [1.23]

    monkeypatch.setattr(joblib, "load", lambda *_: DummyModel())

    import app as app_module
    importlib.reload(app_module)

    client = TestClient(app_module.app)

    r = client.post("/predict", json={
        "MedInc": 8.3, "HouseAge": 21, "AveRooms": 6.5, "AveBedrms": 1.1,
        "Population": 2400, "AveOccup": 3.1, "Latitude": 34.2, "Longitude": -118.3
    })
    assert r.status_code == 200

    assert "prediction" in r.json()


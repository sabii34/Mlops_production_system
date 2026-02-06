# app.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MLOps Production Model API")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"

model = None

def get_model():
    global model
    if model is None:
        if SKIP_MODEL_LOAD:
            # tests will monkeypatch joblib.load; no file check needed
            model = joblib.load(MODEL_PATH)
            return model

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

        model = joblib.load(MODEL_PATH)
    return model

class Features(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: Features):
    try:
        m = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    X = pd.DataFrame([payload.model_dump()])
    pred = float(m.predict(X)[0])
    return {"prediction": pred}


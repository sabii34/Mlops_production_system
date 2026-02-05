from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.config.config import PathsConfig

app = FastAPI(title="MLOps Production Model API")

cfg = PathsConfig()

# Load artifacts once at startup
MODEL_PATH = cfg.models_dir / "model.joblib"
PREPROCESSOR_PATH = cfg.models_dir / "preprocessor.joblib"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Model API is running"}

...
@app.post("/predict")
def predict(input_data: HousingInput):
    df = pd.DataFrame([{
        "MedInc": input_data.MedInc,
        "HouseAge": input_data.HouseAge,
        "AveRooms": input_data.AveRooms,
        "AveBedrms": input_data.AveBedrms,
        "Population": input_data.Population,
        "AveOccup": input_data.AveOccup,
        "Latitude": input_data.Latitude,
        "Longitude": input_data.Longitude,
    }])

    X = preprocessor.transform(df)
    prediction = model.predict(X)

    return {"prediction": float(prediction[0])}


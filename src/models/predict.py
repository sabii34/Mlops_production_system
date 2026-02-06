import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models") / "model.joblib"

# Must match the training feature order
FEATURE_ORDER = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

_model = None

def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_one(features: dict) -> float:
    # Build row in correct order
    row = [features[k] for k in FEATURE_ORDER]

    # IMPORTANT: model was trained with numeric column names 0..7
    X = pd.DataFrame([row], columns=list(range(len(FEATURE_ORDER))))

    model = _load_model()
    pred = model.predict(X)[0]
    return float(pred)

from fastapi import FastAPI
from typing import List
import pandas as pd

from api.schemas import TelemetryRow, PredictionResponse
from src.models.predict import RULPredictor

app = FastAPI(title="Predictive Vehicle Maintenance API")

# Load model ONCE at startup
predictor = RULPredictor()


@app.get("/")
def health():
    return {"status": "ok", "message": "RUL Prediction API running"}


@app.post("/predict", response_model=List[PredictionResponse])
def predict(rows: List[TelemetryRow]):

    # Convert incoming JSON to DataFrame
    data = pd.DataFrame([r.dict() for r in rows])

    # Run inference
    preds_df = predictor.predict_from_dataframe(data)

    # Convert back to JSON
    results = preds_df.to_dict(orient="records")

    return results

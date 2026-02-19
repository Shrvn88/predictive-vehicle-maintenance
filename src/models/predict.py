import joblib
import json
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.data.feature_engineering import add_rolling_features
from src.data.preprocessing import build_features, scale_features

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "gradient_boosting.pkl"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "scalers" / "scaler.pkl"
FEATURE_PATH = PROJECT_ROOT / "artifacts" / "models" / "feature_columns.json"


class RULPredictor:

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

        with open(FEATURE_PATH, "r") as f:
            self.feature_cols = json.load(f)

    def predict_from_dataframe(self, df):
        
        df = df.sort_values("cycle").reset_index(drop=True)

        df = df.copy()

        df = add_rolling_features(df)
        
        df = df.fillna(0)

        df = df[df["cycle"] > 30]

        X_raw = build_features(df)

        # enforce training schema
        X_raw = X_raw.reindex(columns=self.feature_cols, fill_value=0)

        X = scale_features(X_raw, fit=False, scaler_path=SCALER_PATH)

        preds = self.model.predict(X)

        preds = np.clip(preds, 0, 125)
        
        # preds = np.minimum.accumulate(preds[::-1])[::-1]

        print(preds[:5], preds[-5:])

        df["predicted_RUL"] = preds

        return df[["unit", "cycle", "predicted_RUL"]]

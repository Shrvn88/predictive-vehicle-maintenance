import joblib
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from src.data.cmapss_loader import load_cmapss
from src.data.feature_engineering import add_rolling_features
from src.config import PROJECT_ROOT
from src.data.preprocessing import build_features, scale_features


MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "scalers" / "scaler.pkl"
FEATURE_PATH = PROJECT_ROOT / "artifacts" / "models" / "feature_columns.json"


MODEL_DIR.mkdir(parents=True, exist_ok=True)
SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)


def train():

    # -------- Load + Feature Engineering --------
    train_df, _ = load_cmapss()
    train_df = add_rolling_features(train_df)
    
    train_df = train_df[train_df["cycle"] > 30]


    groups = train_df["unit"]

    # -------- Unit-based split --------
    splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(train_df, groups=groups))

    train_split = train_df.iloc[train_idx]
    val_split = train_df.iloc[val_idx]

    # -------- Preprocess --------
    X_train_raw = build_features(train_split)
    y_train = train_split["RUL"]

    X_train = scale_features(X_train_raw, fit=True, scaler_path=SCALER_PATH)

    with open(FEATURE_PATH, "w") as f:
        json.dump(list(X_train_raw.columns), f)

    X_val_raw = build_features(val_split)
    X_val = scale_features(X_val_raw, fit=False, scaler_path=SCALER_PATH)
    y_val = val_split["RUL"]
    

    # -------- Models --------
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    gbr = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )


    models = {
        "random_forest": rf,
        "gradient_boosting": gbr
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)

        print(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}")

        joblib.dump(model, MODEL_DIR / f"{name}.pkl")

        results[name] = {"rmse": rmse, "mae": mae}

    return results


if __name__ == "__main__":
    train()

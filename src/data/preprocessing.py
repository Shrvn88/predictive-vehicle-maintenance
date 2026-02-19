import joblib
import numpy as np

DROP_SENSORS = [
    "sensor_1","sensor_5","sensor_6",
    "sensor_10","sensor_16","sensor_18","sensor_19"
]

DROP_COLS = ["RUL", "unit", "cycle"]


def build_features(df):
    df = df.copy()

    # Remove dead sensors if present
    existing = [c for c in DROP_SENSORS if c in df.columns]
    df.drop(columns=existing, inplace=True)

    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    return X


def scale_features(X, fit=False, scaler_path=None):
    if fit:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

    return X_scaled

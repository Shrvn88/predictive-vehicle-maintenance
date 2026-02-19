import pandas as pd

WINDOWS = [5, 15, 30]

def add_rolling_features(df):

    df = df.copy()
    df = df.sort_values(["unit", "cycle"])
    
    USE_SENSORS = [
        "sensor_2","sensor_3","sensor_4",
        "sensor_7","sensor_8","sensor_11",
        "sensor_12","sensor_15","sensor_17",
        "sensor_20","sensor_21"
    ]
    
    for s in USE_SENSORS:
        if s not in df.columns:
            df[s] = 0.0


    sensor_cols = USE_SENSORS

    new_features = {}

    grouped = df.groupby("unit")

    for w in WINDOWS:
        rolled = grouped[sensor_cols].rolling(w, min_periods=1)

        mean_df = rolled.mean().reset_index(level=0, drop=True)
        std_df = rolled.std().reset_index(level=0, drop=True)

        for col in sensor_cols:
            new_features[f"{col}_mean_{w}"] = mean_df[col]
            new_features[f"{col}_std_{w}"] = std_df[col]

    # Delta features
    delta_df = grouped[sensor_cols].diff().fillna(0)

    for col in sensor_cols:
        new_features[f"{col}_delta"] = delta_df[col]

    # Concatenate all new columns at once (important)
    feature_df = pd.DataFrame(new_features, index=df.index)

    df = pd.concat([df, feature_df], axis=1)

    return df

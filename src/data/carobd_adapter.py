import pandas as pd
import numpy as np

def adapt_carobd(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.reset_index(drop=True)
    
    out = pd.DataFrame()

    # Unit + cycle
    out["unit"] = 1
    out["cycle"] = np.arange(1, len(df) + 1)

    # Primary mappings
    out["sensor_2"] = df["ENGINE_RPM ()"]
    out["sensor_3"] = df["INTAKE_MANIFOLD_PRESSURE ()"]
    out["sensor_4"] = df["ENGINE_LOAD ()"]

    out["sensor_7"] = df["VEHICLE_SPEED ()"]
    out["sensor_8"] = df["THROTTLE ()"]

    out["sensor_11"] = df["COOLANT_TEMPERATURE ()"]
    out["sensor_12"] = df["INTAKE_AIR_TEMP ()"]

    # ---- Synthetic approximations (required by CMAPSS model) ----
    out["sensor_9"] = df["ABSOLUTE_THROTTLE_B ()"]          # proxy airflow
    out["sensor_13"] = df["FUEL_TANK ()"]                  # proxy fuel state
    out["sensor_14"] = df["TIMING_ADVANCE ()"]             # proxy combustion timing

    # Safety cleanup
    out = out.replace([np.inf, -np.inf], 0)
    out = out.fillna(0)

    return out

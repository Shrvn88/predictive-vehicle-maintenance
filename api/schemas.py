from pydantic import BaseModel
from typing import List, Optional


class TelemetryRow(BaseModel):
    unit: int
    cycle: int

    # Only include sensors you actually use
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float


class PredictionResponse(BaseModel):
    unit: int
    cycle: int
    predicted_RUL: float

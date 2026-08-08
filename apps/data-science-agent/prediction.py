"""vehicle prediction helpers (arima + random forest)."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

brands = ["Seat"]
models = ["Leon"]
locations = ["CW", "RT", "S", "KA", "BB"]
conditions = [
    "Normal",
    "Free",
    "Traffic",
    "Emergency Braking",
    "Normal Icy Road",
    "Free Accelaration",
    "Traffic Jam Measurement error",
]

ARIMA_FEATURE_COLUMNS = [
    "engine_coolant_temp",
    "intake_manifold_pressure",
    "engine_rpm",
    "source",
    "destination",
    "condition",
]
CONDITION_FEATURE_COLUMNS = [
    "vehicle_speed",
    "source",
    "destination",
    "hour",
    "minute",
]


def transform_brand(brand: str) -> int:
    return brands.index(brand)


def transform_model(model: str) -> int:
    return models.index(model)


def transform_location(location: str) -> int:
    return locations.index(location)


def transform_condition(condition: str) -> int:
    return conditions.index(condition)


def _model_path(name: str) -> Path:
    return Path(__file__).resolve().parent / name


def get_arima_model():
    with _model_path("arima_model.pkl").open("rb") as handle:
        return pickle.load(handle)


def get_random_forest_model():
    with _model_path("random_forest_model.pkl").open("rb") as handle:
        return pickle.load(handle)


def get_data_for_arima_model(
    engine_coolant_temp,
    intake_manifold_pressure,
    engine_rpm,
    source,
    destination,
    condition,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "engine_coolant_temp": [engine_coolant_temp],
            "intake_manifold_pressure": [intake_manifold_pressure],
            "engine_rpm": [engine_rpm],
            "source": [source],
            "destination": [destination],
            "condition": [condition],
        }
    )


def get_data_for_categorical_model(vehicle_speed, source, destination, hour, minute) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vehicle_speed": [vehicle_speed],
            "source": [source],
            "destination": [destination],
            "hour": [hour],
            "minute": [minute],
        }
    )


def infer_arima_model(frame: pd.DataFrame):
    model = get_arima_model()
    forecast = model.forecast(steps=len(frame), exog=frame[ARIMA_FEATURE_COLUMNS])
    return forecast.iloc[0]


def infer_forest_model(frame: pd.DataFrame):
    model = get_random_forest_model()
    # use values to avoid feature-name mismatch with models trained on legacy column labels
    return model.predict(frame[CONDITION_FEATURE_COLUMNS].to_numpy())[0]

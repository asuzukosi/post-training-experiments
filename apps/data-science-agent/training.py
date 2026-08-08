"""training and analytics helpers extracted from experiments notebook."""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ARIMA_FEATURE_COLUMNS = [
    "Engine Coolant Temperature [°C]",
    "Intake Manifold Absolute Pressure [kPa]",
    "Engine RPM [RPM]",
    "source",
    "destination",
    "condition",
]
ARIMA_TARGET = "Vehicle Speed Sensor [km/h]"
CONDITION_FEATURE_COLUMNS = [
    "Vehicle Speed Sensor [km/h]",
    "source",
    "destination",
    "hour",
    "minute",
]


def plot_correlation_heatmap(df: pd.DataFrame, drop_columns: list[str] | None = None) -> None:
    """plot a correlation heatmap for numeric columns."""
    working = df.drop(columns=drop_columns or [], errors="ignore")
    corr_matrix = working.corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("correlation heatmap")
    plt.show()


def plot_speed_by_day(df: pd.DataFrame) -> None:
    """plot vehicle speed time series for each day and route/condition pair."""
    for day in df["date"].unique():
        plt.figure(figsize=(10, 6))
        plt.title(f"time series plot for {day}")
        plt.xlabel("time")
        plt.ylabel("vehicle speed sensor [km/h]")

        day_df = df[df["date"] == day]
        unique_pairs = day_df[["source", "destination", "condition"]].drop_duplicates()
        for _, row in unique_pairs.iterrows():
            route_df = day_df[
                (day_df["source"] == row["source"])
                & (day_df["destination"] == row["destination"])
                & (day_df["condition"] == row["condition"])
            ]
            label = f"{row['source']}-{row['destination']}-{row['condition']}"
            plt.plot(route_df["Time"], route_df["Vehicle Speed Sensor [km/h]"], label=label)

        plt.grid(True)
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()


def get_category_maps(df: pd.DataFrame) -> dict[str, list]:
    """return unique brand/model/location/condition lists for factorization."""
    return {
        "brands": list(df["brand"].unique()),
        "models": list(df["model"].unique()),
        "locations": list(set(list(df["source"].unique()) + list(df["destination"].unique()))),
        "conditions": list(df["condition"].unique()),
    }


def factorize_categoricals(
    df: pd.DataFrame,
    brands: list[str],
    models: list[str],
    locations: list[str],
    conditions: list[str],
) -> pd.DataFrame:
    """copy df and replace categorical columns with integer indexes."""
    out = df.copy()
    out["brand"] = out["brand"].apply(lambda value: brands.index(value))
    out["model"] = out["model"].apply(lambda value: models.index(value))
    out["source"] = out["source"].apply(lambda value: locations.index(value))
    out["destination"] = out["destination"].apply(lambda value: locations.index(value))
    out["condition"] = out["condition"].apply(lambda value: conditions.index(value))
    return out


def prepare_arima_frame(df: pd.DataFrame) -> pd.DataFrame:
    """keep arima features/target and drop incomplete rows."""
    columns = ["Time", ARIMA_TARGET, *ARIMA_FEATURE_COLUMNS]
    out = df[columns].dropna(how="any").copy()
    return out


def train_arima(
    df: pd.DataFrame,
    order: tuple[int, int, int] = (1, 1, 1),
    train_ratio: float = 0.8,
):
    """fit an arima model with exogenous features. returns model, train, test."""
    frame = prepare_arima_frame(df)
    train_size = int(len(frame) * train_ratio)
    train_data = frame.iloc[:train_size]
    test_data = frame.iloc[train_size:]

    model = sm.tsa.ARIMA(
        train_data[ARIMA_TARGET],
        order=order,
        exog=train_data[ARIMA_FEATURE_COLUMNS],
    )
    fit_model = model.fit()
    return fit_model, train_data, test_data


def forecast_arima(fit_model, exog_df: pd.DataFrame) -> pd.Series:
    """forecast vehicle speed using exogenous feature rows."""
    return fit_model.forecast(steps=len(exog_df), exog=exog_df[ARIMA_FEATURE_COLUMNS])


def plot_arima_forecast(test_data: pd.DataFrame, forecast: pd.Series) -> None:
    """compare actual vs predicted vehicle speed."""
    plt.figure(figsize=(10, 6))
    plt.title("time series plot")
    plt.xlabel("time")
    plt.ylabel("vehicle speed sensor [km/h]")
    plt.plot(test_data["Time"], test_data[ARIMA_TARGET], label="actual progression")
    plt.plot(test_data["Time"], forecast, label="predicted progression")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def prepare_condition_frame(df: pd.DataFrame) -> pd.DataFrame:
    """add hour/minute features and keep condition-classifier columns."""
    out = df.copy()
    out["hour"] = out["Time"].dt.hour
    out["minute"] = out["Time"].dt.minute
    return out[["condition", *CONDITION_FEATURE_COLUMNS]].dropna(how="any")


def train_condition_classifier(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """train a random forest condition classifier. returns model, accuracy, split."""
    frame = prepare_condition_frame(df)
    x = frame.drop(columns=["condition"])
    y = frame["condition"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = (y_pred == y_test).mean()
    return classifier, accuracy, (x_train, x_test, y_train, y_test)


def save_pickle(obj, path: str | Path) -> None:
    path = Path(path)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path: str | Path):
    path = Path(path)
    with path.open("rb") as handle:
        return pickle.load(handle)

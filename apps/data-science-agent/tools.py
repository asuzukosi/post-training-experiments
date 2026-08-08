"""tools for the data science agent."""
from __future__ import annotations
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

from config import get_model_id
from dataset import get_dataset
from prediction import (
    conditions,
    get_data_for_arima_model,
    get_data_for_categorical_model,
    infer_arima_model,
    infer_forest_model,
    transform_condition,
    transform_location,
)


def get_tavily_tool(max_results: int = 1) -> TavilySearchResults:
    """create a tavily search tool."""
    return TavilySearchResults(max_results=max_results)


class VehicleVelocityPredictionSchema(BaseModel):
    engine_coolant_temp: float = Field(description="engine coolant temperature")
    intake_manifold_pressure: float = Field(description="intake manifold absolute pressure")
    engine_rpm: float = Field(description="engine rpm")
    source: str = Field(description="source location")
    destination: str = Field(description="destination location")
    condition: str = Field(
        description=(
            "condition, one of: Normal, Free, Traffic, Emergency Braking, "
            "Normal Icy Road, Free Accelaration, Traffic Jam Measurement error"
        )
    )


@tool("predict-vehicle-velocity", args_schema=VehicleVelocityPredictionSchema)
def predict_vehicle_velocity(
    engine_coolant_temp,
    intake_manifold_pressure,
    engine_rpm,
    source,
    destination,
    condition,
) -> str:
    """predict vehicle velocity from engine and route features."""
    frame = get_data_for_arima_model(
        float(engine_coolant_temp),
        float(intake_manifold_pressure),
        float(engine_rpm),
        transform_location(source),
        transform_location(destination),
        transform_condition(condition),
    )
    result = infer_arima_model(frame)
    return f"{result} Km/h"


class VehicleConditionPredictionSchema(BaseModel):
    vehicle_speed: float = Field(description="vehicle speed in km/h")
    source: str = Field(description="source location")
    destination: str = Field(description="destination location")
    hour: int = Field(description="hour of measurement")
    minute: int = Field(description="minute of measurement")


@tool("predict-vehicle-condition", args_schema=VehicleConditionPredictionSchema)
def predict_vehicle_condition(vehicle_speed, source, destination, hour, minute) -> str:
    """predict driving condition from speed, route, and time."""
    frame = get_data_for_categorical_model(
        float(vehicle_speed),
        transform_location(source),
        transform_location(destination),
        int(hour),
        int(minute),
    )
    prediction = infer_forest_model(frame)
    return f"The predicted condition is {conditions[prediction]}"


def _run_pandas_expression(expression: str) -> str:
    df = get_dataset()
    result = eval(expression, {"__builtins__": {}}, {"df": df, "pd": pd})
    return str(result)


class DataSetQuestionAnswerSchema(BaseModel):
    query: str = Field(description="natural language question about the dataset")


@tool("dataset-question-answer", args_schema=DataSetQuestionAnswerSchema)
def dataset_question_answer(query: str) -> str:
    """answer questions about the vehicle telemetry dataset using pandas."""
    df = get_dataset()
    llm = init_chat_model(get_model_id(), temperature=0)
    prompt = dedent(
        f"""
        you have a pandas dataframe named df.
        columns and dtypes:
        {df.dtypes.to_string()}

        sample rows:
        {df.head(3).to_string()}

        write ONE python expression using df (and pd if needed) that answers:
        {query}

        return only the expression, no markdown, no explanation.
        """
    )
    expression = llm.invoke(prompt).content.strip().strip("`")
    if expression.startswith("python"):
        expression = expression[len("python") :].strip()
    try:
        return _run_pandas_expression(expression)
    except Exception as exc:
        return f"failed to answer with expression `{expression}`: {exc}"


class DataSetDiagramRequestSchema(BaseModel):
    query: str = Field(description="description of the chart to draw from the dataset")


@tool("dataset-diagram-tool", args_schema=DataSetDiagramRequestSchema)
def dataset_diagram_request(query: str) -> str:
    """draw a matplotlib chart from the dataset and save it to disk."""
    df = get_dataset()
    llm = init_chat_model(get_model_id(), temperature=0)
    out_path = Path(__file__).resolve().parent / "diagram.png"
    prompt = dedent(
        f"""
        you have a pandas dataframe named df and matplotlib.pyplot as plt.
        columns:
        {list(df.columns)}

        write python code that plots: {query}
        rules:
        - use df and plt only
        - create a figure
        - save with plt.savefig(r"{out_path}")
        - do not call plt.show()
        return only the code, no markdown.
        """
    )
    code = llm.invoke(prompt).content.strip().strip("`")
    if code.startswith("python"):
        code = code[len("python") :].strip()
    try:
        exec(code, {"__builtins__": {}}, {"df": df, "plt": plt, "pd": pd})
        return f"diagram saved to {out_path}"
    except Exception as exc:
        return f"failed to draw diagram: {exc}"


PREDICTION_TOOLS = [predict_vehicle_velocity, predict_vehicle_condition]
DATASET_TOOLS = [dataset_question_answer, dataset_diagram_request]
AGENT_TOOLS = [*DATASET_TOOLS, *PREDICTION_TOOLS, get_tavily_tool()]

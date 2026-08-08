"""dataset loading helpers for the data science agent."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from ingest_data import DATA_DIRECTORY, load_dataset, postprocess_dataset

_DF: pd.DataFrame | None = None


def get_dataset() -> pd.DataFrame:
    """load or build the postprocessed telemetry dataframe."""
    global _DF
    if _DF is not None:
        return _DF

    cache_path = Path(__file__).resolve().parent / "full_dataframe.pkl"
    if cache_path.exists():
        print("loading dataset")
        with cache_path.open("rb") as handle:
            _DF = pickle.load(handle)
            return _DF

    print("creating dataset")
    frame = postprocess_dataset(load_dataset(DATA_DIRECTORY))
    with cache_path.open("wb") as handle:
        pickle.dump(frame, handle)
    _DF = frame
    return _DF

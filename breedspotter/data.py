"""Utilities for loading CSV data and breed profiles."""
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ENV_DOGS_CSV = "DOGS_CSV"
ENV_BREEDS_CSV = "BREEDS_CSV"

_DEFAULT_DOGS_CSV = "stanford_dogs_metadata.csv"
_DEFAULT_BREEDS_CSV = "breeds_profiles.csv"


def _csv_path(env_name: str, default: str) -> Path:
    from os import getenv
    return Path(getenv(env_name, default)).expanduser().resolve()


def load_metadata() -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(_csv_path(ENV_DOGS_CSV, _DEFAULT_DOGS_CSV))
    breeds = sorted(df["breed"].unique())
    return df, breeds


def load_breed_profiles() -> Dict[str, str]:
    df = pd.read_csv(_csv_path(ENV_BREEDS_CSV, _DEFAULT_BREEDS_CSV))
    return dict(zip(df["breed"], df["text"]))
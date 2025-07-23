# src/thyroid_analysis/data_loader.py

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

def load_raw_data(filename: str) -> pd.DataFrame:
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Raw data file not found: {filepath}")
    return pd.read_csv(filepath)

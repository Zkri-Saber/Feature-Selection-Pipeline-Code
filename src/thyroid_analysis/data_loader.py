
import os
import pandas as pd

def load_dataset(filename: str) -> pd.DataFrame:
    data_path = os.path.join("data", "raw", filename)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    return pd.read_csv(data_path)

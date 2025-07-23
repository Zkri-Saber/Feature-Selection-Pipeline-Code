import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def apply_knn_imputation(df: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Apply KNN imputation to numeric columns only.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = df.copy()
    df_imputed[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df_imputed

def map_diagnostic_group(y: pd.Series) -> pd.Series:
    """
    Map Diagnostic Group labels to integers.
    """
    mapping = {
        0.0: 0,   # No Disease
        1.0: 1,   # Hyperthyroidism
        2.0: 2,   # Euthyroid
        3.0: 3    # Hypothyroidism
    }
    return y.map(mapping)

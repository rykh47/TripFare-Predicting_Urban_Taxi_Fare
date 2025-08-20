from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline


def remove_outliers_iqr(data: pd.DataFrame, columns: List[str], iqr_multiplier: float = 1.5) -> pd.DataFrame:
    cleaned = data.copy()
    mask = pd.Series(True, index=cleaned.index)
    for col in columns:
        if col in cleaned.columns:
            col_series = cleaned[col].astype(float)
            q1 = col_series.quantile(0.25)
            q3 = col_series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask &= col_series.between(lower, upper) | col_series.isna()
    return cleaned[mask]


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    """Build a ColumnTransformer dynamically based on available columns."""
    numeric_candidates = [
        "passenger_count",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "improvement_surcharge",
        "fare_amount",
        "trip_distance_km",
        "trip_duration_min",
        "pickup_hour",
        "pickup_dayofweek",
        "VendorID",
        "RatecodeID",
    ]

    categorical_candidates = [
        "store_and_fwd_flag",
        "payment_type",
        "pickup_am_pm",
        "pickup_is_weekend",
        "is_night",
    ]

    numeric_features = [c for c in numeric_candidates if c in df.columns]
    categorical_features = [c for c in categorical_candidates if c in df.columns]

    numeric_pipeline = [
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scaler", StandardScaler(with_mean=False)),
    ]

    categorical_pipeline = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", PipelineCompat(steps=numeric_pipeline), numeric_features),
            ("cat", PipelineCompat(steps=categorical_pipeline), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_features, categorical_features


class PipelineCompat:
    """Lightweight sklearn-compatible wrapper to avoid direct Pipeline import here.
    We implement only the minimal API used by ColumnTransformer (fit/transform).
    """

    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        current = X
        self.named_steps_ = {}
        for name, step in self.steps:
            step.fit(current, y)
            self.named_steps_[name] = step
            current = step.transform(current)
        self._last_output_shape_ = current.shape if hasattr(current, "shape") else None
        return self

    def transform(self, X):
        current = X
        for _, step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_params(self, deep=True):
        # Return steps so clone() works
        return {"steps": self.steps}

    def set_params(self, **params):
        # Allow setting steps
        for key, value in params.items():
            setattr(self, key, value)
        return self
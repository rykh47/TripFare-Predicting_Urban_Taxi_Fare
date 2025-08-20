import math
from typing import Optional
import numpy as np
import pandas as pd
import pytz


def haversine_distance_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Compute the great-circle distance between two points on Earth in kilometers.
    Inputs are in decimal degrees.
    """
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371.0088
    return pd.Series(earth_radius_km * c, index=lat1.index)


def ensure_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def convert_utc_to_eastern(dt_series: pd.Series) -> pd.Series:
    """Convert UTC timestamps to America/New_York timezone; returns timezone-aware datetime."""
    eastern = pytz.timezone("America/New_York")
    # Ensure UTC timezone-aware
    dt_series_utc = ensure_datetime(dt_series)
    return dt_series_utc.dt.tz_convert(eastern)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready engineered features from raw taxi trip dataframe.

    Required columns when available:
    - pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
    - tpep_pickup_datetime, tpep_dropoff_datetime
    - passenger_count, RatecodeID, payment_type, store_and_fwd_flag
    """
    data = df.copy()

    # Time features
    if "tpep_pickup_datetime" in data.columns:
        pickup_dt_eastern = convert_utc_to_eastern(data["tpep_pickup_datetime"])  # UTC -> NY
        data["pickup_hour"] = pickup_dt_eastern.dt.hour
        data["pickup_dayofweek"] = pickup_dt_eastern.dt.dayofweek  # Monday=0
        data["pickup_is_weekend"] = data["pickup_dayofweek"].isin([5, 6]).astype(int)
        data["pickup_am_pm"] = (pickup_dt_eastern.dt.hour >= 12).map({True: "PM", False: "AM"})
        data["is_night"] = pickup_dt_eastern.dt.hour.isin([22, 23, 0, 1, 2, 3, 4]).astype(int)
    else:
        data["pickup_hour"] = None
        data["pickup_dayofweek"] = None
        data["pickup_is_weekend"] = None
        data["pickup_am_pm"] = None
        data["is_night"] = None

    # Trip duration in minutes
    if "tpep_pickup_datetime" in data.columns and "tpep_dropoff_datetime" in data.columns:
        pickup_dt = ensure_datetime(data["tpep_pickup_datetime"])  # UTC
        dropoff_dt = ensure_datetime(data["tpep_dropoff_datetime"])  # UTC
        data["trip_duration_min"] = (dropoff_dt - pickup_dt).dt.total_seconds() / 60.0
    else:
        data["trip_duration_min"] = None

    # Haversine distance
    coord_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]
    if all(col in data.columns for col in coord_cols):
        data["trip_distance_km"] = haversine_distance_km(
            data["pickup_latitude"],
            data["pickup_longitude"],
            data["dropoff_latitude"],
            data["dropoff_longitude"],
        )
    else:
        data["trip_distance_km"] = None

    return data 
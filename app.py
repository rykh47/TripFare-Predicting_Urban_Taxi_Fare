import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

from src.features import engineer_features


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"

st.set_page_config(page_title="TripFare - Taxi Fare Prediction", page_icon="🚕", layout="centered")

st.title("🚕 TripFare: Predict Taxi Fare")

st.sidebar.header("Trip Details")

# Inputs
# Date picker
pickup_date = st.sidebar.date_input("Pickup Date")

# Time picker
pickup_time = st.sidebar.time_input("Pickup Time")

# Combine into one datetime object
pickup_datetime = datetime.combine(pickup_date, pickup_time)

st.sidebar.write("Selected pickup datetime:", pickup_datetime)

# Date picker
dropoff_date = st.sidebar.date_input("dropoff Date")

# Time picker
dropoff_time = st.sidebar.time_input("dropoff Time")

# Combine into one datetime object
dropoff_datetime = datetime.combine(dropoff_date, dropoff_time)

st.sidebar.write("Selected pickup datetime:", pickup_datetime)

col1, col2 = st.columns(2)
with col1:
    pickup_latitude = st.number_input("Pickup latitude", value=40.7614327, format="%.6f")
    dropoff_latitude = st.number_input("Dropoff latitude", value=40.6513111, format="%.6f")
    passenger_count = st.number_input("Passenger count", min_value=1, max_value=6, value=1)
with col2:
    pickup_longitude = st.number_input("Pickup longitude", value=-73.9798156, format="%.6f")
    dropoff_longitude = st.number_input("Dropoff longitude", value=-73.8803331, format="%.6f")
    ratecode_id = st.selectbox("RatecodeID", options=[1, 2, 3, 4, 5, 6], index=0)

payment_type = st.selectbox("Payment type", options=[0, 1, 2, 3, 4, 5], index=1)
store_and_fwd_flag = st.selectbox("Store and forward flag", options=["N", "Y"], index=0)
extra = st.number_input("Extra", value=0.5, step=0.5)
mta_tax = st.number_input("MTA tax", value=0.5, step=0.5)
tip_amount = st.number_input("Tip amount", value=0.0, step=0.5)
tolls_amount = st.number_input("Tolls amount", value=0.0, step=0.5)
improvement_surcharge = st.number_input("Improvement surcharge", value=0.3, step=0.1)
fare_amount = st.number_input("Base fare amount", value=2.5, step=0.5)

predict_btn = st.button("Predict Fare")

if not MODEL_PATH.exists():
    st.warning("Trained model not found. Please run training first: `python train.py`.")
else:
    model = joblib.load("C:\Users\eldho\Documents\guvi\repos\taxi_fare\artifacts\best_model.pkl")

    if predict_btn:
        # Build a one-row DataFrame matching expected raw schema
        raw = pd.DataFrame(
            [
                {
                    "tpep_pickup_datetime": pd.to_datetime(pickup_datetime, utc=True),
                    "tpep_dropoff_datetime": pd.to_datetime(dropoff_datetime, utc=True),
                    "pickup_latitude": pickup_latitude,
                    "pickup_longitude": pickup_longitude,
                    "dropoff_latitude": dropoff_latitude,
                    "dropoff_longitude": dropoff_longitude,
                    "passenger_count": passenger_count,
                    "RatecodeID": ratecode_id,
                    "payment_type": payment_type,
                    "store_and_fwd_flag": store_and_fwd_flag,
                    "extra": extra,
                    "mta_tax": mta_tax,
                    "tip_amount": tip_amount,
                    "tolls_amount": tolls_amount,
                    "improvement_surcharge": improvement_surcharge,
                    "fare_amount": fare_amount,
                }
            ]
        )

        features = engineer_features(raw)
        # The model pipeline includes preprocessing, so just predict
        pred = model.predict(features)[0]
        st.success(f"Estimated Total Fare: ${pred:.2f}")

st.markdown("---")
st.caption("Model must be trained and saved to artifacts/best_model.pkl before use.") 
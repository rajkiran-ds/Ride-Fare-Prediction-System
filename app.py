# ============================================================
# RIDESENSE AI — STREAMLIT APP
# Run: streamlit run app.py
# ============================================================

import os
import pickle
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RideSense AI",
    page_icon="🚖",
    layout="wide"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    model = pickle.load(
        open("artifacts/best_model.pkl", "rb")
    )

    columns = pickle.load(
        open("artifacts/model_columns.pkl", "rb")
    )

    return model, columns


# ============================================================
# TITLE
# ============================================================

st.title("🚖 RideSense AI — Ride Fare Prediction")

st.write(
    "Enter trip details below to predict the ride fare."
)

st.markdown("---")


# ============================================================
# CHECK MODEL EXISTS
# ============================================================

if not os.path.exists("artifacts/best_model.pkl"):

    st.warning(
        "Model not found. "
        "Please run `python main.py` first."
    )

    st.stop()


model, columns = load_artifacts()


# ============================================================
# USER INPUTS
# ============================================================

c1, c2, c3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with c1:

    dist = st.number_input(
        "Trip Distance (km)",
        min_value=0.5,
        max_value=100.0,
        value=12.0
    )

    duration = st.number_input(
        "Trip Duration (min)",
        min_value=1.0,
        max_value=180.0,
        value=30.0
    )

    surge = st.number_input(
        "Surge Multiplier",
        min_value=1.0,
        max_value=5.0,
        value=1.2
    )

    fuel = st.number_input(
        "Fuel Price per Liter",
        min_value=1.0,
        max_value=10.0,
        value=1.5
    )

    demand = st.number_input(
        "Demand Index",
        min_value=0.0,
        max_value=10.0,
        value=5.0
    )


# ============================================================
# COLUMN 2
# ============================================================

with c2:

    driver_r = st.slider(
        "Driver Rating",
        1.0,
        5.0,
        4.2
    )

    customer_r = st.slider(
        "Customer Rating",
        1.0,
        5.0,
        4.0
    )

    loyalty = st.slider(
        "Customer Loyalty Score",
        0,
        100,
        60
    )

    battery = st.slider(
        "Phone Battery (%)",
        0,
        100,
        75
    )


# ============================================================
# COLUMN 3
# ============================================================

with c3:

    traffic = st.selectbox(
        "Traffic Level",
        ["Rural", "Low", "Medium"]
    )

    vehicle = st.selectbox(
        "Vehicle Type",
        ["Hatchback", "Premium", "SUV", "Sedan"]
    )

    weather = st.selectbox(
        "Weather Condition",
        ["Clear", "Rainy", "Sunny"]
    )

    peak = st.selectbox(
        "Peak Hour",
        ["Yes", "No"]
    )

    zone = st.selectbox(
        "Pickup Zone",
        ["Rural", "SemiUrban", "Urban"]
    )

    music = st.selectbox(
        "Music Preference",
        ["Jazz", "Pop", "Rock"]
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🚖 Predict Fare",
    use_container_width=True
):

    # --------------------------------------------------------
    # CREATE INPUT ROW
    # --------------------------------------------------------

    row = {

        # NUMERIC FEATURES

        "trip_distance_km": dist,

        "trip_duration_min": duration,

        "surge_multiplier": surge,

        "fuel_price_per_liter": fuel,

        "demand_index": demand,

        "driver_rating": driver_r,

        "customer_rating": customer_r,

        "customer_loyalty_score": loyalty,

        "phone_battery_level": battery,

        "peak_hour": 1 if peak == "Yes" else 0,


        # TRAFFIC LEVEL

        "traffic_level_Low":
            1 if traffic == "Low" else 0,

        "traffic_level_Medium":
            1 if traffic == "Medium" else 0,


        # VEHICLE TYPE

        "vehicle_type_Premium":
            1 if vehicle == "Premium" else 0,

        "vehicle_type_SUV":
            1 if vehicle == "SUV" else 0,

        "vehicle_type_Sedan":
            1 if vehicle == "Sedan" else 0,


        # WEATHER

        "weather_condition_Rainy":
            1 if weather == "Rainy" else 0,

        "weather_condition_Sunny":
            1 if weather == "Sunny" else 0,


        # PICKUP ZONE

        "pickup_zone_SemiUrban":
            1 if zone == "SemiUrban" else 0,

        "pickup_zone_Urban":
            1 if zone == "Urban" else 0,


        # MUSIC PREFERENCE

        "music_preference_Pop":
            1 if music == "Pop" else 0,

        "music_preference_Rock":
            1 if music == "Rock" else 0,
    }


    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    X_input = pd.DataFrame([row])


    # --------------------------------------------------------
    # MATCH TRAINING COLUMNS
    # --------------------------------------------------------

    X_input = X_input.reindex(
        columns=columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    prediction = model.predict(X_input)[0]


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    st.markdown("---")

    st.success(
        f"### 🚖 Predicted Ride Fare: ₹ {prediction:,.2f}"
    )

    st.caption(
        "Model: RandomForestRegressor | "
        "Strategy: Winsorized Data"
    )
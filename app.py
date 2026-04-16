import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="centered"
)

# ---- LOAD MODEL & FEATURES ----
@st.cache_resource
def load_model():
    with open('models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_feature_columns():
    with open('models/feature_columns.json', 'r') as f:
        columns = json.load(f)
    return columns

model = load_model()
feature_columns = load_feature_columns()

# ---- HEADER ----
st.title("🏎️ F1 Race Outcome Predictor")
st.markdown("Predict whether a driver will finish in the **Top 3** based on race features.")
st.divider()

# ---- INPUT FORM ----
st.subheader("Enter Driver & Race Details")

col1, col2 = st.columns(2)

with col1:
    grid_position = st.number_input(
        "Grid Position (Starting Position)",
        min_value=1, max_value=20, value=1, step=1
    )
    team_id = st.number_input(
        "Team ID (0-9)",
        min_value=0, max_value=9, value=0, step=1
    )
    is_top_team = st.selectbox(
        "Is Top Team?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    driver_top3_sofar = st.number_input(
        "Driver Top 3 Finishes So Far This Season",
        min_value=0, max_value=22, value=0, step=1
    )
    driver_finish_rate = st.slider(
        "Driver Finish Rate (0.0 - 1.0)",
        min_value=0.0, max_value=1.0, value=0.8, step=0.01
    )

with col2:
    grid_vs_avg = st.number_input(
        "Grid vs Average (Grid - Season Avg)",
        min_value=-20.0, max_value=20.0, value=0.0, step=0.1
    )
    driver_avg_pos_last3 = st.number_input(
        "Driver Avg Position (Last 3 Races)",
        min_value=1.0, max_value=20.0, value=5.0, step=0.1
    )
    team_top3_sofar = st.number_input(
        "Team Top 3 Finishes So Far",
        min_value=0, max_value=44, value=0, step=1
    )
    team_avg_pos = st.number_input(
        "Team Average Position",
        min_value=1.0, max_value=20.0, value=5.0, step=0.1
    )

# ---- CALCULATE DERIVED FEATURES ----
grid_position_squared = grid_position ** 2

# ---- BUILD INPUT DATAFRAME ----
input_data = {
    'GridPosition': grid_position,
    'GridPosition_Squared': grid_position_squared,
    'Grid_vs_Avg': grid_vs_avg,
    'TeamID': team_id,
    'IsTopTeam': is_top_team,
    'Driver_Top3_SoFar': driver_top3_sofar,
    'Driver_AvgPos_Last3': driver_avg_pos_last3,
    'Driver_FinishRate': driver_finish_rate,
    'Team_Top3_SoFar': team_top3_sofar,
    'Team_AvgPos': team_avg_pos
}

input_df = pd.DataFrame([input_data])[feature_columns]

# ---- PREDICT BUTTON ----
st.divider()
if st.button("🏁 Predict", use_container_width=True):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.divider()
    if prediction == 1:
        st.success("🏆 This driver is predicted to finish in the **TOP 3!**")
    else:
        st.error("❌ This driver is predicted to **NOT** finish in the top 3.")

    st.subheader("Prediction Confidence")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Top 3 Probability", f"{probability[1]*100:.1f}%")
    with col2:
        st.metric("Not Top 3 Probability", f"{probability[0]*100:.1f}%")

    st.progress(float(probability[1]))

# ---- FOOTER ----
st.divider()
st.caption("F1 Race Predictor — ML Project | Built with Streamlit & scikit-learn")
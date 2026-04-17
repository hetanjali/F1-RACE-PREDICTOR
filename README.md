# 🏎️ F1 Race Outcome Predictor

A machine learning system that predicts whether a Formula 1 driver will finish in the **Top 3** using real race data from the 2023 F1 World Championship.


---

## Overview

Formula 1 race outcomes depend on a complex mix of driver performance, team strength, and starting position. This project builds a complete ML pipeline — from raw data ingestion to a deployed interactive web application — that predicts podium finishes with measurable accuracy.

Data is fetched directly from the official F1 timing system using the FastF1 API, processed through a feature engineering pipeline, and used to train and compare multiple classification models.

---

## Live App

The deployed Streamlit app allows users to input race parameters and receive an instant Top 3 prediction along with a confidence probability score.

🌐 **Live Demo:** [Click here to try the app](https://hetanjali-f1-race-predictor-app-xe74vw.streamlit.app/)
---

## Technical Stack

| Layer | Tools |
|-------|-------|
| Data Ingestion | FastF1, requests-cache |
| Data Processing | pandas, numpy |
| Feature Engineering | scikit-learn, custom pipeline |
| Modelling | Random Forest, XGBoost |
| Visualisation | matplotlib, seaborn |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

---

## Dataset

- **Source:** FastF1 — official F1 timing and telemetry API
- **Season:** 2023 Formula 1 World Championship
- **Coverage:** 22 races × 20 drivers = 440 records
- **Target:** Binary classification — Top 3 finish (1) or not (0)

---

## Features Used

| Feature | Description |
|---------|-------------|
| `GridPosition` | Starting position on the grid |
| `GridPosition_Squared` | Non-linear grid penalty |
| `Grid_vs_Avg` | Grid position relative to season average |
| `TeamID` | Encoded team identifier |
| `IsTopTeam` | Whether the team is a front-runner |
| `Driver_Top3_SoFar` | Driver's podium count in current season |
| `Driver_AvgPos_Last3` | Average finish over last 3 races |
| `Driver_FinishRate` | Race completion rate |
| `Team_Top3_SoFar` | Team's podium count in current season |
| `Team_AvgPos` | Team's average finishing position |

---

## Models

Two classification models were trained and evaluated:

**Random Forest** — Selected as the final model based on overall performance. Ensemble of decision trees with low variance and strong generalisation.

**XGBoost** — Gradient boosted trees used for comparison. Competitive performance with faster training on larger datasets.

Both models output a predicted class (0 or 1) and a probability score used to display prediction confidence in the app.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_fetch_data` | Pull race session results from FastF1 API and save to CSV |
| `02_eda` | Distribution analysis, correlation heatmaps, podium patterns |
| `03_clean_data` | Handle missing values, encode categoricals, validate schema |
| `04_feature_engineering` | Construct derived features, rolling statistics, team flags |
| `05_train_model` | Train, evaluate, compare and serialise models |
| `06_predict` | Load best model and predict podium finishes for any race scenario |
---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/F1-RACE-PREDICTOR.git
cd F1-RACE-PREDICTOR

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Results

- Best Model: **Random Forest**
- Output: Top 3 prediction + confidence probability
- Interface: Interactive web app with real-time inference

---

## Developers


 **Hetanjali Vaghela** 
 
 **Fiza Saiyed** 
___

*2023 F1 data sourced via [FastF1](https://theoehrly.github.io/Fast-F1/)*


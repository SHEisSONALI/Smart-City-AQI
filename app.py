import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("C:/Users/User/Documents/python/2ridge_model.joblib")
scaler = joblib.load("C:/Users/User/Documents/python/2scaler.joblib")

print("Scaler expects:", scaler.n_features_in_)

# Define features (must match training order)
features = [
    'smart_mobility_Road_Occupancy_%',
    'smart_mobility_Emission_Levels_g_km',
    'smart_mobility_Sentiment_Score',
    'smart_mobility_Energy_Consumption_L_h',
    'AQI_roll7',
    'AQI_roll30',
    'AQI_lag1',
    'AQI_lag3',
    'AQI_lag7',
    'month',
    'dayofweek'
]

default_values = {
    'smart_mobility_Road_Occupancy_%': 20.0,
    'smart_mobility_Emission_Levels_g_km': 5.0,
    'smart_mobility_Sentiment_Score': 0.9,
    'smart_mobility_Energy_Consumption_L_h': 80.0,
    'AQI_roll7': 25.0,
    'AQI_roll30': 30.0,
    'AQI_lag1': 28.0,
    'AQI_lag3': 32.0,
    'AQI_lag7': 35.0,
    'month': 7,       # July
    'dayofweek': 2
}

st.title("Air Quality Prediction App")

st.write("Enter feature values below to predict the Air Quality Index (AQI):")

# Collect user inputs
user_inputs = []
for feature in features:
    if feature == "month":
        val = st.number_input(f"{feature}", min_value=1, max_value=12, step=1, value=default_values['month'])  # July as default
    elif feature == "dayofweek":
        val = st.number_input(f"{feature}", min_value=0, max_value=6, step=1, value=default_values['dayofweek'])   # Tuesday as default (0=Monday)
    else:
        val = st.number_input(f"{feature}", value=default_values.get(feature, 0.0))
    user_inputs.append(val)

if st.button("Predict AQI"):
    user_array = np.array(user_inputs).reshape(1, -1)
    # Scale inputs
    user_scaled = scaler.transform(user_array)
    prediction = model.predict(user_scaled)
    predicted_aqi = prediction[0]

    # Determine AQI category
    if predicted_aqi <= 50:
        category = "Good"
    elif predicted_aqi <= 100:
        category = "Moderate"
    elif predicted_aqi <= 150:
        category = "Unhealthy for sensitive groups"
    elif predicted_aqi <= 200:
        category = "Unhealthy"
    elif predicted_aqi <= 300:
        category = "Very unhealthy"
    else:
        category = "Hazardous"
    st.success(f"Predicted Air Quality Index: {prediction[0]:.2f} ({category})")

# Show model details
st.subheader("Model Details")
coef_df = pd.DataFrame({
    "Feature": features[:len(model.coef_)],
    "Coefficient": model.coef_
})
st.table(coef_df)
st.write("Intercept:", model.intercept_)

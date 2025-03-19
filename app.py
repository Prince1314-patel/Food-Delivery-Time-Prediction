import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background-color: #F5F5F5;
    }
    .header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        color: #FFFFFF;  /* Changed to white as requested */
    }
    .stButton>button {
        background-color: #ff5733;  /* Changed to orange */
        color: white;
        font-size: 1rem;
        font-weight: bold;
        padding: 0.5rem 1rem;
        display: block;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the prediction model
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    try:
        cat_pipeline = model.named_steps['preprocessor'].transformers[1][1]
        cat_pipeline.named_steps['onehot'].handle_unknown = 'ignore'
    except Exception as e:
        st.error(f"Error updating one-hot encoder: {e}")
    return model

model = load_model()

# App Title and Description
st.markdown('<div class="header">Food Delivery Time Prediction</div>', unsafe_allow_html=True)
st.write("This app predicts delivery time in minutes based on your inputs. Please provide the required details below.")

# Mapping dictionaries for selectbox options with icons
weather_options = {
    "Clear ☀️": "Clear",
    "Rainy 🌧️": "Rainy",
    "Snowy ❄️": "Snowy",
    "Foggy 🌫️": "Foggy",
    "Windy 💨": "Windy"
}

vehicle_type_options = {
    "Bike 🚲": "Bike",
    "Scooter 🛵": "Scooter",
    "Car 🚗": "Car"
}

# Input Sections in Main Body
### Delivery Conditions
st.subheader("Delivery Conditions")
col1, col2, col3 = st.columns(3)
with col1:
    weather_display = st.selectbox(
        "Weather",
        options=list(weather_options.keys()),
        key="weather_main",
        help="Select the current weather condition."
    )
with col2:
    traffic_level = st.selectbox(
        "Traffic Level",
        options=["Low", "Medium", "High"],
        key="traffic_main",
        help="Select the traffic level."
    )
with col3:
    time_of_day = st.selectbox(
        "Time of Day",
        options=["Morning", "Afternoon", "Evening", "Night"],
        key="time_main",
        help="Select the time of day."
    )

### Courier Details
st.subheader("Courier Details")
col4, col5 = st.columns(2)
with col4:
    vehicle_type_display = st.selectbox(
        "Vehicle Type",
        options=list(vehicle_type_options.keys()),
        key="vehicle_main",
        help="Select the vehicle type used for delivery."
    )
with col5:
    courier_experience_yrs = st.number_input(
        "Courier Experience (years)",
        min_value=0.0,
        value=2.0,
        step=0.1,
        key="experience_main",
        help="Enter the courier's experience in years."
    )

### Order Details
st.subheader("Order Details")
col6, col7 = st.columns(2)
with col6:
    distance_km = st.number_input(
        "Distance (km)",
        min_value=0.0,
        value=5.0,
        step=0.1,
        key="distance_main",
        help="Enter the delivery distance in kilometers."
    )
with col7:
    preparation_time_min = st.number_input(
        "Preparation Time (min)",
        min_value=0,
        value=15,
        step=1,
        key="prep_time_main",
        help="Enter the time taken to prepare the order in minutes."
    )

# Summary of Inputs
st.subheader("Summary of Inputs")
st.write(f"- **Distance**: {distance_km} km")
st.write(f"- **Weather**: {weather_options[weather_display]}")
st.write(f"- **Traffic Level**: {traffic_level}")
st.write(f"- **Time of Day**: {time_of_day}")
st.write(f"- **Vehicle Type**: {vehicle_type_options[vehicle_type_display]}")
st.write(f"- **Preparation Time**: {preparation_time_min} min")
st.write(f"- **Courier Experience**: {courier_experience_yrs} years")

# Prediction Button and Output
if st.button("Predict Delivery Time"):
    # Input validation
    if distance_km <= 0:
        st.error("Distance must be greater than zero.")
    elif preparation_time_min < 0:
        st.error("Preparation time cannot be negative.")
    else:
        # Map display values to actual values
        weather_actual = weather_options[weather_display]
        vehicle_type_actual = vehicle_type_options[vehicle_type_display]

        # Assemble input data
        input_data = pd.DataFrame({
            'Distance_km': [distance_km],
            'Weather': [weather_actual],
            'Traffic_Level': [traffic_level],
            'Time_of_Day': [time_of_day],
            'Vehicle_Type': [vehicle_type_actual],
            'Preparation_Time_min': [preparation_time_min],
            'Courier_Experience_yrs': [courier_experience_yrs]
        })

        try:
            prediction = model.predict(input_data)
            predicted_time = np.round(prediction[0], 1)
            st.markdown(
                f"<h2 style='text-align: center; color: green;'>Predicted Delivery Time: {predicted_time} minutes</h2>",
                unsafe_allow_html=True
            )
            st.write("*Note: This prediction is based on historical data and may vary depending on actual conditions.*")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
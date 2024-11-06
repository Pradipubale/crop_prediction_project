import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
dataset = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Crop_recommendation.csv")

# Replace categorical values with numerical codes
dataset["label"].replace({'rice':'1', 'maize':'2', 'chickpea':'3', 'kidneybeans':'4', 
                          'pigeonpeas':'5', 'mothbeans':'6', 'mungbean':'7', 'blackgram':'8', 
                          'lentil':'9', 'watermelon':'10', 'muskmelon':'11', 'cotton':'12', 'jute':'13'}, inplace=True)

dataset["season"].replace({'rainy':'1', 'winter':'2', 'spring':'3', 'summer':'4'}, inplace=True)

# Split the data for training and testing
x_train, x_test, y_train, y_test = train_test_split(
    dataset[['temperature', 'humidity', 'ph', 'water availability', 'season']],
    dataset['label'], 
    test_size=0.3
)

# Train a Logistic Regression model
lr = LogisticRegression(max_iter=500)
lr.fit(x_train, y_train)

# Streamlit App
st.title("Crop Prediction App")
st.write("Predict the type of crop based on input features like temperature, humidity, pH, water availability, and season.")

# Input sliders for user data
temperature = st.slider("Temperature (°C)", min_value=10, max_value=50, value=25)
humidity = st.slider("Humidity (%)", min_value=10, max_value=100, value=50)
ph = st.slider("pH Value", min_value=3.0, max_value=9.0, step=0.1, value=6.5)
water_availability = st.slider("Water Availability", min_value=10, max_value=300, value=100)
season = st.selectbox("Season", options=["Rainy", "Winter", "Spring", "Summer"])

# Convert season to numerical code
season_code = {"Rainy": 1, "Winter": 2, "Spring": 3, "Summer": 4}[season]

# Predict using the trained model
input_data = pd.DataFrame([[temperature, humidity, ph, water_availability, season_code]], 
                          columns=['temperature', 'humidity', 'ph', 'water availability', 'season'])
prediction = lr.predict(input_data)

# Map prediction back to crop name
crop_map = {1: 'Rice', 2: 'Maize', 3: 'Chickpea', 4: 'Kidneybeans', 5: 'Pigeonpeas',
            6: 'Mothbeans', 7: 'Mungbean', 8: 'Blackgram', 9: 'Lentil', 10: 'Watermelon',
            11: 'Muskmelon', 12: 'Cotton', 13: 'Jute'}

st.subheader("Predicted Crop")
st.write(crop_map[int(prediction[0])])

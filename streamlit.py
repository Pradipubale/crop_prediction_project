import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the dataset
dataset = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\Crop_recommendation.csv")

# Preprocess data
dataset["label"].replace({'rice': '1', 'maize': '2', 'chickpea': '3', 'kidneybeans': '4', 'pigeonpeas': '5',
                          'mothbeans': '6', 'mungbean': '7', 'blackgram': '8', 'lentil': '9', 'watermelon': '10',
                          'muskmelon': '11', 'cotton': '12', 'jute': '13'}, inplace=True)
dataset["season"].replace({'rainy': '1', 'winter': '2', 'spring': '3', 'summer': '4'}, inplace=True)

# Split data for training
X = dataset[['temperature', 'humidity', 'ph', 'water availability', 'season']]
y = dataset['label'].astype(int)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize and train model
lr = LogisticRegression(max_iter=200)
lr.fit(x_train, y_train)

# Streamlit app
st.title("Crop Recommendation System")

st.write("Enter the following parameters to get the recommended crop:")

# User input fields
temperature = st.number_input("Temperature", min_value=0.0, max_value=50.0, step=0.1)
humidity = st.number_input("Humidity", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.01)
water_availability = st.number_input("Water Availability", min_value=0.0, step=0.1)
season = st.selectbox("Season", options={"Rainy": 1, "Winter": 2, "Spring": 3, "Summer": 4})

# Prediction button
if st.button("Predict Crop"):
    input_data = pd.DataFrame([[temperature, humidity, ph, water_availability, season]],
                              columns=['temperature', 'humidity', 'ph', 'water availability', 'season'])
    prediction = lr.predict(input_data)
    
    # Convert label to crop name
    crop_dict = {1: 'Rice', 2: 'Maize', 3: 'Chickpea', 4: 'Kidneybeans', 5: 'Pigeonpeas', 6: 'Mothbeans', 
                 7: 'Mungbean', 8: 'Blackgram', 9: 'Lentil', 10: 'Watermelon', 11: 'Muskmelon', 
                 12: 'Cotton', 13: 'Jute'}
    crop = crop_dict.get(int(prediction[0]), "Unknown Crop")
    
    st.write(f"Recommended Crop: **{crop}**")

# Visualizations
st.write("### Pairplot of Crop Data")
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.pairplot(dataset[['temperature', 'humidity', 'ph', 'water availability', 'label']], hue='label')
st.pyplot()

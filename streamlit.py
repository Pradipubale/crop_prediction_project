import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
pd.read_csv("Crop_recommendation.csv")  # Capital C

# Map categorical values to numerical codes
label_mapping = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 
    'watermelon': 10, 'muskmelon': 11, 'cotton': 12, 'jute': 13
}
season_mapping = {'rainy': 1, 'winter': 2, 'spring': 3, 'summer': 4}

dataset['label'] = dataset['label'].map(label_mapping)
dataset['season'] = dataset['season'].map(season_mapping)

# Split the data into features and target variable
X = dataset[['temperature', 'humidity', 'ph', 'water availability', 'season']]
y = dataset['label']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# Function to predict crop
def predict_crop(temperature, humidity, ph, water_availability, season):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[temperature, humidity, ph, water_availability, season]], 
                              columns=['temperature', 'humidity', 'ph', 'water availability', 'season'])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Map numerical prediction back to crop name
    crop_mapping = {v: k for k, v in label_mapping.items()}
    predicted_crop = crop_mapping[prediction[0]]
    
    return predicted_crop

# Streamlit App
st.title("Crop Prediction System")

# Input fields
temperature = st.number_input("Enter Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
humidity = st.number_input("Enter Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input("Enter pH Value", min_value=0.0, max_value=14.0, value=6.5)
water_availability = st.number_input("Enter Water Availability (in mm)", min_value=0.0, max_value=500.0, value=100.0)
season = st.selectbox("Select Season", ["rainy", "winter", "spring", "summer"])

# Convert season to numerical code
season_code = season_mapping[season]

# Prediction button
if st.button("Predict Crop"):
    result = predict_crop(temperature, humidity, ph, water_availability, season_code)
    st.success(f"The predicted crop is: {result}")

# Display model accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")

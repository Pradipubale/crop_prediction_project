import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv("Crop_recommendation.csv")

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
    input_data = pd.DataFrame([[temperature, humidity, ph, water_availability, season]], 
                              columns=['temperature', 'humidity', 'ph', 'water availability', 'season'])
    prediction = model.predict(input_data)
    crop_mapping = {v: k for k, v in label_mapping.items()}
    predicted_crop = crop_mapping[prediction[0]]
    return predicted_crop

# --- Streamlit App UI ---

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        color: #2C3E50;
        text-align: center;
    }
    .prediction {
        font-size: 24px;
        font-weight: bold;
        color: #27AE60;
        text-align: center;
    }
    .accuracy {
        font-size: 18px;
        color: #2980B9;
        text-align: center;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title'>🌱 Smart Crop Prediction System</h1>", unsafe_allow_html=True)
st.markdown("---")

# Input Fields in two columns
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=20.0)
        ph = st.number_input("🧪 pH Value", min_value=0.0, max_value=14.0, value=6.5)
        season = st.selectbox("🗓️ Season", ["rainy", "winter", "spring", "summer"])

    with col2:
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
        water_availability = st.number_input("🚿 Water Availability (mm)", min_value=0.0, max_value=500.0, value=100.0)

# Map season to numerical code
season_code = season_mapping[season]

# Prediction Button
if st.button("🔍 Predict Crop"):
    result = predict_crop(temperature, humidity, ph, water_availability, season_code)
    st.markdown(f"<div class='prediction'>✅ Predicted Crop: {result.title()}</div>", unsafe_allow_html=True)

# Model Accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
st.markdown(f"<div class='accuracy'>🔍 Model Accuracy: {accuracy:.2%}</div>", unsafe_allow_html=True)

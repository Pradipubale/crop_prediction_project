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

# Split data into features and target
X = dataset[['temperature', 'humidity', 'ph', 'water availability', 'season']]
y = dataset['label']

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# Prediction function
def predict_crop(temperature, humidity, ph, water_availability, season):
    input_data = pd.DataFrame([[temperature, humidity, ph, water_availability, season]], 
                              columns=['temperature', 'humidity', 'ph', 'water availability', 'season'])
    prediction = model.predict(input_data)
    crop_mapping = {v: k for k, v in label_mapping.items()}
    return crop_mapping[prediction[0]]

# --- Streamlit App UI ---

# Custom CSS for a clean dashboard
st.markdown("""
    <style>
        .main-container {
            background-color: #f5f7fa;
            padding: 25px;
            border-radius: 10px;
        }
        .title {
            font-size: 42px;
            color: #2c3e50;
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 18px;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 30px;
        }
        .result {
            font-size: 28px;
            color: #27ae60;
            text-align: center;
            margin-top: 30px;
            font-weight: bold;
        }
        .accuracy {
            text-align: center;
            font-size: 18px;
            color: #2980b9;
            margin-top: 20px;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #95a5a6;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🌿 About")
st.sidebar.info("""
A smart AI-powered Crop Prediction system that recommends the best crop based on environmental parameters.

Developed with 💚 by Prathamesh.
""")
st.sidebar.markdown("---")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Dataset:** 13 Crops, 4 Seasons")

# Title and Subtitle
st.markdown("<h1 class='title'>🌱 Smart Crop Recommendation</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter environmental conditions and we'll recommend the ideal crop for you.</div>", unsafe_allow_html=True)

# Main input container
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider("🌡️ Temperature (°C)", 0.0, 50.0, 25.0)
        ph = st.slider("🧪 Soil pH", 0.0, 14.0, 6.5)
        season = st.selectbox("🗓️ Season", ["rainy", "winter", "spring", "summer"])

    with col2:
        humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 65.0)
        water_availability = st.slider("🚿 Water Availability (mm)", 0.0, 500.0, 120.0)

    season_code = season_mapping[season]

    predict = st.button("🔍 Predict Ideal Crop")

    if predict:
        result = predict_crop(temperature, humidity, ph, water_availability, season_code)
        st.markdown(f"<div class='result'>🌾 Recommended Crop: {result.title()}</div>", unsafe_allow_html=True)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.markdown(f"<div class='accuracy'>📊 Model Accuracy: {accuracy:.2%}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2025 Crop Recommendation AI. All rights reserved.</div>", unsafe_allow_html=True)

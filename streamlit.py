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

# Custom CSS for a modern, vibrant dashboard
st.markdown("""
    <style>
        body {
            background-color: #e8f5e9;
        }
        .main-container {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin: 20px;
        }
        .title {
            font-size: 48px;
            color: #1a3c34;
            text-align: center;
            font-weight: 700;
            margin-bottom: 10px;
            font-family: 'Arial', sans-serif;
        }
        .subtitle {
            font-size: 20px;
            color: #4a7043;
            text-align: center;
            margin-bottom: 30px;
            font-family: 'Arial', sans-serif;
        }
        .result {
            font-size: 32px;
            color: #2ecc71;
            text-align: center;
            margin-top: 30px;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
        }
        .accuracy {
            text-align: center;
            font-size: 20px;
            color: #3498db;
            margin-top: 20px;
            font-family: 'Arial', sans-serif;
        }
        .footer {
            text-align: center;
            font-size: 16px;
            color: #7f8c8d;
            margin-top: 50px;
            font-family: 'Arial', sans-serif;
        }
        .input-label {
            font-size: 18px;
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .stButton>button {
            background-color: #27ae60;
            color: white;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #219653;
        }
        .sidebar .sidebar-content {
            background-color: #f0f4f8;
            border-radius: 10px;
            padding: 20px;
        }
        .sidebar-title {
            font-size: 24px;
            color: #1a3c34;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .crop-image {
            display: block;
            margin: 20px auto;
            border-radius: 10px;
            max-width: 100%;
            height: auto;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<div class='sidebar-title'>🌿 About the App</div>", unsafe_allow_html=True)
st.sidebar.info("""
This AI-powered Crop Recommendation system uses machine learning to suggest the best crop based on environmental conditions like temperature, humidity, soil pH, water availability, and season.

**Developed with 💚 by Prathamesh**
""")
st.sidebar.markdown("---")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Dataset:** 13 Crops, 4 Seasons")
st.sidebar.image("https://via.placeholder.com/150x100.png?text=Crop+Field", caption="Agricultural Insight", use_column_width=True)

# Title and Subtitle
st.markdown("<h1 class='title'>🌾 Smart Crop Recommendation</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Input environmental conditions to discover the ideal crop for your farm.</div>", unsafe_allow_html=True)

# Main input container
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # Input form with improved layout
    st.markdown("### 📊 Enter Environmental Parameters")
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown("<div class='input-label'>🌡️ Temperature (°C)</div>", unsafe_allow_html=True)
        temperature = st.slider("", 0.0, 50.0, 25.0, key="temp")
        
        st.markdown("<div class='input-label'>🧪 Soil pH</div>", unsafe_allow_html=True)
        ph = st.slider("", 0.0, 14.0, 6.5, key="ph")
        
        st.markdown("<div class='input-label'>🗓️ Season</div>", unsafe_allow_html=True)
        season = st.selectbox("", ["rainy", "winter", "spring", "summer"], key="season")

    with col2:
        st.markdown("<div class='input-label'>💧 Humidity (%)</div>", unsafe_allow_html=True)
        humidity = st.slider("", 0.0, 100.0, 65.0, key="humidity")
        
        st.markdown("<div class='input-label'>🚿 Water Availability (mm)</div>", unsafe_allow_html=True)
        water_availability = st.slider("", 0.0, 500.0, 120.0, key="water")

    season_code = season_mapping[season]

    # Center the predict button
    st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    predict = st.button("🔍 Predict Ideal Crop")
    st.markdown("</div>", unsafe_allow_html=True)

    if predict:
        result = predict_crop(temperature, humidity, ph, water_availability, season_code)
        st.markdown(f"<div class='result'>🌱 Recommended Crop: {result.title()}</div>", unsafe_allow_html=True)
        # Display a placeholder crop image (replace with actual crop images if available)
        st.image(f"https://via.placeholder.com/300x200.png?text={result.title()}+Crop", caption=f"{result.title()} Field", use_column_width=True)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.markdown(f"<div class='accuracy'>📈 Model Accuracy: {accuracy:.2%}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2025 Crop Recommendation AI | Powered by xAI</div>", unsafe_allow_html=True)

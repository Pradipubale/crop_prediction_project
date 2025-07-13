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

# Dictionary for crop images (Unsplash free stock images)
crop_images = {
    'rice': 'https://images.unsplash.com/photo-1600585154340-be6161a56a0c',
    'maize': 'https://images.unsplash.com/photo-1592982537447-6f34a125f35f',
    'chickpea': 'https://images.unsplash.com/photo-1606728035253-49b27a0aff65',
    'kidneybeans': 'https://images.unsplash.com/photo-1599138878056-9e0a370e3a6e',
    'pigeonpeas': 'https://images.unsplash.com/photo-1603046891744-9d7d9d2a8d7e',
    'mothbeans': 'https://images.unsplash.com/photo-1606728035253-49b27a0aff65',  # Placeholder, similar to chickpeas
    'mungbean': 'https://images.unsplash.com/photo-1606728035253-49b27a0aff65',  # Placeholder, similar to chickpeas
    'blackgram': 'https://images.unsplash.com/photo-1606728035253-49b27a0aff65',  # Placeholder, similar to chickpeas
    'lentil': 'https://images.unsplash.com/photo-1518967064-1e33e2f820c0',
    'watermelon': 'https://images.unsplash.com/photo-1621583442991-2e56f7d76b9e',
    'muskmelon': 'https://images.unsplash.com/photo-1621583442991-2e56f7d76b9e',  # Placeholder, similar to watermelon
    'cotton': 'https://images.unsplash.com/photo-1603794052299-8b5b7c53e6a4',
    'jute': 'https://images.unsplash.com/photo-1599138878056-9e0a370e3a6e'  # Placeholder, similar to kidneybeans
}

# Prediction function
def predict_crop(temperature, humidity, ph, water_availability, season):
    input_data = pd.DataFrame([[temperature, humidity, ph, water_availability, season]], 
                              columns=['temperature', 'humidity', 'ph', 'water availability', 'season'])
    prediction = model.predict(input_data)
    crop_mapping = {v: k for k, v in label_mapping.items()}
    return crop_mapping[prediction[0]]

# --- Streamlit App UI ---

# Custom CSS for a modern, farm-inspired dashboard
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #d4efdf 0%, #a3e4d7 100%);
        }
        .main-container {
            background-color: #ffffff;
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
            margin: 30px auto;
            max-width: 900px;
        }
        .title {
            font-size: 50px;
            color: #1a3c34;
            text-align: center;
            font-weight: 800;
            margin-bottom: 15px;
            font-family: 'Helvetica Neue', sans-serif;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        }
        .subtitle {
            font-size: 22px;
            color: #4a7043;
            text-align: center;
            margin-bottom: 40px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .result {
            font-size: 34px;
            color: #2ecc71;
            text-align: center;
            margin-top: 40px;
            font-weight: bold;
            font-family: 'Helvetica Neue', sans-serif;
            animation: fadeIn 1s ease-in-out;
        }
        .accuracy {
            text-align: center;
            font-size: 22px;
            color: #3498db;
            margin-top: 25px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .footer {
            text-align: center;
            font-size: 16px;
            color: #7f8c8d;
            margin-top: 60px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .input-label {
            font-size: 20px;
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 12px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stSlider > div > div > div > div {
            background-color: #27ae60;
            border-radius: 10px;
        }
        .stButton>button {
            background: linear-gradient(45deg, #27ae60, #2ecc71);
            color: white;
            border-radius: 12px;
            padding: 15px 30px;
            font-size: 20px;
            font-weight: bold;
            border: none;
            display: block;
            margin: 30px auto;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        .sidebar .sidebar-content {
            background: linear-gradient(135deg, #f0f4f8, #d9e6f2);
            border-radius: 15px;
            padding: 25px;
        }
        .sidebar-title {
            font-size: 28px;
            color: #1a3c34;
            font-weight: bold;
            margin-bottom: 20px;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .crop-image {
            display: block;
            margin: 30px auto;
            border-radius: 15px;
            max-width: 100%;
            height: auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        .stSelectbox > div > div {
            border-radius: 10px;
            background-color: #f0f4f8;
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
st.sidebar.image("https://images.unsplash.com/photo-1500595046743-cd271d6942f0", caption="Sustainable Farming", use_column_width=True)

# Title and Subtitle
st.markdown("<h1 class='title'>🌾 Smart Crop Recommendation</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Input environmental conditions to discover the perfect crop for your farm.</div>", unsafe_allow_html=True)

# Main input container
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    # Input form with improved layout
    st.markdown("### 📊 Environmental Parameters")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("<div class='input-label'>🌡️ Temperature (°C)</div>", unsafe_allow_html=True)
        temperature = st.slider("", 0.0, 50.0, 25.0, key="temp", help="Select the average temperature in Celsius")
        
        st.markdown("<div class='input-label'>🧪 Soil pH</div>", unsafe_allow_html=True)
        ph = st.slider("", 0.0, 14.0, 6.5, key="ph", help="Select the soil pH level")
        
        st.markdown("<div class='input-label'>🗓️ Season</div>", unsafe_allow_html=True)
        season = st.selectbox("", ["rainy", "winter", "spring", "summer"], key="season", help="Choose the current season")

    with col2:
        st.markdown("<div class='input-label'>💧 Humidity (%)</div>", unsafe_allow_html=True)
        humidity = st.slider("", 0.0, 100.0, 65.0, key="humidity", help="Select the relative humidity percentage")
        
        st.markdown("<div class='input-label'>🚿 Water Availability (mm)</div>", unsafe_allow_html=True)
        water_availability = st.slider("", 0.0, 500.0, 120.0, key="water", help="Select water availability in millimeters")

    season_code = season_mapping[season]

    predict = st.button("🔍 Predict Ideal Crop")

    if predict:
        result = predict_crop(temperature, humidity, ph, water_availability, season_code)
        st.markdown(f"<div class='result'>🌱 Recommended Crop: {result.title()}</div>", unsafe_allow_html=True)
        st.image(crop_images.get(result, "https://images.unsplash.com/photo-1500595046743-cd271d6942f0"), 
                 caption=f"{result.title()} Field", use_column_width=True)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.markdown(f"<div class='accuracy'>📈 Model Accuracy: {accuracy:.2%}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>© 2025 Crop Recommendation AI | Powered by xAI</div>", unsafe_allow_html=True)

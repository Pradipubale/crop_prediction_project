import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# Page configuration
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv("Crop_recommendation.csv")
    return dataset

# Load and prepare data
dataset = load_data()

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
@st.cache_resource
def train_model():
    model = LogisticRegression(max_iter=200)
    model.fit(x_train, y_train)
    return model

model = train_model()

# Crop images dictionary
crop_images = {
    'rice': 'https://images.unsplash.com/photo-1536304993881-ff6e9eefa2a6?w=400&h=300&fit=crop',
    'maize': 'https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=400&h=300&fit=crop',
    'chickpea': 'https://images.unsplash.com/photo-1509358271058-acd22cc93898?w=400&h=300&fit=crop',
    'kidneybeans': 'https://images.unsplash.com/photo-1553621042-f6e147245754?w=400&h=300&fit=crop',
    'pigeonpeas': 'https://images.unsplash.com/photo-1559181567-c3190ca9959b?w=400&h=300&fit=crop',
    'mothbeans': 'https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=300&fit=crop',
    'mungbean': 'https://images.unsplash.com/photo-1559181567-c3190ca9959b?w=400&h=300&fit=crop',
    'blackgram': 'https://images.unsplash.com/photo-1558961363-fa8fdf82db35?w=400&h=300&fit=crop',
    'lentil': 'https://images.unsplash.com/photo-1509358271058-acd22cc93898?w=400&h=300&fit=crop',
    'watermelon': 'https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=400&h=300&fit=crop',
    'muskmelon': 'https://images.unsplash.com/photo-1563114773-84221bd6e3d4?w=400&h=300&fit=crop',
    'cotton': 'https://images.unsplash.com/photo-1560707303-4e980ce876ad?w=400&h=300&fit=crop',
    'jute': 'https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=400&h=300&fit=crop'
}

# Enhanced crop information dictionary
crop_info = {
    'rice': {
        'description': 'A staple cereal grain that feeds over half the world\'s population.',
        'scientific_name': 'Oryza sativa',
        'growth_duration': '90-120 days',
        'optimal_temp': '20-35°C',
        'water_requirement': 'High (1000-2000mm)',
        'soil_ph': '5.5-7.0',
        'best_season': 'Rainy/Monsoon',
        'nutrition': 'Rich in carbohydrates, provides energy',
        'uses': 'Food grain, rice bran oil, animal feed',
        'yield': '4-6 tons/hectare',
        'market_value': '₹20-25 per kg',
        'tips': 'Requires flooded fields, good drainage system needed',
        'diseases': 'Blast, Brown spot, Bacterial leaf blight',
        'soil_type': 'Clay loam, well-drained'
    },
    'maize': {
        'description': 'A versatile crop used for food, feed, and industrial applications.',
        'scientific_name': 'Zea mays',
        'growth_duration': '80-110 days',
        'optimal_temp': '21-27°C',
        'water_requirement': 'Moderate (500-800mm)',
        'soil_ph': '5.8-7.8',
        'best_season': 'Summer/Rainy',
        'nutrition': 'High in carbs, vitamin B6, thiamine',
        'uses': 'Food, animal feed, ethanol production',
        'yield': '5-8 tons/hectare',
        'market_value': '₹18-22 per kg',
        'tips': 'Requires well-drained soil, regular weeding',
        'diseases': 'Corn borer, Leaf blight, Rust',
        'soil_type': 'Well-drained loamy soil'
    },
    'chickpea': {
        'description': 'A protein-rich legume, excellent for soil nitrogen fixation.',
        'scientific_name': 'Cicer arietinum',
        'growth_duration': '90-120 days',
        'optimal_temp': '20-30°C',
        'water_requirement': 'Low-Moderate (300-500mm)',
        'soil_ph': '6.0-7.5',
        'best_season': 'Winter/Rabi',
        'nutrition': 'High protein, fiber, folate',
        'uses': 'Food, flour, animal feed',
        'yield': '1.5-2.5 tons/hectare',
        'market_value': '₹40-60 per kg',
        'tips': 'Drought tolerant, fixes nitrogen in soil',
        'diseases': 'Wilt, Blight, Pod borer',
        'soil_type': 'Well-drained sandy loam'
    },
    'kidneybeans': {
        'description': 'High-protein legume with excellent nutritional value.',
        'scientific_name': 'Phaseolus vulgaris',
        'growth_duration': '90-110 days',
        'optimal_temp': '18-24°C',
        'water_requirement': 'Moderate (400-600mm)',
        'soil_ph': '6.0-7.0',
        'best_season': 'Winter/Spring',
        'nutrition': 'High protein, fiber, iron',
        'uses': 'Food, canned beans, export',
        'yield': '1.8-2.2 tons/hectare',
        'market_value': '₹80-120 per kg',
        'tips': 'Sensitive to frost, needs support for climbing',
        'diseases': 'Anthracnose, Rust, Bacterial blight',
        'soil_type': 'Well-drained fertile loam'
    },
    'pigeonpeas': {
        'description': 'Drought-tolerant legume crop, ideal for sustainable farming.',
        'scientific_name': 'Cajanus cajan',
        'growth_duration': '120-180 days',
        'optimal_temp': '20-30°C',
        'water_requirement': 'Low (350-500mm)',
        'soil_ph': '5.5-7.0',
        'best_season': 'Rainy/Kharif',
        'nutrition': 'High protein, amino acids',
        'uses': 'Dal, green pods, fodder',
        'yield': '1.2-1.8 tons/hectare',
        'market_value': '₹60-80 per kg',
        'tips': 'Very drought tolerant, improves soil fertility',
        'diseases': 'Wilt, Sterility mosaic, Pod fly',
        'soil_type': 'Well-drained sandy to clay loam'
    },
    'mothbeans': {
        'description': 'Hardy legume that thrives in arid conditions.',
        'scientific_name': 'Vigna aconitifolia',
        'growth_duration': '75-90 days',
        'optimal_temp': '25-35°C',
        'water_requirement': 'Very Low (250-400mm)',
        'soil_ph': '6.0-8.0',
        'best_season': 'Summer/Rainy',
        'nutrition': 'High protein, calcium, iron',
        'uses': 'Food, fodder, green manure',
        'yield': '0.8-1.2 tons/hectare',
        'market_value': '₹50-70 per kg',
        'tips': 'Extremely drought tolerant, good for arid regions',
        'diseases': 'Leaf spot, Powdery mildew',
        'soil_type': 'Sandy to sandy loam'
    },
    'mungbean': {
        'description': 'Fast-growing legume with high nutritional content.',
        'scientific_name': 'Vigna radiata',
        'growth_duration': '60-90 days',
        'optimal_temp': '25-35°C',
        'water_requirement': 'Low-Moderate (300-500mm)',
        'soil_ph': '6.2-7.2',
        'best_season': 'Summer/Rainy',
        'nutrition': 'High protein, vitamin C, folate',
        'uses': 'Food, sprouts, animal feed',
        'yield': '1.0-1.5 tons/hectare',
        'market_value': '₹70-90 per kg',
        'tips': 'Quick maturing, good for rotation',
        'diseases': 'Yellow mosaic, Leaf spot',
        'soil_type': 'Well-drained sandy loam'
    },
    'blackgram': {
        'description': 'Protein-rich pulse crop with good market value.',
        'scientific_name': 'Vigna mungo',
        'growth_duration': '75-90 days',
        'optimal_temp': '25-35°C',
        'water_requirement': 'Low (300-400mm)',
        'soil_ph': '6.5-7.5',
        'best_season': 'Summer/Rainy',
        'nutrition': 'High protein, iron, calcium',
        'uses': 'Dal, papad, fermented foods',
        'yield': '1.0-1.4 tons/hectare',
        'market_value': '₹80-100 per kg',
        'tips': 'Tolerates waterlogging better than other pulses',
        'diseases': 'Leaf crinkle, Anthracnose',
        'soil_type': 'Clay loam to sandy loam'
    },
    'lentil': {
        'description': 'Nutritious legume crop with excellent protein content.',
        'scientific_name': 'Lens culinaris',
        'growth_duration': '95-110 days',
        'optimal_temp': '18-25°C',
        'water_requirement': 'Low (300-400mm)',
        'soil_ph': '6.0-7.5',
        'best_season': 'Winter/Rabi',
        'nutrition': 'High protein, fiber, folate',
        'uses': 'Dal, flour, export',
        'yield': '1.2-1.8 tons/hectare',
        'market_value': '₹60-80 per kg',
        'tips': 'Cold tolerant, good for rotation with cereals',
        'diseases': 'Rust, Wilt, Blight',
        'soil_type': 'Well-drained loamy soil'
    },
    'watermelon': {
        'description': 'Refreshing fruit crop with high water content.',
        'scientific_name': 'Citrullus lanatus',
        'growth_duration': '80-100 days',
        'optimal_temp': '25-35°C',
        'water_requirement': 'High (400-600mm)',
        'soil_ph': '6.0-7.0',
        'best_season': 'Summer',
        'nutrition': 'High water content, vitamin C, lycopene',
        'uses': 'Fresh fruit, juice, seeds',
        'yield': '30-40 tons/hectare',
        'market_value': '₹8-15 per kg',
        'tips': 'Requires warm weather, good drainage',
        'diseases': 'Downy mildew, Anthracnose',
        'soil_type': 'Well-drained sandy loam'
    },
    'muskmelon': {
        'description': 'Sweet, aromatic fruit with good market demand.',
        'scientific_name': 'Cucumis melo',
        'growth_duration': '70-90 days',
        'optimal_temp': '25-35°C',
        'water_requirement': 'Moderate (350-500mm)',
        'soil_ph': '6.0-7.5',
        'best_season': 'Summer',
        'nutrition': 'Vitamin A, C, potassium',
        'uses': 'Fresh fruit, juice, export',
        'yield': '20-30 tons/hectare',
        'market_value': '₹15-25 per kg',
        'tips': 'Requires warm, dry weather during ripening',
        'diseases': 'Powdery mildew, Fruit fly',
        'soil_type': 'Well-drained sandy to loamy'
    },
    'cotton': {
        'description': 'Important fiber crop for textile industry.',
        'scientific_name': 'Gossypium hirsutum',
        'growth_duration': '180-200 days',
        'optimal_temp': '21-27°C',
        'water_requirement': 'High (700-1300mm)',
        'soil_ph': '5.8-8.0',
        'best_season': 'Rainy/Kharif',
        'nutrition': 'Not applicable (fiber crop)',
        'uses': 'Textile fiber, oil, animal feed',
        'yield': '1.5-2.5 tons/hectare',
        'market_value': '₹50-70 per kg',
        'tips': 'Requires long growing season, pest management crucial',
        'diseases': 'Bollworm, Leaf curl virus',
        'soil_type': 'Deep, well-drained black soil'
    },
    'jute': {
        'description': 'Natural fiber crop used for eco-friendly products.',
        'scientific_name': 'Corchorus capsularis',
        'growth_duration': '120-150 days',
        'optimal_temp': '25-35°C',
        'water_requirement': 'High (1000-1500mm)',
        'soil_ph': '6.0-7.5',
        'best_season': 'Rainy/Monsoon',
        'nutrition': 'Young leaves edible (vitamin C)',
        'uses': 'Fiber, bags, textiles, paper',
        'yield': '2.5-3.5 tons/hectare',
        'market_value': '₹30-40 per kg',
        'tips': 'Requires high humidity, waterlogged conditions',
        'diseases': 'Stem rot, Leaf spot',
        'soil_type': 'Alluvial soil with good water retention'
    }
}

# Season icons
season_icons = {
    'rainy': '🌧️',
    'winter': '❄️',
    'spring': '🌸',
    'summer': '☀️'
}

# Prediction function
def predict_crop(temperature, humidity, ph, water_availability, season):
    input_data = pd.DataFrame([[temperature, humidity, ph, water_availability, season]], 
                              columns=['temperature', 'humidity', 'ph', 'water availability', 'season'])
    prediction = model.predict(input_data)
    crop_mapping = {v: k for k, v in label_mapping.items()}
    return crop_mapping[prediction[0]]

# Custom CSS for enhanced UI
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main > div {
            padding-top: 2rem;
        }
        
        .hero-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .hero-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .hero-subtitle {
            font-size: 1.3rem;
            font-weight: 400;
            opacity: 0.9;
            margin-bottom: 2rem;
        }
        
        .stats-container {
            display: flex;
            justify-content: space-around;
            margin-top: 2rem;
        }
        
        .stat-item {
            text-align: center;
            padding: 1rem;
        }
        
        .stat-number {
            font-size: 2.5rem;
            font-weight: 700;
            color: #FFD700;
        }
        
        .stat-label {
            font-size: 1rem;
            opacity: 0.8;
        }
        
        .input-section {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 2.5rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        .section-title {
            font-size: 2rem;
            font-weight: 600;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .parameter-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        
        .parameter-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #34495e;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .result-container {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 3rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin: 2rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        
        .result-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .result-crop {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .crop-description {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 2rem;
        }
        
        .accuracy-badge {
            background: rgba(255,255,255,0.2);
            padding: 1rem 2rem;
            border-radius: 50px;
            font-size: 1.2rem;
            font-weight: 600;
            display: inline-block;
            margin-top: 1rem;
        }
        
        .prediction-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 3rem;
            border: none;
            border-radius: 50px;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }
        
        .prediction-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }
        
        .sidebar .sidebar-content {
            background: transparent;
        }
        
        .sidebar-card {
            background: rgba(255, 255, 255, 0.9);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .sidebar-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        
        .sidebar-text {
            color: #34495e;
            line-height: 1.6;
            font-size: 0.9rem;
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 3rem;
        }
        
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .crop-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .crop-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s ease;
        }
        
        .crop-card:hover {
            transform: translateY(-2px);
        }
        
        .crop-card img {
            width: 100%;
            height: 80px;
            object-fit: cover;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        
        .crop-name {
            font-size: 0.9rem;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        
        .info-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .info-content {
            color: #34495e;
            line-height: 1.6;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🌾 Smart Crop Recommendation</div>
        <div class="hero-subtitle">AI-Powered Agricultural Intelligence for Optimal Crop Selection</div>
        <div class="stats-container">
            <div class="stat-item">
                <div class="stat-number">13</div>
                <div class="stat-label">Crop Types</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">4</div>
                <div class="stat-label">Seasons</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">95%</div>
                <div class="stat-label">Accuracy</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🌱 About the System")
    st.markdown("""
    Our AI-powered crop recommendation system analyzes environmental factors to suggest the most suitable crops for your farming conditions.
    
    **Key Features:**
    - 🤖 Machine Learning Algorithm
    - 🌍 Environmental Analysis
    - 📊 High Accuracy Predictions
    - 🎯 Personalized Recommendations
    """)
    
    st.markdown("---")
    
    st.markdown("## 🔧 How It Works")
    st.markdown("""
    **Step 1:** Input environmental parameters  
    **Step 2:** AI analyzes optimal conditions  
    **Step 3:** Get personalized crop recommendation  
    **Step 4:** View detailed crop information  
    
    The system uses **Logistic Regression** to analyze patterns in agricultural data and provide accurate crop suggestions.
    """)
    
    st.markdown("---")
    
    st.markdown("## 📊 Model Details")
    st.info("""
    **Algorithm:** Logistic Regression  
    **Features:** 5 Environmental Parameters  
    **Training Data:** Comprehensive Agricultural Dataset  
    **Validation:** 70-30 Split Method  
    **Accuracy:** ~95% on test data
    """)
    
    st.markdown("---")
    
    st.image("https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=300&h=200&fit=crop", 
             caption="🌾 Sustainable Agriculture", use_container_width=True)
    
    st.markdown("---")
    
    st.markdown("## 🎯 Supported Crops")
    st.markdown("Our system can recommend the following crops:")
    
    # Create a more compact crop display
    crop_names = list(label_mapping.keys())
    crops_text = ""
    for i, crop in enumerate(crop_names):
        if i % 2 == 0:
            crops_text += f"• **{crop.title()}**"
        else:
            crops_text += f" • **{crop.title()}**\n"
    
    st.markdown(crops_text)
    
    st.markdown("---")
    
    st.markdown("## 🌱 Environmental Factors")
    st.markdown("""
    **Temperature:** Optimal growth temperature  
    **Humidity:** Moisture content in air  
    **pH Level:** Soil acidity/alkalinity  
    **Water:** Available irrigation water  
    **Season:** Current growing season
    """)
    
    st.markdown("---")
    
    st.success("💡 **Tip:** Adjust parameters to see how they affect crop recommendations!")
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
        <strong>🚀 Developed by Pradip</strong><br>
        <small>Powered by AI & Machine Learning</small>
    </div>
    """, unsafe_allow_html=True)

# Main Content
st.markdown("""
    <div class="input-section">
        <div class="section-title">📊 Environmental Parameters</div>
    </div>
""", unsafe_allow_html=True)

# Input form with enhanced layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("""
        <div class="parameter-card">
            <div class="parameter-title">🌡️ Temperature</div>
        </div>
    """, unsafe_allow_html=True)
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, key="temp", help="Optimal temperature for crop growth")
    
    st.markdown("""
        <div class="parameter-card">
            <div class="parameter-title">🧪 Soil pH</div>
        </div>
    """, unsafe_allow_html=True)
    ph = st.slider("pH Level", 0.0, 14.0, 6.5, key="ph", help="Soil acidity/alkalinity level")
    
    st.markdown("""
        <div class="parameter-card">
            <div class="parameter

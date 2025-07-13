import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import os

# Optional for Streamlit — uncomment if using
# import streamlit as st

# --- Debug: Check current directory and files
print("Current directory:", os.getcwd())
print("Files here:", os.listdir())

# --- Load the dataset ---
try:
    dataset = pd.read_csv("crop_recommendation.csv")  # Ensure correct case
except FileNotFoundError:
    print("❌ File not found. Please make sure 'crop_recommendation.csv' is in the same folder.")
    exit()

# --- Check columns ---
print("Columns:", dataset.columns.tolist())

# --- Ensure correct column names ---
expected_columns = ['temperature', 'humidity', 'ph', 'water availability', 'season', 'label']
missing_cols = [col for col in expected_columns if col not in dataset.columns]
if missing_cols:
    print(f"❌ Missing columns in dataset: {missing_cols}")
    exit()

# --- Map categorical values to numeric ---
label_mapping = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9,
    'watermelon': 10, 'muskmelon': 11, 'cotton': 12, 'jute': 13
}
season_mapping = {'rainy': 1, 'winter': 2, 'spring': 3, 'summer': 4}

dataset['label'] = dataset['label'].map(label_mapping)
dataset['season'] = dataset['season'].map(season_mapping)

# --- Check for NaNs ---
if dataset.isnull().sum().any():
    print("❌ Dataset contains missing values after mapping:")
    print(dataset.isnull().sum())
    exit()

# --- Features and target ---
X = dataset[['temperature', 'humidity', 'ph', 'water availability', 'season']]
y = dataset['label']

# --- Train-test split ---
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Train Logistic Regression Model ---
model = LogisticRegression(max_iter=200, solver='lbfgs', multi_class='multinomial')
model.fit(x_train, y_train)

# --- Evaluate ---
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy of the model:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Pairplot visualization (optional for Streamlit) ---
try:
    pairplot_fig = sns.pairplot(dataset[['temperature', 'humidity', 'ph', 'water availability', 'label']], hue='label')
    plt.show()

    # If using Streamlit:
    # st.pyplot(pairplot_fig)
except Exception as e:
    print("⚠️ Could not render pairplot:", e)

# --- Prediction Function ---
def predict_crop(temperature, humidity, ph, water_availability, season):
    input_data = pd.DataFrame([[temperature, humidity, ph, water_availability, season]],
                              columns=['temperature', 'humidity', 'ph', 'water availability', 'season'])
    prediction = model.predict(input_data)
    crop_mapping = {v: k for k, v in label_mapping.items()}
    return crop_mapping.get(prediction[0], "Unknown Crop")

# --- Test Example ---
print("🔎 Predicted Crop:", predict_crop(20, 82.1, 6.11, 202.12, 1))

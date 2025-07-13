import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
dataset = dataset = pd.read_csv("Crop_recommendation.csv")


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

# Evaluate the model
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the model:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the data using seaborn pairplot
sns.pairplot(dataset[['temperature', 'humidity', 'ph', 'water availability', 'label']], hue='label')
plt.show()

# Crop Prediction Function
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

# Example usage
print("Predicted Crop:", predict_crop(20, 82.1, 6.11, 202.12, 1))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns

# Load the dataset
dataset = pd.read_csv("Crop_recommendation.csv")

# Replace categorical values with numerical codes for 'label' and 'season'
label_mapping = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 
    'watermelon': 10, 'muskmelon': 11, 'cotton': 12, 'jute': 13
}
season_mapping = {'rainy': 1, 'winter': 2, 'spring': 3, 'summer': 4}

dataset['label'] = dataset['label'].map(label_mapping)
dataset['season'] = dataset['season'].map(season_mapping)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    dataset[['temperature', 'humidity', 'ph', 'water availability', 'season']], 
    dataset['label'], 
    test_size=0.3, 
    random_state=42
)

# Initialize the Logistic Regression model with increased max_iter
lr = LogisticRegression(max_iter=200)
lr.fit(x_train, y_train)

# Evaluate the model
score = lr.score(x_test, y_test)
print("Model Score:", score)

# Visualize the data
sns.pairplot(dataset[['temperature', 'humidity', 'ph', 'water availability', 'label']], hue='label')

# Create a DataFrame with a single sample for prediction
input_data = pd.DataFrame([[20, 82.1, 6.11, 202.12, 1]], 
                           columns=['temperature', 'humidity', 'ph', 'water availability', 'season'])

# Make a prediction
prediction = lr.predict(input_data)
print("Prediction:", prediction)

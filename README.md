# Crop Recommendation System

This project is a Crop Recommendation System built using Python, Scikit-Learn, and Streamlit. The application uses a Logistic Regression model trained on crop data to predict the most suitable crop based on the user's input parameters such as temperature, humidity, pH level, water availability, and season.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Contributing](#contributing)

## Project Overview
The Crop Recommendation System helps farmers and agriculture enthusiasts select the most appropriate crop for given environmental conditions. The model leverages features such as temperature, humidity, soil pH, water availability, and season to predict a crop label. It then maps the label to a specific crop, providing an easy-to-understand recommendation.

## Dataset
The dataset used for this project contains environmental parameters and crop labels. Here are the main columns:
- **temperature**: Average temperature for the crop to grow
- **humidity**: Average humidity required for the crop
- **ph**: Soil pH level suitable for the crop
- **water availability**: Average water availability
- **season**: Season suitable for the crop (encoded as 1 for Rainy, 2 for Winter, etc.)
- **label**: Crop label (mapped to specific crops in the app)

## Requirements
- Python 3.6 or above
- Streamlit
- Pandas
- Scikit-Learn
- Seaborn

## Installation and Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/crop-recommendation-system.git
    cd crop-recommendation-system
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure that the dataset file (`Crop_recommendation.csv`) is in the correct directory or update the path in the code.

4. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open the app in a browser (the link will appear in the terminal after running the command).
2. Enter values for **Temperature**, **Humidity**, **pH Level**, **Water Availability**, and **Season**.
3. Click the **Predict Crop** button to get a recommendation.
4. The recommended crop based on input parameters will be displayed.

## File Structure

```markdown
.
├── Crop_recommendation.csv    # Dataset file with crop data
├── app.py                      # Main Streamlit app script
├── README.md                   # Project documentation
└── requirements.txt            # List of dependencies

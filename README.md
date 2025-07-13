
---

## 📄 Final `README.md`

```markdown
# 🌾 Smart Crop Recommendation System

An intelligent web application built with **Streamlit** and **Scikit-learn** that predicts the most suitable crop to cultivate based on environmental parameters like **temperature**, **humidity**, **soil pH**, **water availability**, and **season**.  
The model uses **Logistic Regression** to classify crops, aiming to assist farmers and agricultural planners in making informed crop selection decisions.

---

## 📸 Live Demo

👉 [Check out the live app here 🚀](https://croppredictionproject-zrpwxkqjfofbubenatkjvi.streamlit.app/)

---

## 📖 Features

- 📊 **User-friendly interface** powered by Streamlit.
- 🔍 Predicts suitable crop based on:
  - Temperature
  - Humidity
  - Soil pH
  - Water Availability
  - Season
- 🌸 Displays:
  - Crop image
  - Short description
  - Model accuracy
- 🎨 Clean, responsive, and visually enhanced UI with custom CSS styling.
- 🚀 Fast predictions using Logistic Regression.

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Pandas**
- **Scikit-learn (Logistic Regression)**
- **Custom CSS styling**
- **Google Fonts**
- **Unsplash crop images**

---

## 📂 Project Structure

```

├── Crop\_recommendation.csv      # Dataset
├── app.py                       # Streamlit app
├── README.md                    # Project documentation
└── requirements.txt             # Required Python libraries

````

---

## 📦 Installation & Running the App

1️⃣ **Clone the repository**
```bash
git clone https://github.com/pradipubale/smart-crop-recommendation.git
cd smart-crop-recommendation
````

2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

3️⃣ **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 📈 Dataset Information

* Source: `Crop_recommendation.csv`
* Features:

  * `temperature`
  * `humidity`
  * `ph`
  * `water availability`
  * `season`
* Target:

  * `label` (crop name)

Crops considered:
`rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `watermelon`, `muskmelon`, `cotton`, `jute`.

---

## 📊 Model Performance

* Algorithm: **Logistic Regression**
* Training/Testing Split: 70/30
* Model Accuracy: Displayed dynamically within the app interface.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

* **Pradip Ubale**
* GitHub: [@pradipubale](https://github.com/pradipubale)

---

## 🌟 Acknowledgements

* [Streamlit Documentation](https://docs.streamlit.io/)
* [Scikit-learn Documentation](https://scikit-learn.org/stable/)
* [Unsplash](https://unsplash.com/) for free images.

---

## 🚀 Future Improvements

* Integrate additional ML models for comparison.
* Support multi-season crop planning.
* Include soil type as an input parameter.
* Deploy on **Render** or **Streamlit Cloud** (already live ✅)

---

```

---

## 📦 `requirements.txt`

Here’s a clean, working `requirements.txt` for your project:

```

streamlit
pandas
scikit-learn
numpy

```

If you used any additional libraries (like `matplotlib`, `PIL`, or custom CSS importers) — let me know, I can add those too.

---

✅ Done!  
Would you like me to generate a `LICENSE` file (MIT license) as well?
```

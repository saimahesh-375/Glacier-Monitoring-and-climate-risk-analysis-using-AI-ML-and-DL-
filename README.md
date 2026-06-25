# 🌍 Glacier Monitoring and Climate Risk Analysis using AI, ML, and DL

An AI-powered web application that monitors glaciers, predicts climate risk levels, and forecasts future glacier retreat using Machine Learning and Deep Learning techniques. The system also provides interactive satellite visualization for glacier locations.

---

## 🚀 Live Demo

🌐 https://glacier-monitoring.onrender.com

---

## 📂 GitHub Repository

https://github.com/saimahesh-375/Glacier-Monitoring-AI

---

## 📖 Project Overview

Glaciers are important indicators of climate change. Continuous glacier melting leads to rising sea levels and environmental imbalance. This project uses Artificial Intelligence techniques to analyze glacier data and provide:

- Glacier Risk Classification (HIGH / LOW)
- Future Glacier Retreat Prediction
- Interactive Satellite Map Visualization
- Physical Glacier Information

The application is developed using Python Flask and integrates Machine Learning, Deep Learning, and GIS visualization into a single platform.

---

## ✨ Features

- Glacier selection from dropdown
- Physical glacier information
- Climate risk prediction using Random Forest
- Future retreat prediction using LSTM
- Interactive satellite map using Folium
- User-friendly web interface
- Deployed on Render

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Web Framework
- Flask

### Machine Learning
- Random Forest (Scikit-learn)

### Deep Learning
- LSTM (TensorFlow/Keras)

### Data Processing
- Pandas
- NumPy

### Visualization
- Folium
- OpenStreetMap
- Esri Satellite Imagery

### Model Persistence
- Joblib

### Frontend
- HTML
- CSS
- JavaScript

### Deployment
- Render

---

## 📊 Dataset

The dataset contains glacier information including:

- Glacier Name
- Length (km)
- Area (km²)
- Elevation (m)
- Retreat Rate (m/year)
- Temperature Increase (°C)
- Latitude
- Longitude

---

## ⚙️ System Workflow

User Selects Glacier
↓
Retrieve Glacier Data
↓
Random Forest → Risk Prediction
↓
LSTM → Future Retreat Prediction
↓
Satellite Map Visualization
↓
Display Results

---

## 📁 Project Structure

```
Glacier-Monitoring/
│
├── app.py
├── requirements.txt
├── data/
│   └── glacier_climate.csv
├── models/
│   ├── ml_risk_model.pkl
│   ├── scaler.pkl
│   └── dl_retreat_model.keras
├── static/
├── templates/
│   ├── index.html
│   └── glacier_detail.html
└── README.md
```

---

## 🧠 Machine Learning Model

Algorithm:
- Random Forest Classifier

Purpose:
- Predict glacier climate risk

Output:
- HIGH Risk
- LOW Risk

---

## 🤖 Deep Learning Model

Algorithm:
- LSTM (Long Short-Term Memory)

Purpose:
- Predict future glacier retreat trends

---

## 🌐 Deployment

The application is deployed on Render and can be accessed using:

[https://glacier-monitoring.onrender.com]

---

## 👨‍💻 Author

**B Sai Mahesh**

GitHub:
https://github.com/saimahesh-375

---

## 📜 License

This project is developed for academic and educational purposes.

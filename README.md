# 🌍 Climate Insight Pro — AI-Powered Climate Analytics Dashboard

## 📌 Overview
Climate Insight Pro is an end-to-end data science project that combines Exploratory Data Analysis (EDA), Machine Learning, Time-Series Analysis, and Deep Learning (LSTM) to analyze climate patterns and predict rainfall.

It provides an interactive Streamlit dashboard for real-time prediction and visualization.

---

## 🚀 Features

### 📊 Exploratory Data Analysis
- Rainfall distribution
- Correlation heatmap
- Feature insights

### 📈 Trend & Seasonality
- Rainfall trend over time
- Monthly seasonal patterns

### 🤖 Machine Learning
- RandomForest Classifier
- Predicts Rainfall (Yes/No)
- Handles imbalance
- Shows prediction probability

### ⏳ Time-Series Analysis
- Rolling mean (30-day)
- Trend smoothing

### 🧠 Deep Learning (LSTM)
- Forecasts future rainfall
- Captures time-based patterns

### 🌐 Streamlit Dashboard
- Interactive UI
- Sidebar inputs
- Real-time prediction

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 📂 Project Structure

climate_project/
│
├── app.py
├── train_model.py
├── lstm_model.py
├── model.pkl
├── lstm_model.h5
├── scaler.pkl
├── weatherAUS.csv
├── requirements.txt
└── README.md

---

## ▶️ How to Run

### Install dependencies
pip install -r requirements.txt

### Train models
python train_model.py  
python lstm_model.py  

### Run app
streamlit run app.py

---

## 📊 Dataset
Weather Dataset (Australia)

---

## 🔮 Future Improvements
- LSTM tuning
- ARIMA forecasting
- Live weather API
- Cloud deployment

---

## 💡 Impact
- Climate trend analysis  
- Rainfall prediction  
- Agriculture planning  

---

## 👨‍💻 Author
Om
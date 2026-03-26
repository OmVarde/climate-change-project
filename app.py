import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
lstm_model = load_model("lstm_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load dataset
data = pd.read_csv("weatherAUS.csv")
data = data.dropna()

# Date processing
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")
data["Month"] = data["Date"].dt.month

# Page config
st.set_page_config(page_title="Climate Insight Pro", layout="wide")

st.title("🌍 Climate Insight Pro Dashboard")

# ================= SIDEBAR =================
st.sidebar.header("🌦 Input Weather Data")

MinTemp = st.sidebar.slider("Min Temp", 0, 40, 15)
MaxTemp = st.sidebar.slider("Max Temp", 0, 50, 25)
Humidity9am = st.sidebar.slider("Humidity 9am", 0, 100, 50)
Humidity3pm = st.sidebar.slider("Humidity 3pm", 0, 100, 50)
Pressure9am = st.sidebar.slider("Pressure 9am", 900, 1100, 1000)
Pressure3pm = st.sidebar.slider("Pressure 3pm", 900, 1100, 1000)
Temp9am = st.sidebar.slider("Temp 9am", 0, 40, 20)
Temp3pm = st.sidebar.slider("Temp 3pm", 0, 50, 25)

# ================= PREDICTION =================
if st.sidebar.button("Predict Rainfall"):

    input_data = np.array([[MinTemp, MaxTemp, Humidity9am, Humidity3pm,
                            Pressure9am, Pressure3pm, Temp9am, Temp3pm]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.success(f"🌧 Rain Expected (Confidence: {prob*100:.2f}%)")
    else:
        st.warning(f"☀ No Rain (Confidence: {(1-prob)*100:.2f}%)")

# ================= EDA =================
st.header("📊 Exploratory Data Analysis")

# Distribution
st.subheader("Rainfall Distribution")
fig1 = px.histogram(data, x="Rainfall", nbins=50)
st.plotly_chart(fig1, use_container_width=True)

# ================= TREND =================
st.subheader("📈 Rainfall Trend Over Time")
st.line_chart(data["Rainfall"])

# ================= SEASONALITY =================
st.subheader("📅 Monthly Rainfall Pattern")
monthly = data.groupby("Month")["Rainfall"].mean()
st.bar_chart(monthly)

# ================= TIME SERIES =================
st.subheader("⏳ Time-Series Analysis (Rolling Mean)")
data["RollingMean"] = data["Rainfall"].rolling(30).mean()
st.line_chart(data[["Rainfall", "RollingMean"]])

# ================= HEATMAP =================
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.select_dtypes(include=['float64','int64']).corr(), ax=ax)
st.pyplot(fig)

# ================= FEATURE OVERVIEW =================
st.subheader("📊 Input Feature Overview")

df = pd.DataFrame({
    "Feature": ["MinTemp","MaxTemp","Humidity9am","Humidity3pm",
                "Pressure9am","Pressure3pm","Temp9am","Temp3pm"],
    "Value": [MinTemp,MaxTemp,Humidity9am,Humidity3pm,
              Pressure9am,Pressure3pm,Temp9am,Temp3pm]
})

fig2 = px.bar(df, x="Feature", y="Value", color="Feature")
st.plotly_chart(fig2, use_container_width=True)
# ================= DEEP LEARNING =================
st.header("🧠 Deep Learning Forecast (LSTM)")

rainfall = data["Rainfall"].values.reshape(-1, 1)
rainfall_scaled = scaler.transform(rainfall)

last_30 = rainfall_scaled[-30:]
last_30 = last_30.reshape(1, 30, 1)

future = []

for _ in range(10):
    pred = lstm_model.predict(last_30)[0][0]
    future.append(pred)

    last_30 = np.append(last_30[:,1:,:], [[[pred]]], axis=1)

# Convert back
future = scaler.inverse_transform(np.array(future).reshape(-1,1))

st.line_chart(future)


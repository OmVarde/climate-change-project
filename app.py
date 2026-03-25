import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset
data = pd.read_csv("weatherAUS.csv")
data = data.dropna()

# Page config
st.set_page_config(page_title="Climate Insight Pro", layout="wide")

st.title("🌍 Climate Insight Dashboard")

# Sidebar
st.sidebar.header("Input Weather Data")

MinTemp = st.sidebar.slider("Min Temp", 0, 40, 15)
MaxTemp = st.sidebar.slider("Max Temp", 0, 50, 25)
Humidity9am = st.sidebar.slider("Humidity 9am", 0, 100, 50)
Humidity3pm = st.sidebar.slider("Humidity 3pm", 0, 100, 50)
Pressure9am = st.sidebar.slider("Pressure 9am", 900, 1100, 1000)
Pressure3pm = st.sidebar.slider("Pressure 3pm", 900, 1100, 1000)
Temp9am = st.sidebar.slider("Temp 9am", 0, 40, 20)
Temp3pm = st.sidebar.slider("Temp 3pm", 0, 50, 25)

# Prediction
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

# ---------------- EDA SECTION ---------------- #

st.header("📊 Exploratory Data Analysis")

# 1. Distribution
st.subheader("Rainfall Distribution")
fig1 = px.histogram(data, x="Rainfall", nbins=50)
st.plotly_chart(fig1, use_container_width=True)

# 2. Trend (Time Series)
st.subheader("📈 Rainfall Trend")

data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date")

st.line_chart(data["Rainfall"])

# 3. Correlation Heatmap
st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.select_dtypes(include=['float64','int64']).corr(), ax=ax)
st.pyplot(fig)

# 4. Feature Overview
st.subheader("📊 Input Feature Overview")

df = pd.DataFrame({
    "Feature": ["MinTemp","MaxTemp","Humidity9am","Humidity3pm",
                "Pressure9am","Pressure3pm","Temp9am","Temp3pm"],
    "Value": [MinTemp,MaxTemp,Humidity9am,Humidity3pm,
              Pressure9am,Pressure3pm,Temp9am,Temp3pm]
})

fig2 = px.bar(df, x="Feature", y="Value", color="Feature")
st.plotly_chart(fig2, use_container_width=True)

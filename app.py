# ==============================
# Climate Insight Pro - App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ==============================
# Load Model
# ==============================
model = pickle.load(open("model.pkl", "rb"))

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Climate Insight Pro",
    layout="wide",
    page_icon="🌍"
)

# ==============================
# Custom Styling
# ==============================
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: white;
    }
    h1, h2, h3 {
        color: #00adb5;
    }
    .stButton>button {
        background-color: #00adb5;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Title Section
# ==============================
st.title("🌍 Climate Insight Pro")
st.markdown("### AI Powered Rainfall Prediction Dashboard")

# ==============================
# Sidebar Navigation
# ==============================
page = st.sidebar.selectbox("📌 Navigation", [
    "🏠 Home",
    "📊 Prediction",
    "📈 Insights",
    "ℹ About"
])

# ==============================
# HOME PAGE
# ==============================
if page == "🏠 Home":
    st.markdown("## Welcome 👋")
    st.write("""
    This project predicts **Rain Tomorrow** using Machine Learning.

    🔹 Built with: Streamlit + Scikit-learn  
    🔹 Model: Random Forest  
    🔹 Dataset: Weather Data  

    👉 Go to Prediction tab to try it live.
    """)

# ==============================
# PREDICTION PAGE
# ==============================
elif page == "📊 Prediction":

    st.subheader("📥 Enter Weather Details")

    col1, col2 = st.columns(2)

    with col1:
        mintemp = st.slider("Min Temperature", -10, 40, 15)
        maxtemp = st.slider("Max Temperature", 0, 50, 25)
        rainfall = st.slider("Rainfall", 0.0, 50.0, 0.0)
        evaporation = st.slider("Evaporation", 0.0, 20.0, 5.0)

    with col2:
        sunshine = st.slider("Sunshine", 0.0, 12.0, 7.0)
        windspeed = st.slider("Wind Speed", 0, 100, 20)
        humidity = st.slider("Humidity", 0, 100, 50)
        pressure = st.slider("Pressure", 980, 1050, 1010)

    # Simple input (numeric only for now)
    input_data = np.array([[mintemp, maxtemp, rainfall, evaporation,
                            sunshine, windspeed, humidity, pressure]])

    if st.button("🚀 Predict Rainfall"):

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.markdown("---")
        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.success(f"🌧 Rain Expected (Confidence: {prob:.2f})")
            st.progress(int(prob * 100))
        else:
            st.warning(f"☀ No Rain (Confidence: {1-prob:.2f})")
            st.progress(int((1 - prob) * 100))


# ==============================
# INSIGHTS PAGE
# ==============================
elif page == "📈 Insights":

    st.subheader("📊 Sample Data Visualization")

    # Dummy data for visualization (replace later with real dataset)
    df = pd.DataFrame({
        "Feature": ["Temp", "Humidity", "Pressure", "Rainfall"],
        "Value": [25, 60, 1010, 5]
    })

    fig = px.bar(df, x="Feature", y="Value", color="Feature",
                 title="Weather Overview")

    st.plotly_chart(fig, use_container_width=True)


# ==============================
# ABOUT PAGE
# ==============================
elif page == "ℹ About":

    st.markdown("""
    ## About This Project

    This is a Machine Learning web app built using:

    - 🧠 Random Forest Classifier  
    - 🌐 Streamlit UI  
    - 📊 Plotly Visualizations  

    ### Developer:
    Om 🚀

    ### Goal:
    To predict rainfall and analyze climate patterns.
    """)

# ==============================
# Footer
# ==============================
st.markdown("---")


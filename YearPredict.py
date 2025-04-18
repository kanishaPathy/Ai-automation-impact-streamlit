import streamlit as st
from PIL import Image
import pandas as pd
import joblib
import numpy as np
from streamlit_lottie import st_lottie
import json
import os


model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

# Page config
st.set_page_config(page_title="AI & Automation Impact Predictor", layout="wide")

# Load model
model = joblib.load("models/xgboost_automation_model.pkl")

# Load optional Lottie animation
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_ai = None
if os.path.exists("ai_animation.json"):
    lottie_ai = load_lottie_file("ai_animation.json")

# Try loading the image
image = None
if os.path.exists("ai_icon.png"):
    image = Image.open("ai_icon.png")

# Header section
with st.container():
    col_img, col_title = st.columns([1, 4])
    
    with col_img:
        if image:
            st.image(image, width=100)
        elif lottie_ai:
            st_lottie(lottie_ai, height=100)
        else:
            st.markdown("🤖")
    
    with col_title:
        st.markdown("<h1 style='margin-bottom: 0;'>AI & Automation Impact Predictor</h1>", unsafe_allow_html=True)
        st.markdown("Understand the projected impact of AI on your sector using key insights")

st.markdown("---")

# Input form
with st.container():
    st.subheader("📥 Input Your Data")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("Country", ["USA", "India", "Ireland", "Germany", "Canada"])
            sector = st.selectbox("Sector", ["IT", "Healthcare", "Finance", "Retail", "Education"])
            year = st.slider("Year", min_value=2010, max_value=2030, value=2024)
            education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

        with col2:
            avg_pre_ai = st.number_input("Average Pre-AI Unemployment Rate", min_value=0.0, max_value=100.0, step=0.1)
            avg_post_ai = st.number_input("Average Post-AI Unemployment Rate", min_value=0.0, max_value=100.0, step=0.1)
            avg_automation_impact = st.number_input("Avg. Automation Impact (%)", min_value=0.0, max_value=100.0, step=0.1)
            avg_ai_role_jobs = st.number_input("Avg. AI Role Jobs Created", min_value=0.0, step=0.1)
            avg_reskilling_programs = st.number_input("Avg. Reskilling Programs", min_value=0.0, step=0.1)
            avg_economic_impact = st.number_input("Avg. Economic Impact (%)", min_value=-100.0, max_value=100.0, step=0.1)
            avg_sector_growth = st.number_input("Avg. Sector Growth (%)", min_value=-100.0, max_value=100.0, step=0.1)

        submitted = st.form_submit_button("🚀 Predict Impact")

# Prediction logic
if submitted:
    input_data = {
        '_id.Country': [country],
        '_id.Sector': [sector],
        '_id.Year': [year],
        '_id.EducationLevel': [education_level],
        'Avg_PreAI': [avg_pre_ai],
        'Avg_PostAI': [avg_post_ai],
        'Avg_Automation_Impact': [avg_automation_impact],
        'Avg_AI_Role_Jobs': [avg_ai_role_jobs],
        'Avg_ReskillingPrograms': [avg_reskilling_programs],
        'Avg_EconomicImpact': [avg_economic_impact],
        'Avg_SectorGrowth': [avg_sector_growth]
    }

    df = pd.DataFrame(input_data)

    # Encode categorical variables
    df_encoded = pd.get_dummies(df)

    # Ensure all columns match the model
    model_features = model.get_booster().feature_names
    for feature in model_features:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0  # Add missing features as 0

    df_encoded = df_encoded[model_features]  # Reorder

    # Predict
    prediction = model.predict(df_encoded)[0]

    st.success(f"📊 Predicted Automation Impact: **{prediction:.2f}%**")

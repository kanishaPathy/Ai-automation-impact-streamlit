import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import joblib
import json
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="AI Automation Impact Predictor", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ai = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_2glqweqs.json")
# You must have this JSON animation file
image = Image.open("ai_icon.png")          # Tech-themed image

# Load model
model = joblib.load("models/xgboost_automation_model.pkl")

# Load CSV
try:
    df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
except FileNotFoundError:
    st.error("CSV file not found. Please check the file path.")
    st.stop()

# Flatten _id if needed
if '_id' in df.columns and isinstance(df['_id'].iloc[0], str) == False:
    df = pd.concat([df.drop(columns=['_id']), df['_id'].apply(pd.Series)], axis=1)

# Sidebar layout
with st.sidebar:
    st_lottie(lottie_ai, height=200)
    st.image(image, width=150)
    st.markdown("### Navigate")
    st.page_link("YearPredict.py", label="📊 Yearly Impact Prediction", disabled=True)

# Main layout
st.markdown(
    "<h1 style='font-size: 50px; color: #4A90E2;'>🔮 Predict AI & Automation Impact</h1>", 
    unsafe_allow_html=True
)
st.markdown("##### Fill in the details below to predict future automation impact in your sector.")

# Input selections
with st.form("predict_form"):
    cols = st.columns(2)

    country = cols[0].selectbox("Country", sorted(df['Country'].dropna().unique()))
    sector = cols[1].selectbox("Sector", sorted(df['Sector'].dropna().unique()))
    
    year = cols[0].selectbox("Year", sorted(df['Year'].dropna().unique()))
    education_level = cols[1].selectbox("Education Level", sorted(df['EducationLevel'].dropna().unique()))

    avg_pre_ai = st.slider("Average Pre-AI Unemployment", 0.0, 100.0, 50.0)
    avg_post_ai = st.slider("Average Post-AI Unemployment", 0.0, 100.0, 50.0)
    avg_automation_impact = st.slider("Automation Impact (%)", 0.0, 100.0, 50.0)
    avg_ai_role_jobs = st.slider("AI Role Jobs (%)", 0.0, 100.0, 50.0)
    avg_reskilling_programs = st.slider("Reskilling Programs Availability (%)", 0.0, 100.0, 50.0)
    avg_economic_impact = st.slider("Economic Impact Score", 0.0, 100.0, 50.0)
    avg_sector_growth = st.slider("Sector Growth (%)", 0.0, 100.0, 50.0)

    submit_btn = st.form_submit_button("🚀 Predict Impact")

if submit_btn:
    # Prepare input for model
    input_data = pd.DataFrame({
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
    })

    # One-hot encoding if needed (depends on how the model was trained)
    input_encoded = pd.get_dummies(input_data)
    model_columns = model.get_booster().feature_names
    for col in model_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[model_columns]

    # Make prediction
    prediction = model.predict(input_encoded)[0]

    # Display result
    st.success(f"✅ Predicted Automation Impact for {year}: **{round(prediction, 2)}%**")

    st.markdown("---")
    st.markdown("### 📈 Analysis")
    st.write("Compare this result to previous trends using your dataset or visualize it below.")

    st.bar_chart(pd.DataFrame({
        "Predicted Impact": [prediction],
        "Sector Growth": [avg_sector_growth],
        "AI Role Jobs": [avg_ai_role_jobs]
    }))

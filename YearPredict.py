import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

model = joblib.load("xgboost_model.pkl")

# Page config
st.set_page_config(
    page_title="AI & Automation Impact Predictor",
    page_icon="🤖",
    layout="wide"
)

# Custom styling
st.markdown(
    """
    <style>
        .main-title {
            font-size: 48px;
            font-weight: 700;
            color: #4A90E2;
        }
        .sub-title {
            font-size: 20px;
            color: #555;
            margin-top: -15px;
        }
        .stButton>button {
            background-color: #4A90E2;
            color: white;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title + image
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=100)

with col2:
    st.markdown('<p class="main-title">AI & Automation Impact Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Analyze how AI & automation are shaping the future of work across sectors</p>', unsafe_allow_html=True)

st.markdown("---")

# Load CSV data
try:
    df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
except Exception as e:
    st.error("Error loading data. Please ensure the CSV file is in the directory.")
    st.stop()

# Flatten '_id' dictionary if needed
if '_id' in df.columns and isinstance(df['_id'].iloc[0], str) == False:
    df['_id.Country'] = df['_id'].apply(lambda x: x.get('Country'))
    df['_id.Sector'] = df['_id'].apply(lambda x: x.get('Sector'))
    df['_id.Year'] = df['_id'].apply(lambda x: x.get('Year'))
    df['_id.EducationLevel'] = df['_id'].apply(lambda x: x.get('EducationLevel'))

# Load model
try:
    model = joblib.load("models/xgboost_automation_model.pkl")
except Exception as e:
    st.error("Model file not found. Please ensure 'xgboost_automation_model.pkl' is in the 'models' folder.")
    st.stop()

# Input section
with st.form("prediction_form"):
    st.subheader("🔍 Input Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        country = st.selectbox("Country", df['_id.Country'].dropna().unique())
        year = st.selectbox("Year", sorted(df['_id.Year'].dropna().unique()))

    with col2:
        sector = st.selectbox("Sector", df['_id.Sector'].dropna().unique())
        education_level = st.selectbox("Education Level", df['_id.EducationLevel'].dropna().unique())

    with col3:
        avg_pre_ai = st.number_input("Avg_PreAI", min_value=0.0)
        avg_post_ai = st.number_input("Avg_PostAI", min_value=0.0)
        avg_automation_impact = st.number_input("Avg_Automation_Impact", min_value=0.0)
        avg_ai_role_jobs = st.number_input("Avg_AI_Role_Jobs", min_value=0.0)
        avg_reskilling_programs = st.number_input("Avg_ReskillingPrograms", min_value=0.0)
        avg_economic_impact = st.number_input("Avg_EconomicImpact", min_value=0.0)
        avg_sector_growth = st.number_input("Avg_SectorGrowth", min_value=0.0)

    submit = st.form_submit_button("🚀 Predict Impact")

# Prediction
if submit:
    input_df = pd.DataFrame({
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

    # Drop non-numeric columns
    numeric_input = input_df.select_dtypes(include=[np.number])

    prediction = model.predict(numeric_input)[0]

    st.success(f"📊 Predicted Impact Score: **{prediction:.2f}**")

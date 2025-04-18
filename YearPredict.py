import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image
import json
import requests

# Optional: Lottie animation (make sure to install streamlit-lottie)
try:
    from streamlit_lottie import st_lottie
    def load_lottie_url(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_ai = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json")  # AI animation
except ImportError:
    lottie_ai = None

# ---------- Layout Setup ----------
st.set_page_config(page_title="AI Impact Predictor", layout="wide")

# Sidebar with logo or Lottie animation
with st.sidebar:
    if lottie_ai:
        st_lottie(lottie_ai, height=200)
    else:
        image = Image.open("ai_icon.png")
        st.image(image, use_column_width=True)

    st.title("⚙️ Predict Automation Impact")
    st.markdown("Explore how AI & automation influence employment trends.")

# ---------- Load Data ----------
try:
    data = pd.read_csv("data/automation_data.csv")
except FileNotFoundError:
    st.error("CSV file not found. Please make sure it's uploaded and path is correct.")
    st.stop()

# Flatten '_id' if it's from MongoDB
if '_id' in data.columns and isinstance(data['_id'].iloc[0], str) == False:
    _id_df = pd.json_normalize(data['_id'])
    data = data.drop(columns=['_id']).join(_id_df)

# Display columns for debug
# st.write("Available columns:", data.columns.tolist())

# ---------- Filters ----------
st.markdown("### 🔍 Select Filters Below")

col1, col2, col3 = st.columns(3)

with col1:
    countries = sorted(data['Country'].unique())
    selected_country = st.selectbox("🌍 Country", countries)

with col2:
    sectors = sorted(data['Sector'].unique())
    selected_sector = st.selectbox("🏭 Sector", sectors)

with col3:
    education_levels = sorted(data['EducationLevel'].unique())
    selected_education = st.selectbox("🎓 Education Level", education_levels)

# Slider for year
years = sorted(data['Year'].unique())
selected_year = st.slider("📅 Year", min_value=int(min(years)), max_value=int(max(years)), step=1)

# ---------- Filter Data ----------
filtered_data = data[
    (data['Country'] == selected_country) &
    (data['Sector'] == selected_sector) &
    (data['EducationLevel'] == selected_education)
]

if filtered_data.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

# ---------- Predict Button ----------
st.markdown("### 📊 Predicted Impact of Automation & AI")

model_features = [
    'Avg_PreAI',
    'Avg_PostAI',
    'Avg_Automation_Impact',
    'Avg_AI_Role_Jobs',
    'Avg_ReskillingPrograms',
    'Avg_EconomicImpact',
    'Avg_SectorGrowth'
]

input_row = filtered_data[filtered_data['Year'] == selected_year]

if input_row.empty:
    st.warning("No data found for the selected year. Please try a different year.")
    st.stop()

X_input = input_row[model_features]

# Load model
try:
    model = joblib.load("models/xgboost_automation_model.pkl")
    predicted_impact = model.predict(X_input)[0]
    st.success(f"📈 Predicted Automation Impact Score for {selected_year}: **{predicted_impact:.2f}**")
except FileNotFoundError:
    st.error("Model file not found. Please upload `xgboost_automation_model.pkl` in `models/` directory.")
    st.stop()

# ---------- Line Chart ----------
st.markdown("### 📈 Automation Impact Trend Over Time")

trend_data = filtered_data.sort_values(by='Year')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(trend_data['Year'], trend_

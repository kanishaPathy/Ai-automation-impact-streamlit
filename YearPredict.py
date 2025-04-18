import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# Load your data
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

# Set page config
st.set_page_config(page_title="AI & Automation Impact", layout="wide")

# --- HEADER SECTION ---
st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='font-size: 48px; color: #1f77b4;'>🌐 AI & Automation Impact Explorer</h1>
        <p style='font-size: 20px; color: gray;'>Visualize and predict the automation effect by Country, Sector, and Year</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- SIDEBAR FILTERS ---
st.sidebar.header("🔍 Filter Criteria")

# Dropdowns & sliders
years = sorted(data['_id.Year'].unique())
selected_year_range = st.sidebar.slider("Select Year Range", int(min(years)), int(max(years)), (2010, 2022))

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    sorted(data['_id.Country'].unique()),
    default=["USA", "IND"]
)

selected_sector = st.sidebar.selectbox(
    "Select Sector",
    sorted(data['_id.Sector'].unique())
)

selected_edu = st.sidebar.selectbox(
    "Select Education Level",
    sorted(data['_id.EducationLevel'].unique())
)

# --- FILTER DATA ---
filtered_data = data[
    (data['_id.Year'] >= selected_year_range[0]) &
    (data['_id.Year'] <= selected_year_range[1]) &
    (data['_id.Country'].isin(selected_countries)) &
    (data['_id.Sector'] == selected_sector) &
    (data['_id.EducationLevel'] == selected_edu)
]

# --- MAIN VIEW ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Automation Impact Over Time")
    fig = px.line(
        filtered_data,
        x="_id.Year",
        y="Avg_Automation_Impact",
        color="_id.Country",
        markers=True,
        labels={"_id.Year": "Year", "Avg_Automation_Impact": "Automation Impact"},
        title=""
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💼 AI Role Jobs Trend")
    fig2 = px.line(
        filtered_data,
        x="_id.Year",
        y="Avg_AI_Role_Jobs",
        color="_id.Country",
        markers=True,
        labels={"_id.Year": "Year", "Avg_AI_Role_Jobs": "AI Jobs Created"},
        title=""
    )
    fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, use_container_width=True)

# --- PREDICTION SECTION ---
st.markdown("---")
st.header("🤖 Predict Automation Impact")

input_country = st.selectbox("Country", sorted(data['_id.Country'].unique()))
input_sector = st.selectbox("Sector", sorted(data['_id.Sector'].unique()))
input_year = st.slider("Year", min(years), max(years), 2022)
input_edu = st.selectbox("Education Level", sorted(data['_id.EducationLevel'].unique()))

# Dummy input fields (you can use actual means or sliders)
avg_pre_ai = st.number_input("Avg_PreAI", value=50.0)
avg_post_ai = st.number_input("Avg_PostAI", value=65.0)
avg_ai_jobs = st.number_input("Avg_AI_Role_Jobs", value=2000.0)
avg_reskill = st.number_input("Avg_ReskillingPrograms", value=30.0)
avg_eco = st.number_input("Avg_EconomicImpact", value=10.0)
avg_growth = st.number_input("Avg_SectorGrowth", value=5.0)

# CTA button
if st.button("Predict Automation Impact 🚀"):
    model_path = "models/xgboost_automation_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        input_df = pd.DataFrame({
            '_id.Country': [input_country],
            '_id.Sector': [input_sector],
            '_id.Year': [input_year],
            '_id.EducationLevel': [input_edu],
            'Avg_PreAI': [avg_pre_ai],
            'Avg_PostAI': [avg_post_ai],
            'Avg_AI_Role_Jobs': [avg_ai_jobs],
            'Avg_ReskillingPrograms': [avg_reskill],
            'Avg_EconomicImpact': [avg_eco],
            'Avg_SectorGrowth': [avg_growth]
        })

        prediction = model.predict(input_df.drop(columns=['_id.Country', '_id.Sector', '_id.Year', '_id.EducationLevel']))
        st.success(f"🔮 Predicted Automation Impact: **{prediction[0]:.2f}**")
    else:
        st.error("❌ Model file not found. Please make sure the model exists in the 'models/' directory.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>© 2025 AI Employment Impact Dashboard</p>",
    unsafe_allow_html=True
)

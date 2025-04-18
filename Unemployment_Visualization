import streamlit as st
import pandas as pd
from PIL import Image

# ---------------------- CONFIGURATION ----------------------
st.set_page_config(page_title="AI & Automation Impact Explorer", layout="wide")

# Load an image for the header (replace with your own)
image = Image.open("ai_icon.png")  # Place a tech-related image in your directory

# ---------------------- HEADER ----------------------
col_img, col_title = st.columns([1, 5])
with col_img:
    st.image(image, width=100)

with col_title:
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-size: 3em; color: #2c3e50; margin-bottom: 0;'>AI & Automation Impact Explorer</h1>
            <p style='font-size: 1.3em; color: #7f8c8d; margin-top: 5px;'>Predict, Compare, and Visualize the Future of Jobs in a Tech-Driven World</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------------- INPUT FORM ----------------------

with st.form("input_form"):
    st.subheader("📊 Enter Prediction Parameters")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        country = st.selectbox("🌍 Country", ["USA", "India", "Ireland", "Germany", "Canada"])
    with col2:
        sector = st.selectbox("🏢 Sector", ["IT", "Healthcare", "Manufacturing", "Finance", "Education"])
    with col3:
        year = st.slider("📅 Year", 2010, 2024, 2022)
    with col4:
        education_level = st.selectbox("🎓 Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

    st.markdown("### ")

    col5, col6, col7 = st.columns(3)
    with col5:
        avg_pre_ai = st.number_input("Avg_PreAI", min_value=0.0, step=0.1)
        avg_automation_impact = st.number_input("Avg_Automation_Impact", min_value=0.0, step=0.1)
    with col6:
        avg_post_ai = st.number_input("Avg_PostAI", min_value=0.0, step=0.1)
        avg_ai_role_jobs = st.number_input("Avg_AI_Role_Jobs", min_value=0.0, step=0.1)
    with col7:
        avg_reskilling_programs = st.number_input("Avg_ReskillingPrograms", min_value=0.0, step=0.1)
        avg_economic_impact = st.number_input("Avg_EconomicImpact", min_value=0.0, step=0.1)

    avg_sector_growth = st.number_input("Avg_SectorGrowth", min_value=0.0, step=0.1)

    submit_button = st.form_submit_button("🔍 Predict Impact")

# ---------------------- PREDICTION LOGIC ----------------------

if submit_button:
    st.success("✅ Prediction Started!")

    # Construct input data as a dictionary
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

    # Show input data preview
    st.markdown("### 📥 Input Data Preview")
    st.dataframe(df)

    # ------------------ 🔮 PLACEHOLDER FOR MODEL PREDICTION ------------------
    # Example: prediction = model.predict(df) 
    # You can load your XGBoost or other trained model here and use this dataframe

    st.markdown("### 📈 Prediction & Visualization (Coming Next...)")
    st.info("This is where you’ll plug in your trained model and visualizations.")

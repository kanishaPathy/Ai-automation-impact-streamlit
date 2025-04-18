import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import plotly.express as px
import joblib

# --- Load Model & Data ---
model = XGBRegressor()
model.load_model("your_model.json")  # Replace with your model path

df = pd.read_csv("your_dataset.csv")  # Replace with your CSV file path

# --- Set Streamlit Page Config ---
st.set_page_config(page_title="📊 Yearly Impact Prediction", layout="wide")

st.title("📈 Yearly Impact of AI & Automation")

st.markdown("Use the dropdowns below to select the parameters and predict automation impact for a specific year.")

# --- Sidebar Filters ---
with st.form("prediction_form"):
    st.subheader("🎯 Select Input Parameters")

    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("📅 Year", sorted(df['_id.Year'].unique()))
        sector = st.selectbox("🏭 Sector", sorted(df['_id.Sector'].unique()))
    with col2:
        country = st.selectbox("🌍 Country", sorted(df['_id.Country'].unique()))
        education = st.selectbox("🎓 Education Level", sorted(df['_id.EducationLevel'].unique()))

    submitted = st.form_submit_button("🔍 Predict Impact")

# --- Prediction Logic ---
if submitted:
    try:
        input_df = pd.DataFrame([{
            "_id.Year": year,
            "_id.Country": country,
            "_id.Sector": sector,
            "_id.EducationLevel": education
        }])

        # Encode categorical columns (using training label encoders if available)
        label_encoders = joblib.load("label_encoders.pkl")  # Make sure you saved label encoders during training

        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        prediction = model.predict(input_df)[0]

        st.success(f"📌 Predicted Automation Impact for {sector} in {country} ({year}, {education}): `{prediction:.2f}`")

        # Optional: Show matching records
        filtered = df[
            (df['_id.Year'] == year) &
            (df['_id.Country'] == country) &
            (df['_id.Sector'] == sector) &
            (df['_id.EducationLevel'] == education)
        ]

        if not filtered.empty:
            st.subheader("📂 Matching Data Sample")
            st.dataframe(filtered)

        # Optional Plot
        st.subheader("📊 Country-wise Sector Comparison")
        chart_df = df[df['_id.Year'] == year]
        fig = px.bar(
            chart_df,
            x="_id.Sector",
            y="Automation_Impact",
            color="_id.Country",
            barmode="group",
            title="Automation Impact by Sector and Country"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Something went wrong: {str(e)}")

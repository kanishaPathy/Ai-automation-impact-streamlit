import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# Load model and preprocessors
model = XGBRegressor()
model.load_model("xgb_model.json")

label_encoders = joblib.load("label_encoders.pkl")
expected_features = joblib.load("model_features.pkl")

# Sample data to test with (or load your actual cleaned dataset)
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")  # Make sure it contains _id.Year, _id.Country, etc.

st.set_page_config(page_title="📅 Year Prediction", layout="wide")
st.title("📅 Predict Automation Impact for a Given Year")

st.markdown("### Select Input Parameters")

# UI layout
col1, col2, col3, col4 = st.columns(4)

year = col1.selectbox("Select Year", sorted(df['_id.Year'].unique()))
country = col2.selectbox("Select Country", sorted(df['_id.Country'].unique()))
sector = col3.selectbox("Select Sector", sorted(df['_id.Sector'].unique()))
education = col4.selectbox("Select Education Level", sorted(df['_id.EducationLevel'].unique()))

st.markdown("---")

# Submit button
if st.button("🚀 Predict Automation Impact"):

    st.write(f"Analyzing impact for **{sector}** in **{country}** for year **{year}** with education level **{education}**.")

    # Prepare input data
    input_data = {
        "_id.Year": [year],
        "_id.Country": [country],
        "_id.Sector": [sector],
        "_id.EducationLevel": [education]
    }

    input_df = pd.DataFrame(input_data)

    # Apply label encoders
    for col in input_df.columns:
        if col in label_encoders:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError as e:
                st.error(f"Error: {e}")
                st.stop()

    # Reorder columns to match training
    try:
        input_df = input_df[expected_features]
    except KeyError as e:
        st.error(f"Column mismatch: {e}")
        st.stop()

    # Make prediction
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💡 Predicted Automation Impact: **{prediction:.2f}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

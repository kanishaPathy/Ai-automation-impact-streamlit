
import pandas as pd
import joblib
import streamlit as st

# Load trained XGBoost model
model = joblib.load(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\xgboost_model.pkl")

st.title("AI Automation Impact Prediction")

# Inputs
year = st.selectbox("Select Year", [2021, 2022, 2023])
country = st.selectbox("Select Country", ['Canada', 'Germany', 'India', 'Ireland', 'USA'])
sector = st.selectbox("Select Sector", ['Agriculture', 'Education', 'Finance', 'Healthcare', 'IT',
                                        'Manufacturing', 'Media & Entertainment', 'Retail', 'Transportation'])
education_level = st.selectbox("Select Education Level", ['High School', 'Bachelor', 'Master', 'PhD', 'Unknown'])
avg_pre_ai = st.slider("Average Pre-AI Impact", 0, 100, 50)
avg_post_ai = st.slider("Average Post-AI Impact", 0, 100, 50)
avg_automation_impact = st.slider("Average Automation Impact", 0, 100, 50)
avg_ai_role_jobs = st.slider("Average AI Role Jobs", 0, 100, 50)
avg_reskilling_programs = st.slider("Average Reskilling Programs", 0, 100, 50)
avg_economic_impact = st.slider("Average Economic Impact", 0, 100, 50)
avg_sector_growth = st.slider("Average Sector Growth", 0, 100, 50)

# Create DataFrame from inputs
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

# One-hot encode input
input_encoded = pd.get_dummies(input_df)

# Load training data to align columns
training_df = pd.read_csv(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\Unemployment_jobcreation_db.Unemployment_data.csv")
training_encoded = pd.get_dummies(training_df.drop(columns=['Avg_Automation_Impact']))

# Add missing columns to input
missing_cols = set(training_encoded.columns) - set(input_encoded.columns)
for col in missing_cols:
    input_encoded[col] = 0

# Reorder input columns
input_encoded = input_encoded[training_encoded.columns]

# Predict
prediction = model.predict(input_encoded)

# Show result
st.success(f"Predicted Automation Impact: {prediction[0]:.2f}")

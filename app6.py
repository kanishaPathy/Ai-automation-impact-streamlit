import joblib
import pandas as pd
import streamlit as st

# Load the model
model = joblib.load("xgboost_model.pkl")

# Create a function to make the prediction
def make_prediction(year, avg_pre_ai, avg_post_ai, avg_ai_role_jobs, avg_reskilling_programs, avg_economic_impact, avg_sector_growth,
                     country, sector, education_level):
    # Create a DataFrame for the input
    input_df = pd.DataFrame({
        "_id.Year": [year],
        "Avg_PreAI": [avg_pre_ai],
        "Avg_PostAI": [avg_post_ai],
        "Avg_AI_Role_Jobs": [avg_ai_role_jobs],
        "Avg_ReskillingPrograms": [avg_reskilling_programs],
        "Avg_EconomicImpact": [avg_economic_impact],
        "Avg_SectorGrowth": [avg_sector_growth],
        "_id.Country_Canada": [1 if country == 'Canada' else 0],
        "_id.Country_Germany": [1 if country == 'Germany' else 0],
        "_id.Country_India": [1 if country == 'India' else 0],
        "_id.Country_Ireland": [1 if country == 'Ireland' else 0],
        "_id.Country_USA": [1 if country == 'USA' else 0],
        "_id.Sector_Agriculture": [1 if sector == 'Agriculture' else 0],
        "_id.Sector_Education": [1 if sector == 'Education' else 0],
        "_id.Sector_Finance": [1 if sector == 'Finance' else 0],
        "_id.Sector_Healthcare": [1 if sector == 'Healthcare' else 0],
        "_id.Sector_IT": [1 if sector == 'IT' else 0],
        "_id.Sector_Manufacturing": [1 if sector == 'Manufacturing' else 0],
        "_id.Sector_Media & Entertainment": [1 if sector == 'Media & Entertainment' else 0],
        "_id.Sector_Retail": [1 if sector == 'Retail' else 0],
        "_id.Sector_Transportation": [1 if sector == 'Transportation' else 0],
        "_id.EducationLevel_Bachelor": [1 if education_level == 'Bachelor' else 0],
        "_id.EducationLevel_High School": [1 if education_level == 'High School' else 0],
        "_id.EducationLevel_Master": [1 if education_level == 'Master' else 0],
        "_id.EducationLevel_PhD": [1 if education_level == 'PhD' else 0],
        "_id.EducationLevel_Unknown": [1 if education_level == 'Unknown' else 0]
    })

    # Reorder the columns to match the model's feature names
    input_df = input_df[model.get_booster().feature_names]

    # Make the prediction
    prediction = model.predict(input_df)

    return prediction

# Streamlit inputs for the user
st.title("AI & Automation Impact Prediction")

year = st.number_input("Year", min_value=2010, max_value=2024)
avg_pre_ai = st.number_input("Avg Pre-AI")
avg_post_ai = st.number_input("Avg Post-AI")
avg_ai_role_jobs = st.number_input("Avg AI Role Jobs")
avg_reskilling_programs = st.number_input("Avg Reskilling Programs")
avg_economic_impact = st.number_input("Avg Economic Impact")
avg_sector_growth = st.number_input("Avg Sector Growth")

country = st.selectbox("Select Country", ['Canada', 'Germany', 'India', 'Ireland', 'USA'])
sector = st.selectbox("Select Sector", ['Agriculture', 'Education', 'Finance', 'Healthcare', 'IT', 'Manufacturing', 'Media & Entertainment', 'Retail', 'Transportation'])
education_level = st.selectbox("Select Education Level", ['Bachelor', 'High School', 'Master', 'PhD', 'Unknown'])

if st.button("Make Prediction"):
    prediction = make_prediction(year, avg_pre_ai, avg_post_ai, avg_ai_role_jobs, avg_reskilling_programs, avg_economic_impact, avg_sector_growth, country, sector, education_level)
    st.write(f"Prediction: {prediction}")

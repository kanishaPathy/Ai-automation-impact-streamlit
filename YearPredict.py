import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load model and data
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

# Sample data (replace with your actual dataset)
df = pd.DataFrame({
    '_id.Country': ['USA', 'India', 'Ireland'],
    '_id.Sector': ['IT', 'Retail', 'Finance'],
    '_id.EducationLevel': ['Bachelors', 'Masters', 'PhD'],
    '_id.Year': [2010, 2015, 2020],
    'PreAI': [5, 6, 7],
    'PostAI': [6, 7, 8],
    'Automation_Impact': [0.3, 0.4, 0.5],
    'AI_Role_Jobs': [100, 150, 120],
    'Reskilling_Programs': [20, 30, 25],
    'Economic_Impact': [500, 600, 550]
})

# Sidebar for user inputs
st.sidebar.title("🌍 Country Comparison")
year = st.sidebar.selectbox("📅 Select Year", sorted(df['_id.Year'].unique()))
country = st.sidebar.selectbox("🌍 Select Country", sorted(df['_id.Country'].unique()))
sector = st.sidebar.selectbox("🏢 Select Sector", sorted(df['_id.Sector'].unique()))
education = st.sidebar.selectbox("🎓 Select Education Level", sorted(df['_id.EducationLevel'].unique()))

submit_button = st.sidebar.button("Submit")

if submit_button:
    # Label Encoding
    label_encoder_country = LabelEncoder()
    label_encoder_sector = LabelEncoder()
    label_encoder_education = LabelEncoder()

    df['_id.Country'] = label_encoder_country.fit_transform(df['_id.Country'])
    df['_id.Sector'] = label_encoder_sector.fit_transform(df['_id.Sector'])
    df['_id.EducationLevel'] = label_encoder_education.fit_transform(df['_id.EducationLevel'])

    # Preparing input data
    input_data = {
        "_id.Year": year,
        "_id.Country": label_encoder_country.transform([country])[0],
        "_id.Sector": label_encoder_sector.transform([sector])[0],
        "_id.EducationLevel": label_encoder_education.transform([education])[0],
        "PreAI": df.loc[df['_id.Year'] == year, 'PreAI'].values[0],
        "PostAI": df.loc[df['_id.Year'] == year, 'PostAI'].values[0],
        "Automation_Impact": df.loc[df['_id.Year'] == year, 'Automation_Impact'].values[0],
        "AI_Role_Jobs": df.loc[df['_id.Year'] == year, 'AI_Role_Jobs'].values[0],
        "Reskilling_Programs": df.loc[df['_id.Year'] == year, 'Reskilling_Programs'].values[0],
        "Economic_Impact": df.loc[df['_id.Year'] == year, 'Economic_Impact'].values[0]
    }

    input_df = pd.DataFrame([input_data])

    # Predict automation impact
    prediction = model.predict(input_df)[0]

    # Visualization
    fig = px.bar(df, x='_id.Sector', y='Automation_Impact', color='_id.Country',
                 labels={'_id.Sector': 'Sector', 'Automation_Impact': 'Impact of Automation'},
                 title=f"Impact of Automation by Sector in {year}")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"### Selected Inputs:")
        st.write(f"**Year**: {year}")
        st.write(f"**Country**: {country}")
        st.write(f"**Sector**: {sector}")
        st.write(f"**Education Level**: {education}")

    with col2:
        st.write("### Impact Visualization:")
        st.plotly_chart(fig)

    st.write(f"### Predicted Automation Impact for {country} in {year}")
    st.write(f"**Impact Level**: {prediction:.2f}")

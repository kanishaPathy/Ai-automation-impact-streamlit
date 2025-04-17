
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Load the trained XGBoost model
model = joblib.load(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\xgboost_model.pkl")

# Set the Streamlit title
st.title("AI Automation Impact Prediction")

# Sidebar for User Inputs
st.sidebar.header("Enter the following details for Prediction")

year = st.sidebar.selectbox("Select Year", list(range(2010, 2026)))
country = st.sidebar.selectbox("Select Country", ['Canada', 'Germany', 'India', 'Ireland', 'USA'])
sector = st.sidebar.selectbox("Select Sector", ['Agriculture', 'Education', 'Finance', 'Healthcare', 'IT',
                                               'Manufacturing', 'Media & Entertainment', 'Retail', 'Transportation'])
education_level = st.sidebar.selectbox("Select Education Level", ['High School', 'Bachelor', 'Master', 'PhD', 'Unknown'])

avg_pre_ai = st.sidebar.slider("Average Pre-AI Impact", 0, 100, 50)
avg_post_ai = st.sidebar.slider("Average Post-AI Impact", 0, 100, 50)
avg_automation_impact = st.sidebar.slider("Average Automation Impact", 0, 100, 50)
avg_ai_role_jobs = st.sidebar.slider("Average AI Role Jobs", 0, 100, 50)
avg_reskilling_programs = st.sidebar.slider("Average Reskilling Programs", 0, 100, 50)
avg_economic_impact = st.sidebar.slider("Average Economic Impact", 0, 100, 50)
avg_sector_growth = st.sidebar.slider("Average Sector Growth", 0, 100, 50)

# Create DataFrame for input values
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

# Load training data and encode
training_df = pd.read_csv(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\Unemployment_jobcreation_db.Unemployment_data.csv")
X_train = training_df.drop(columns=['Avg_Automation_Impact'])
X_train_encoded = pd.get_dummies(X_train)

# Encode user input
input_encoded = pd.get_dummies(input_df)

# Add missing columns to input
for col in X_train_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Ensure correct column order
input_encoded = input_encoded[X_train_encoded.columns]

# Prediction
prediction = model.predict(input_encoded)
st.write("Predicted Automation Impact:", prediction[0])

# Country-wise Pre-AI vs Post-AI comparison plot
st.subheader("Country-wise Pre-AI vs Post-AI Comparison")
country_comparison_df = training_df.groupby(['_id.Country']).agg({'Avg_PreAI': 'mean', 'Avg_PostAI': 'mean'}).reset_index()

fig = plt.figure(figsize=(6, 4))
sns.barplot(x='Avg_PreAI', y='_id.Country', data=country_comparison_df, label='Pre-AI Impact', color='skyblue')
sns.barplot(x='Avg_PostAI', y='_id.Country', data=country_comparison_df, label='Post-AI Impact', color='orange')
plt.title('Country-wise Pre-AI vs Post-AI Impact')
plt.legend()
st.pyplot(fig)

# Unemployment trend (Pre-AI vs Post-AI) from 2010 to 2025
st.subheader("Unemployment Trend (Pre-AI vs Post-AI) Over Time")
unemployment_df = training_df.groupby(['_id.Year']).agg({'Avg_PreAI': 'mean', 'Avg_PostAI': 'mean'}).reset_index()

fig = plt.figure(figsize=(6, 4))
plt.plot(unemployment_df['_id.Year'], unemployment_df['Avg_PreAI'], label='Pre-AI Impact', color='skyblue', marker='o')
plt.plot(unemployment_df['_id.Year'], unemployment_df['Avg_PostAI'], label='Post-AI Impact', color='orange', marker='o')
plt.title('Unemployment Impact Over Time (Pre-AI vs Post-AI)')
plt.xlabel('Year')
plt.ylabel('Impact')
plt.legend()
st.pyplot(fig)

# Education level impact on AI (Pre-AI and Post-AI)
st.subheader("Education Level Impact on AI")
education_comparison_df = training_df.groupby(['_id.EducationLevel']).agg({'Avg_PreAI': 'mean', 'Avg_PostAI': 'mean'}).reset_index()

fig = plt.figure(figsize=(6, 4))
sns.barplot(x='Avg_PreAI', y='_id.EducationLevel', data=education_comparison_df, label='Pre-AI Impact', color='skyblue')
sns.barplot(x='Avg_PostAI', y='_id.EducationLevel', data=education_comparison_df, label='Post-AI Impact', color='orange')
plt.title('Education Level Impact on AI (Pre-AI vs Post-AI)')
plt.legend()
st.pyplot(fig)

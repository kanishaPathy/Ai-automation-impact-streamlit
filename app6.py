import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the model
model = joblib.load("xgboost_model.pkl")

# Load the dataset
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

# Clean column names for easy access (optional, but makes your life easier)
df.rename(columns=lambda x: x.replace("_id.", ""), inplace=True)

# Title
st.title("AI & Automation Impact Prediction App")

# Sidebar selections
st.sidebar.header("User Input Features")
country = st.sidebar.selectbox("Select Country", df['Country'].unique())
sector = st.sidebar.selectbox("Select Sector", df['Sector'].unique())
year = st.sidebar.selectbox("Select Year", sorted(df['Year'].unique()))
education_level = st.sidebar.selectbox("Select Education Level", df['EducationLevel'].unique())

# Filter data for selected options (optional)
filtered_df = df[
    (df['Country'] == country) &
    (df['Sector'] == sector) &
    (df['Year'] == year) &
    (df['EducationLevel'] == education_level)
]

st.subheader("Filtered Data Preview")
st.write(filtered_df)

# Feature inputs
st.subheader("Enter Additional Features")
ai_adoption = st.slider("AI Adoption Rate", 0.0, 1.0, 0.5)
automation_impact = st.slider("Automation Impact", 0.0, 1.0, 0.5)
reskilling_programs = st.slider("Reskilling Programs (0–1)", 0.0, 1.0, 0.5)

# Combine features into a single dataframe for prediction
input_df = pd.DataFrame({
    'AI_Adoption_Rate': [ai_adoption],
    'Automation_Impact': [automation_impact],
    'ReskillingPrograms': [reskilling_programs]
})

# Make prediction
if st.button("Predict Unemployment Impact"):
    prediction = model.predict(input_df)
    st.success(f"📊 Predicted Unemployment Impact: **{prediction[0]:.2f}**")

# Visualization (Optional)
st.subheader("Unemployment Before vs After AI")
if not filtered_df.empty:
    fig = px.bar(filtered_df, x='Year', y=['PreAI', 'PostAI'], barmode='group', title=f"Impact in {sector} - {country}")
    st.plotly_chart(fig)

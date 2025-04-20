import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 AI & Automation Impact Predictor")

df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")
df3 = pd.read_csv("Sectors_Growth_AI_Adoption_dirty_100k.csv")

# Check column names
st.write("📄 df1 columns:", df1.columns.tolist())
st.write("📄 df2 columns:", df2.columns.tolist())
st.write("📄 df3 columns:", df3.columns.tolist())

# Rename if necessary
if "Country_x" in df1.columns or "_id.Country" in df1.columns:
    df1.rename(columns={"_id.Country": "Country"}, inplace=True)
if "_id.Sector" in df1.columns:
    df1.rename(columns={"_id.Sector": "Sector"}, inplace=True)
if "_id.EducationLevel" in df1.columns:
    df1.rename(columns={"_id.EducationLevel": "EducationLevel"}, inplace=True)
if "_id.Year" in df1.columns:
    df1.rename(columns={"_id.Year": "Year"}, inplace=True)


# Merge the datasets
merged = df1.merge(df2, on=["Country", "Sector", "Year", "EducationLevel"], how="left")
merged = merged.merge(df3, on=["Country", "Sector", "Year"], how="left")

# Fill missing values
merged.fillna(0, inplace=True)

# Load XGBoost model
model = joblib.load("xgboost_model.pkl")

# One-hot encoding
encoded = pd.get_dummies(merged, columns=["Country", "Sector", "EducationLevel"], prefix=["_id.Country", "_id.Sector", "_id.EducationLevel"])

# Ensure same columns as training
model_features = model.feature_names_in_
for col in model_features:
    if col not in encoded.columns:
        encoded[col] = 0

# Align column order
encoded = encoded[model_features]

# Fix non-numeric issue
merged["Automation_Impact_Level"] = pd.to_numeric(merged["Automation_Impact_Level"], errors="coerce")
merged["AI_Adoption_Rate"] = pd.to_numeric(merged["AI_Adoption_Rate"], errors="coerce")

# Predict and add to merged DataFrame
merged["Predicted_Impact"] = model.predict(encoded)

# Sidebar selections
with st.sidebar:
    st.header("🔍 Filter")
    country = st.selectbox("Select Country", merged["Country"].unique())
    sector = st.selectbox("Select Sector", merged["Sector"].unique())
    year = st.slider("Select Year", int(merged["Year"].min()), int(merged["Year"].max()))

# Filter data
filtered = merged[(merged["Country"] == country) & 
                  (merged["Sector"] == sector) & 
                  (merged["Year"] == year)]

# Plot
st.subheader(f"Impact in {country} - {sector} ({year})")
fig = px.bar(filtered,
             x="EducationLevel",
             y=["Avg_Automation_Impact", "Automation_Impact_Level", "Predicted_Impact"],
             barmode="group",
             title="AI & Automation Impact by Education Level")
st.plotly_chart(fig, use_container_width=True)

# Table view
st.subheader("📋 Data Preview")
st.dataframe(filtered[["Country", "Sector", "Year", "EducationLevel", 
                      "Avg_Automation_Impact", "Automation_Impact_Level", "Predicted_Impact"]])

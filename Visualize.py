import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load datasets
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")
df3 = pd.read_csv("Sectors_Growth_AI_Adoption_dirty_100k.csv")

# Merge all 3 datasets on Country, Sector, Year, and EducationLevel
merged_df = df1.merge(df2, on=["Country", "Sector", "Year", "EducationLevel"], how="left")
merged_df = merged_df.merge(df3, on=["Country", "Sector", "Year"], how="left")

# Replace missing values
merged_df.fillna(0, inplace=True)

# Load trained model
model = joblib.load("xgboost_model.pkl")

# Get model expected features
model_features = model.feature_names_in_

# One-hot encode the required columns
encoded_df = pd.get_dummies(merged_df, columns=['Country', 'Sector', 'EducationLevel'], prefix=['_id.Country', '_id.Sector', '_id.EducationLevel'])

# Add any missing columns expected by the model
for col in model_features:
    if col not in encoded_df.columns:
        encoded_df[col] = 0

# Ensure column order matches the model
encoded_df = encoded_df[model_features]

# Perform prediction
merged_df['Predicted_Impact'] = model.predict(encoded_df)

# Convert object columns to numeric if needed for visualization
merged_df["Automation_Impact_Level"] = pd.to_numeric(merged_df["Automation_Impact_Level"], errors='coerce')
merged_df["AI_Adoption_Rate"] = pd.to_numeric(merged_df["AI_Adoption_Rate"], errors='coerce')

# Streamlit UI
st.title("📈 AI & Automation Impact: Dataset Comparison")

selected_country = st.selectbox("Select Country", merged_df["Country"].unique())
selected_sector = st.selectbox("Select Sector", merged_df["Sector"].unique())
selected_year = st.slider("Select Year", int(merged_df["Year"].min()), int(merged_df["Year"].max()))

# Filter data
filtered = merged_df[
    (merged_df["Country"] == selected_country) &
    (merged_df["Sector"] == selected_sector) &
    (merged_df["Year"] == selected_year)
]

# Show bar plot
st.subheader("Impact Metrics")
fig = px.bar(filtered, x="EducationLevel",
             y=["Avg_Automation_Impact", "Automation_Impact_Level", "Predicted_Impact"],
             barmode="group", title=f"Automation Impact - {selected_country}, {selected_sector}, {selected_year}")
st.plotly_chart(fig)

# Show raw data
st.subheader("Predicted Data")
st.write(filtered[["Country", "Sector", "Year", "EducationLevel", "Predicted_Impact"]])

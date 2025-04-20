import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from xgboost import XGBRegressor

st.set_page_config(page_title="AI & Automation Impact Dashboard", layout="wide")

# 🔄 Load datasets
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")
df3 = pd.read_csv("Sectors_Growth_AI_Adoption_dirty_100k.csv")

# 🧹 Preprocess
for df in [df1, df2, df3]:
    for col in ['Country', 'Sector', 'EducationLevel']:
        if col in df.columns:
            df[col] = df[col].astype(str)

# 🧩 Merge datasets on common keys
df_merge1 = pd.merge(df1, df2, on=["Country", "Sector", "Year", "EducationLevel"], how="outer")
df_final = pd.merge(df_merge1, df3, on=["Country", "Sector", "Year", "EducationLevel"], how="outer")

# 🔍 Clean missing values
df_final.fillna(0, inplace=True)

# 🎯 Load Model
model = joblib.load("xgboost_model.pkl")

# 🎯 Prediction
feature_columns = ['Year', 'Avg_Unemployment_PreAI', 'Avg_Unemployment_PostAI',
                   'AI_Role_Jobs', 'Automation_Impact', 'ReskillingPrograms',
                   'Skills_Gap', 'Upskilling_Programs', 'Automation_Impact_Level',
                   'AI_Adoption_Rate', 'Sector_Growth_Index']

X = df_final[feature_columns]
df_final['Predicted_Impact'] = model.predict(X)

# 🎨 Title
st.title("🤖 AI & Automation Impact Comparison Dashboard")

# 🎛 Filters
with st.sidebar:
    selected_countries = st.multiselect("Select Country", options=df_final['Country'].unique(), default=df_final['Country'].unique())
    selected_sectors = st.multiselect("Select Sector", options=df_final['Sector'].unique(), default=df_final['Sector'].unique())
    selected_edu = st.multiselect("Select Education Level", options=df_final['EducationLevel'].unique(), default=df_final['EducationLevel'].unique())
    selected_years = st.slider("Select Year Range", int(df_final['Year'].min()), int(df_final['Year'].max()), (2010, 2024))

# 🧽 Apply filters
filtered_df = df_final[
    (df_final['Country'].isin(selected_countries)) &
    (df_final['Sector'].isin(selected_sectors)) &
    (df_final['EducationLevel'].isin(selected_edu)) &
    (df_final['Year'].between(selected_years[0], selected_years[1]))
]

# 📊 Visualization Blocks

st.subheader("🌍 Country-wise Actual vs Predicted Impact")
fig_country = px.bar(
    filtered_df.groupby('Country')[['Automation_Impact', 'Predicted_Impact']].mean().reset_index(),
    x='Country', y=['Automation_Impact', 'Predicted_Impact'],
    barmode='group',
    title='Actual vs Predicted Impact by Country'
)
st.plotly_chart(fig_country, use_container_width=True)

st.subheader("🏭 Sector-wise Actual vs Predicted Impact")
fig_sector = px.bar(
    filtered_df.groupby('Sector')[['Automation_Impact', 'Predicted_Impact']].mean().reset_index(),
    x='Sector', y=['Automation_Impact', 'Predicted_Impact'],
    barmode='group',
    title='Impact by Sector'
)
st.plotly_chart(fig_sector, use_container_width=True)

st.subheader("🎓 Education Level-wise Impact")
fig_edu = px.bar(
    filtered_df.groupby('EducationLevel')[['Automation_Impact', 'Predicted_Impact']].mean().reset_index(),
    x='EducationLevel', y=['Automation_Impact', 'Predicted_Impact'],
    barmode='group',
    title='Impact by Education Level'
)
st.plotly_chart(fig_edu, use_container_width=True)

st.subheader("📅 Year-wise Trend")
fig_year = px.line(
    filtered_df.groupby('Year')[['Automation_Impact', 'Predicted_Impact']].mean().reset_index(),
    x='Year', y=['Automation_Impact', 'Predicted_Impact'],
    title='Automation Impact Over Time'
)
st.plotly_chart(fig_year, use_container_width=True)

# 📋 Dataset Viewer
st.subheader("🧾 Final Merged Dataset (Filtered)")
st.dataframe(filtered_df.head(100))

# 📁 Download Option
st.download_button("Download Filtered Dataset", filtered_df.to_csv(index=False), "filtered_impact_data.csv")

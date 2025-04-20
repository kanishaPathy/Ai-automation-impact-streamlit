import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="AI Automation Impact: Full Comparison", layout="wide")

# 📥 Load datasets
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")
df3 = pd.read_csv("Sectors_Growth_AI_Adoption_dirty_100k.csv")

# 🔄 Convert key columns to string
for df in [df1, df2, df3]:
    for col in ['Country', 'Sector', 'EducationLevel']:
        if col in df.columns:
            df[col] = df[col].astype(str)

# 🔗 Merge datasets
df_merge1 = pd.merge(df1, df2, on=["Country", "Sector", "Year", "EducationLevel"], how="outer")
df_final = pd.merge(df_merge1, df3, on=["Country", "Sector", "Year", "EducationLevel"], how="outer")

# 🧹 Clean missing values
df_final.fillna(0, inplace=True)

# 📦 Load model
model = joblib.load("xgboost_model.pkl")

# 🧠 Predict
feature_columns = [
    'Year', 'Avg_Unemployment_PreAI', 'Avg_Unemployment_PostAI',
    'AI_Role_Jobs', 'Automation_Impact', 'ReskillingPrograms',
    'Skills_Gap', 'Upskilling_Programs', 'Automation_Impact_Level',
    'AI_Adoption_Rate', 'Sector_Growth_Index'
]
X = df_final[feature_columns]
df_final['Predicted_Impact'] = model.predict(X)

# 🧭 Sidebar filters
st.sidebar.header("🔍 Filters")
selected_countries = st.sidebar.multiselect("Select Country", df_final['Country'].unique(), default=df_final['Country'].unique())
selected_sectors = st.sidebar.multiselect("Select Sector", df_final['Sector'].unique(), default=df_final['Sector'].unique())
selected_edu = st.sidebar.multiselect("Select Education Level", df_final['EducationLevel'].unique(), default=df_final['EducationLevel'].unique())
selected_years = st.sidebar.slider("Select Year Range", int(df_final['Year'].min()), int(df_final['Year'].max()), (2010, 2024))

# 🧼 Filter data
filtered_df = df_final[
    (df_final['Country'].isin(selected_countries)) &
    (df_final['Sector'].isin(selected_sectors)) &
    (df_final['EducationLevel'].isin(selected_edu)) &
    (df_final['Year'].between(selected_years[0], selected_years[1]))
]

# 📊 Title
st.title("📊 Full Comparison: AI & Automation Impact Across Datasets")

# 📈 Visuals

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌍 Country-wise Impact")
    fig1 = px.bar(
        filtered_df.groupby("Country")[["Automation_Impact", "Predicted_Impact"]].mean().reset_index(),
        x="Country", y=["Automation_Impact", "Predicted_Impact"],
        barmode="group", title="Actual vs Predicted Impact by Country"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("🏭 Sector-wise Impact")
    fig2 = px.bar(
        filtered_df.groupby("Sector")[["Automation_Impact", "Predicted_Impact"]].mean().reset_index(),
        x="Sector", y=["Automation_Impact", "Predicted_Impact"],
        barmode="group", title="Impact by Sector"
    )
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("🎓 Education Level-wise Impact")
    fig3 = px.bar(
        filtered_df.groupby("EducationLevel")[["Automation_Impact", "Predicted_Impact"]].mean().reset_index(),
        x="EducationLevel", y=["Automation_Impact", "Predicted_Impact"],
        barmode="group", title="Impact by Education Level"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("📅 Year-wise Impact Trend")
    fig4 = px.line(
        filtered_df.groupby("Year")[["Automation_Impact", "Predicted_Impact"]].mean().reset_index(),
        x="Year", y=["Automation_Impact", "Predicted_Impact"],
        title="Automation Impact Over Time"
    )
    st.plotly_chart(fig4, use_container_width=True)

# 🔎 View data
with st.expander("🔍 Preview Filtered Data"):
    st.dataframe(filtered_df.head(100))

# 📁 Download option
st.download_button("⬇️ Download Filtered Data", filtered_df.to_csv(index=False), "filtered_data.csv")


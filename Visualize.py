import streamlit as st
import pandas as pd
import plotly.express as px
import xgboost as xgb

# Load datasets
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")
df3 = pd.read_csv("Sectors_Growth_AI_Adoption_dirty_100k.csv")

# Merge datasets on common columns (assuming they share 'Country', 'Sector', 'Year')
df = pd.merge(df1, df2, on=["Country", "Sector", "Year", "EducationLevel"], how="outer")
df = pd.merge(df, df3, on=["Country", "Sector", "Year"], how="outer")

st.set_page_config(page_title="AI & Automation Impact Dashboard", layout="wide")
st.title("🤖 Impact of AI & Automation on Employment")

# Sidebar filters
st.sidebar.header("Filter Data")
countries = st.sidebar.multiselect("Select Countries", df["Country"].dropna().unique(), default=df["Country"].dropna().unique())
sectors = st.sidebar.multiselect("Select Sectors", df["Sector"].dropna().unique(), default=df["Sector"].dropna().unique())
ed_levels = st.sidebar.multiselect("Select Education Levels", df["EducationLevel"].dropna().unique(), default=df["EducationLevel"].dropna().unique())
year_range = st.sidebar.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), (2010, 2024))

# Filter dataset
df = df[(df["Country"].isin(countries)) &
        (df["Sector"].isin(sectors)) &
        (df["EducationLevel"].isin(ed_levels)) &
        (df["Year"].between(*year_range))]

# ---------- METRICS DASHBOARD ----------
st.markdown("## 🧮 Key Metrics Overview")
metrics_df = df.copy()
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Avg Automation Impact", f"{metrics_df['Avg_Automation_Impact'].mean():.2f}")
col2.metric("📉 Pre-AI Unemployment", f"{metrics_df['Avg_PreAI'].mean():.2f}")
col3.metric("📈 Post-AI Unemployment", f"{metrics_df['Avg_PostAI'].mean():.2f}")
col4.metric("🧠 Reskilling Demand", f"{metrics_df['Reskilling_Demand'].mean():.2f}")

# ---------- EDUCATION LEVEL ANALYSIS ----------
st.markdown("## 🎓 Impact by Education Level")
fig_edu = px.bar(df, x="EducationLevel", y="Avg_PostAI", color="Sector", barmode="group",
                 title="Post-AI Unemployment by Education Level")
st.plotly_chart(fig_edu, use_container_width=True)

# ---------- SECTOR GROWTH vs AI ADOPTION ----------
st.markdown("## 🚀 Sector Growth & AI Adoption Heatmap")
heatmap = df.groupby(["Sector", "Year"]).agg({
    "Employment_Growth": "mean",
    "AI_Adoption_Rate": "mean"
}).reset_index()
fig_heatmap = px.density_heatmap(
    heatmap, x="Year", y="Sector", z="AI_Adoption_Rate",
    color_continuous_scale="Viridis", title="AI Adoption Rate Heatmap by Sector"
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------- SKILL GAP & RESKILLING ----------
st.markdown("## ⚙️ Skill Level & Reskilling Need")
if "Skill_Level" in df.columns:
    skill_df = df.copy()
    fig_skills = px.bar(skill_df, x="Skill_Level", y="Reskilling_Demand", color="Sector",
                        title="Skill Level vs Reskilling Demand")
    st.plotly_chart(fig_skills, use_container_width=True)

# ---------- CSV Download ----------
st.download_button("📥 Download Filtered Dataset", data=df.to_csv(index=False),
                   file_name="filtered_ai_impact_data.csv", mime="text/csv")

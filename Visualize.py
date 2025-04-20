import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="AI & Automation Impact", layout="wide")

# 📌 Load Datasets
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")
df3 = pd.read_csv("Sectors_Growth_AI_Adoption_dirty_100k.csv")

# ✅ Rename columns in df1 for merge compatibility
df1.rename(columns={
    "_id.Country": "Country",
    "_id.Sector": "Sector",
    "_id.Year": "Year",
    "_id.EducationLevel": "EducationLevel"
}, inplace=True)

# ✅ Rename df2 column for consistency
df2.rename(columns={"Education_Level": "EducationLevel"}, inplace=True)

# ✅ Drop unused columns (optional but cleaner)
df2 = df2.drop(columns=["ID", "Gender_Distribution", "Date", "Last_Updated"])
df3 = df3.drop(columns=["ID", "tech investment", "Sector_Growth_Decline", "Date", "Last_Updated"])

# ✅ Merge all datasets
merged_1_2 = df1.merge(df2, on=["Country", "Sector", "Year", "EducationLevel"], how="left")
final_df = merged_1_2.merge(df3, on=["Country", "Sector", "Year"], how="left")

# 🌐 Streamlit App Interface
st.title("📈 AI & Automation Impact: Dataset Comparison")

# Filters
countries = st.multiselect("🌍 Select Countries", final_df["Country"].dropna().unique(), default=["USA", "India"])
sectors = st.multiselect("🏭 Select Sectors", final_df["Sector"].dropna().unique(), default=["IT", "Healthcare"])
years = st.slider("📆 Select Year Range", int(final_df["Year"].min()), int(final_df["Year"].max()), (2015, 2024))

# Apply filters
filtered = final_df[
    (final_df["Country"].isin(countries)) &
    (final_df["Sector"].isin(sectors)) &
    (final_df["Year"] >= years[0]) &
    (final_df["Year"] <= years[1])
]

# 💡 Display Data Preview
st.subheader("🔍 Filtered Dataset Preview")
st.dataframe(filtered.head())

# 📊 Visualization 1: Grouped Bar Chart
if not filtered.empty:
    st.subheader("📊 Avg Automation vs Predicted Impact")
    fig = px.bar(
        filtered,
        x="Country",
        y=["Avg_Automation_Impact", "Automation_Impact_Level", "Sector_Impact_Score"],
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig, use_container_width=True)

# 📈 Visualization 2: Line Chart for Growth Trends
    st.subheader("📈 AI Adoption & Growth Rate Over Time")
    fig2 = px.line(
        filtered,
        x="Year",
        y=["AI_Adoption_Rate", "growth rate"],
        color="Country",
        markers=True
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("No data found for selected filters. Try changing the selections.")

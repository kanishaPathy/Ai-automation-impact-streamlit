import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Automation: Full Dataset Comparison", layout="wide")

# Load data
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")
df3 = pd.read_csv("Sectors_Growth_AI_Adoption_dirty_100k.csv")

# 🧹 Clean column names
df1.rename(columns={
    '_id.Country': 'Country',
    '_id.Sector': 'Sector',
    '_id.Year': 'Year',
    '_id.EducationLevel': 'EducationLevel'
}, inplace=True)

df2.rename(columns={'Education_Level': 'EducationLevel'}, inplace=True)

# 🧼 Drop unnecessary columns
df2.drop(columns=['ID', 'Date', 'Last_Updated'], inplace=True, errors='ignore')
df3.drop(columns=['ID', 'Date', 'Last_Updated'], inplace=True, errors='ignore')

# 🔁 Fill missing EducationLevel in df3 for merging (if needed)
df3['EducationLevel'] = 'Unknown'

# 🔗 Merge df1 and df2
merged1 = pd.merge(df1, df2, on=["Country", "Sector", "Year", "EducationLevel"], how="outer")

# 🔗 Merge with df3
final_df = pd.merge(merged1, df3, on=["Country", "Sector", "Year", "EducationLevel"], how="outer")

# Fill missing values
final_df.fillna(0, inplace=True)

# 🎯 Prediction model (optional)
model_path = "xgboost_model.pkl"
if model_path:
    try:
        model = joblib.load(model_path)
        # Add default columns if missing
        feature_cols = [
            'Avg_PreAI', 'Avg_PostAI', 'Avg_Automation_Impact', 'Avg_AI_Role_Jobs',
            'Avg_ReskillingPrograms', 'Skills_Gap', 'Reskilling_Demand', 'Upskilling_Programs',
            'Automation_Impact_Level', 'AI_Adoption_Rate', 'Sector_Impact_Score'
        ]
        for col in feature_cols:
            if col not in final_df.columns:
                final_df[col] = 0
        # Convert categorical columns to numeric
        le = LabelEncoder()
        final_df['Automation_Impact_Level'] = le.fit_transform(final_df['Automation_Impact_Level'].astype(str))
        final_df['AI_Adoption_Rate'] = le.fit_transform(final_df['AI_Adoption_Rate'].astype(str))
        # Predict
        final_df["Predicted_Impact"] = model.predict(final_df[feature_cols])
    except Exception as e:
        st.warning(f"Model prediction skipped: {e}")

# 🎛 Filters
st.sidebar.title("Filters")
countries = st.sidebar.multiselect("Country", final_df["Country"].unique(), default=final_df["Country"].unique())
sectors = st.sidebar.multiselect("Sector", final_df["Sector"].unique(), default=final_df["Sector"].unique())
education = st.sidebar.multiselect("Education Level", final_df["EducationLevel"].unique(), default=final_df["EducationLevel"].unique())
year_range = st.sidebar.slider("Year Range", int(final_df["Year"].min()), int(final_df["Year"].max()), (2010, 2024))

filtered = final_df[
    (final_df["Country"].isin(countries)) &
    (final_df["Sector"].isin(sectors)) &
    (final_df["EducationLevel"].isin(education)) &
    (final_df["Year"].between(year_range[0], year_range[1]))
]

# 📊 Visualizations
st.title("📈 AI & Automation Impact: Dataset Comparison")

tab1, tab2, tab3 = st.tabs(["Country", "Sector", "Year"])

with tab1:
    st.subheader("Impact by Country")
    fig = px.bar(filtered, x="Country", y=["Avg_Automation_Impact", "Automation_Impact_Level", "Predicted_Impact"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Sector-wise Trends")
    fig2 = px.bar(filtered, x="Sector", y=["Avg_SectorGrowth", "Sector_Impact_Score", "Predicted_Impact"], barmode="group")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Year-wise Trends")
    fig3 = px.line(filtered, x="Year", y=["Avg_PostAI", "AI_Adoption_Rate", "Predicted_Impact"], markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# 📥 Download
st.download_button("Download Filtered Data", filtered.to_csv(index=False), "combined_filtered_data.csv")

# 📋 Preview
with st.expander("Show Filtered Data"):
    st.dataframe(filtered.head(100))

import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🌐 AI & Automation Impact Visualization Dashboard")

# Load datasets
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")
df3 = pd.read_csv("Sectors_Growth_AI_Adoption_dirty_100k.csv")

# Rename columns in df1 for merging
df1.rename(columns={
    '_id.Country': 'Country',
    '_id.Sector': 'Sector',
    '_id.Year': 'Year',
    '_id.EducationLevel': 'EducationLevel'
}, inplace=True)

# Ensure required columns exist in df2 and df3
required_columns = ['Country', 'Sector', 'Year', 'EducationLevel']
for col in required_columns:
    if col not in df2.columns:
        df2[col] = None
    if col not in df3.columns:
        df3[col] = None

# Merge all three datasets
df_merged = pd.merge(df1, df2, on=required_columns, how="outer")
df_final = pd.merge(df_merged, df3, on=required_columns, how="outer")

# Sidebar filters
st.sidebar.header("🔍 Filter Options")
years = sorted(df_final["Year"].dropna().unique())
selected_year = st.sidebar.selectbox("Select Year", years)

sectors = df_final["Sector"].dropna().unique()
selected_sector = st.sidebar.selectbox("Select Sector", sectors)

countries = df_final["Country"].dropna().unique()
selected_country = st.sidebar.selectbox("Select Country", countries)

edu_levels = df_final["EducationLevel"].dropna().unique()
selected_edu = st.sidebar.selectbox("Select Education Level", edu_levels)

# Filter based on selections
filtered_df = df_final[
    (df_final["Year"] == selected_year) &
    (df_final["Sector"] == selected_sector) &
    (df_final["Country"] == selected_country) &
    (df_final["EducationLevel"] == selected_edu)
]

st.subheader("📊 Filtered Data")
st.dataframe(filtered_df)

# Prediction Section (XGBoost)
if 'Automation_Impact' in df_final.columns:
    st.subheader("📈 Predicting Automation Impact")

    features = ['PreAI_Unemployment', 'PostAI_Unemployment', 'AI_Role_Jobs', 'Skills_Gap', 'Upskilling_Programs']
    available_features = [f for f in features if f in df_final.columns]
    df_model = df_final.dropna(subset=available_features + ['Automation_Impact'])

    X = df_model[available_features]
    y = df_model['Automation_Impact']

    model = xgb.XGBRegressor()
    model.fit(X, y)

    df_final['Predicted_Impact'] = model.predict(df_final[available_features].fillna(0))

    st.write("✅ Prediction complete. Here's a preview:")
    st.dataframe(df_final[['Country', 'Sector', 'Year', 'EducationLevel', 'Predicted_Impact']].dropna())

    # Visualization
    st.subheader("📉 Actual vs Predicted Automation Impact")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y, y=model.predict(X), ax=ax)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Automation Impact")
    st.pyplot(fig)

    # Download results
    csv = df_final.to_csv(index=False)
    st.download_button("⬇️ Download Full Data with Predictions", csv, file_name="merged_automation_predictions.csv", mime="text/csv")

else:
    st.warning("⚠️ 'Automation_Impact' column not found. Prediction not available.")

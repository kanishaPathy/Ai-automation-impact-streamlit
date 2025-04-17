
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the trained XGBoost model
model = joblib.load(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\xgboost_model.pkl")

st.set_page_config(page_title="AI Impact Predictor", layout="wide")
st.title("🌐 AI Automation Impact Prediction Dashboard")

# User inputs
col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Select Year", [2021, 2022, 2023])
    country = st.selectbox("Select Country", ['Canada', 'Germany', 'India', 'Ireland', 'USA'])
    sector = st.selectbox("Select Sector", ['Agriculture', 'Education', 'Finance', 'Healthcare', 'IT',
                                            'Manufacturing', 'Media & Entertainment', 'Retail', 'Transportation'])
    education_level = st.selectbox("Select Education Level", ['High School', 'Bachelor', 'Master', 'PhD', 'Unknown'])

with col2:
    avg_pre_ai = st.slider("Average Pre-AI Impact", 0, 100, 50)
    avg_post_ai = st.slider("Average Post-AI Impact", 0, 100, 50)
    avg_automation_impact = st.slider("Average Automation Impact", 0, 100, 50)
    avg_ai_role_jobs = st.slider("Average AI Role Jobs", 0, 100, 50)
    avg_reskilling_programs = st.slider("Average Reskilling Programs", 0, 100, 50)
    avg_economic_impact = st.slider("Average Economic Impact", 0, 100, 50)
    avg_sector_growth = st.slider("Average Sector Growth", 0, 100, 50)

# Create DataFrame
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

# Load training data to align encoding
training_df = pd.read_csv(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\Unemployment_jobcreation_db.Unemployment_data.csv")
X_train = training_df.drop(columns=['Avg_Automation_Impact'])
X_train_encoded = pd.get_dummies(X_train)

# Encode and align user input
input_encoded = pd.get_dummies(input_df)
for col in X_train_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_train_encoded.columns]

# Predict
prediction = model.predict(input_encoded)
rounded_prediction = round(prediction[0], 2)

# Display result
st.subheader("📊 Prediction Result")
st.markdown(f"### 🔮 Predicted Automation Impact: **{rounded_prediction}**")

# --- Bar chart of input features
st.subheader("📈 Input Feature Overview")
feature_values = {
    "Pre-AI Impact": avg_pre_ai,
    "Post-AI Impact": avg_post_ai,
    "AI Role Jobs": avg_ai_role_jobs,
    "Reskilling Programs": avg_reskilling_programs,
    "Economic Impact": avg_economic_impact,
    "Sector Growth": avg_sector_growth
}

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=list(feature_values.keys()), y=list(feature_values.values()), palette="coolwarm", ax=ax)
ax.set_title("Input Feature Comparison")
ax.set_ylabel("Impact Score")
plt.xticks(rotation=30)
st.pyplot(fig)

# --- Download input + prediction
result_df = input_df.copy()
result_df["Predicted_Automation_Impact"] = prediction[0]

csv = result_df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Prediction Data as CSV", data=csv, file_name="automation_impact_prediction.csv", mime='text/csv')

# --- Feature importance (optional, if available)
try:
    importance = model.feature_importances_
    feature_names = input_encoded.columns
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values(by="Importance", ascending=False)

    st.subheader("🔍 Feature Importance (from XGBoost)")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(data=imp_df.head(10), x="Importance", y="Feature", palette="viridis", ax=ax2)
    st.pyplot(fig2)
except Exception as e:
    st.info("Feature importance not available for this model.")

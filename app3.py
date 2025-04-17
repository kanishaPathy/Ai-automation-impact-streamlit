import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\xgboost_model.pkl")

# Load dataset
df = pd.read_csv(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\Unemployment_jobcreation_db.Unemployment_data.csv")

st.title("AI Automation Impact Prediction & Analysis Dashboard")

st.sidebar.header("Prediction Inputs")
year = st.sidebar.selectbox("Select Year", [2021, 2022, 2023])
country = st.sidebar.selectbox("Select Country", ['Canada', 'Germany', 'India', 'Ireland', 'USA'])
sector = st.sidebar.selectbox("Select Sector", ['Agriculture', 'Education', 'Finance', 'Healthcare', 'IT',
                                                'Manufacturing', 'Media & Entertainment', 'Retail', 'Transportation'])
education_level = st.sidebar.selectbox("Select Education Level", ['High School', 'Bachelor', 'Master', 'PhD', 'Unknown'])
avg_pre_ai = st.sidebar.slider("Average Pre-AI Impact", 0, 100, 50)
avg_post_ai = st.sidebar.slider("Average Post-AI Impact", 0, 100, 50)
avg_automation_impact = st.sidebar.slider("Average Automation Impact", 0, 100, 50)
avg_ai_role_jobs = st.sidebar.slider("Average AI Role Jobs", 0, 100, 50)
avg_reskilling_programs = st.sidebar.slider("Average Reskilling Programs", 0, 100, 50)
avg_economic_impact = st.sidebar.slider("Average Economic Impact", 0, 100, 50)
avg_sector_growth = st.sidebar.slider("Average Sector Growth", 0, 100, 50)

# Input for prediction
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

# Prepare data for model prediction
X_train = df.drop(columns=['Avg_Automation_Impact'])
X_train_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df)

for col in X_train_encoded.columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[X_train_encoded.columns]

# Predict
prediction = model.predict(input_encoded)
st.success(f"Predicted Automation Impact: {prediction[0]:.2f}")

st.markdown("---")
st.header("1️⃣ Country-wise Comparison")

country1 = st.selectbox("Select First Country", df['_id.Country'].unique(), key='c1')
country2 = st.selectbox("Select Second Country", df['_id.Country'].unique(), key='c2')
compare_df = df[df['_id.Country'].isin([country1, country2])]

fig1, ax1 = plt.subplots(figsize=(6, 3.5))
sns.barplot(data=compare_df, x='_id.Sector', y='Avg_Automation_Impact', hue='_id.Country', ax=ax1)
ax1.set_title(f"Automation Impact: {country1} vs {country2}", fontsize=10)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, fontsize=8)
ax1.legend(fontsize=7)
st.pyplot(fig1)

st.markdown("---")
st.header("2️⃣ Unemployment Impact (Pre-AI vs Post-AI)")

sector_choice = st.selectbox("Select Sector for Unemployment Impact", df['_id.Sector'].unique())
df_sector = df[df['_id.Sector'] == sector_choice]
df_sector_grouped = df_sector.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(6, 3.5))
df_sector_grouped.set_index('_id.Year').plot(kind='bar', ax=ax2)
ax2.set_title(f"Unemployment Impact in {sector_choice}", fontsize=10)
ax2.set_ylabel("Impact Score", fontsize=9)
ax2.set_xlabel("Year", fontsize=9)
st.pyplot(fig2)

st.markdown("---")
st.header("3️⃣ Education Level Impact on AI Automation")

edu_df = df.groupby('_id.EducationLevel')['Avg_Automation_Impact'].mean().reset_index()

fig3, ax3 = plt.subplots(figsize=(6, 3.5))
sns.barplot(data=edu_df, x='_id.EducationLevel', y='Avg_Automation_Impact', palette="viridis", ax=ax3)
ax3.set_title("Avg Automation Impact by Education Level", fontsize=10)
ax3.set_ylabel("Impact", fontsize=9)
ax3.set_xlabel("Education Level", fontsize=9)
st.pyplot(fig3)

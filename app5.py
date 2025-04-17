
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load model and dataset
model = joblib.load(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\xgboost_model.pkl")
df = pd.read_csv(r"C:\Users\Kanisha Pathy\OneDrive\Desktop\KP\Project-Updated_AI_impact\Unemployment_jobcreation_db.Unemployment_data.csv")

# Page title
st.set_page_config(layout="wide")
st.title("🤖 AI Automation Impact Prediction & Analysis Dashboard")

# Sidebar inputs
st.sidebar.header("📌 Prediction Inputs")
year = st.sidebar.selectbox("Select Year", list(range(2010, 2026)), key='year_select')
country = st.sidebar.selectbox("Select Country", ['Canada', 'Germany', 'India', 'Ireland', 'USA'], key='country_select')
sector = st.sidebar.selectbox("Select Sector", ['Agriculture', 'Education', 'Finance', 'Healthcare', 'IT',
                                                'Manufacturing', 'Media & Entertainment', 'Retail', 'Transportation'], key='sector_select')
education_level = st.sidebar.selectbox("Select Education Level", ['High School', 'Bachelor', 'Master', 'PhD', 'Unknown'], key='education_select')
avg_pre_ai = st.sidebar.slider("Average Pre-AI Impact", 0, 100, 50, key='pre_ai_slider')
avg_post_ai = st.sidebar.slider("Average Post-AI Impact", 0, 100, 50, key='post_ai_slider')
avg_automation_impact = st.sidebar.slider("Average Automation Impact", 0, 100, 50, key='automation_impact_slider')
avg_ai_role_jobs = st.sidebar.slider("Average AI Role Jobs", 0, 100, 50, key='ai_role_jobs_slider')
avg_reskilling_programs = st.sidebar.slider("Average Reskilling Programs", 0, 100, 50, key='reskilling_slider')
avg_economic_impact = st.sidebar.slider("Average Economic Impact", 0, 100, 50, key='economic_impact_slider')
avg_sector_growth = st.sidebar.slider("Average Sector Growth", 0, 100, 50, key='sector_growth_slider')

# Input DataFrame
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

# Prediction
prediction = model.predict(input_encoded)
st.success(f"🔮 Predicted Automation Impact Score: **{prediction[0]:.2f}**")

# Country-wise Comparison
st.markdown("---")
st.header("1️⃣ Country-wise Comparison of Automation Impact")
country1 = st.selectbox("Select First Country", df['_id.Country'].unique(), key='country1_select')
country2 = st.selectbox("Select Second Country", df['_id.Country'].unique(), key='country2_select')
compare_df = df[df['_id.Country'].isin([country1, country2])]

fig1, ax1 = plt.subplots(figsize=(8, 4))
sns.barplot(data=compare_df, x='_id.Sector', y='Avg_Automation_Impact', hue='_id.Country', ax=ax1)
ax1.set_title(f"Automation Impact: {country1} vs {country2}", fontsize=12)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
st.pyplot(fig1)

# Unemployment Trend Over Time
st.markdown("---")
st.header("2️⃣ Unemployment Impact (Pre-AI vs Post-AI) Over Time")
unemployment_df = df.groupby(['_id.Year']).agg({'Avg_PreAI': 'mean', 'Avg_PostAI': 'mean'}).reset_index()
fig2 = plt.figure(figsize=(8, 4))
plt.plot(unemployment_df['_id.Year'], unemployment_df['Avg_PreAI'], label='Pre-AI Impact', color='skyblue', marker='o')
plt.plot(unemployment_df['_id.Year'], unemployment_df['Avg_PostAI'], label='Post-AI Impact', color='orange', marker='o')
plt.title('Unemployment Impact Over Time')
plt.xlabel('Year')
plt.ylabel('Impact')
plt.legend()
st.pyplot(fig2)

# Sector-wise Trend
st.markdown("---")
st.header("3️⃣ Sector-wise Unemployment Comparison")
sector_choice = st.selectbox("Select Sector", df['_id.Sector'].unique(), key='sector_choice_select')
df_sector = df[df['_id.Sector'] == sector_choice]
df_sector_grouped = df_sector.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3, ax3 = plt.subplots(figsize=(8, 4))
df_sector_grouped.set_index('_id.Year').plot(kind='bar', ax=ax3)
ax3.set_title(f"Unemployment Impact in {sector_choice}", fontsize=12)
ax3.set_ylabel("Impact Score")
ax3.set_xlabel("Year")
st.pyplot(fig3)

# Education Level Impact
st.markdown("---")
st.header("4️⃣ Education Level Impact on Automation")
edu_df = df.groupby('_id.EducationLevel').agg({'Avg_PreAI': 'mean', 'Avg_PostAI': 'mean', 'Avg_Automation_Impact': 'mean'}).reset_index()
fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.barplot(x='Avg_PreAI', y='_id.EducationLevel', data=edu_df, label='Pre-AI Impact', color='skyblue')
sns.barplot(x='Avg_PostAI', y='_id.EducationLevel', data=edu_df, label='Post-AI Impact', color='orange')
plt.title('Education Level Impact on AI (Pre vs Post)')
plt.legend()
st.pyplot(fig4)

# Export input + prediction
st.markdown("---")
if st.button("📁 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction[0]
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("✅ Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Developed by [Your Name] — Powered by Streamlit, XGBoost, Seaborn, Plotly")


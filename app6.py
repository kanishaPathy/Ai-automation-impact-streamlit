import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np

# Load your dataset
df = pd.read_csv("your_dataset.csv")  # Replace with your dataset path

# Load the model
model = xgb.XGBRegressor()
model.load_model("xgboost_model.json")  # Path to your XGBoost model

# Streamlit layout
st.title("🚀 AI & Automation Impact Analysis")

# 1️⃣ Country-wise Comparison
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

# 2️⃣ Unemployment Impact Over Time (Pre vs Post AI)
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

# 3️⃣ Sector-wise Unemployment Comparison
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

# 4️⃣ Education Level Impact on Automation
st.markdown("---")
st.header("4️⃣ Education Level Impact on Automation")
edu_df = df.groupby('_id.EducationLevel').agg({'Avg_PreAI': 'mean', 'Avg_PostAI': 'mean', 'Avg_Automation_Impact': 'mean'}).reset_index()
fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.barplot(x='Avg_PreAI', y='_id.EducationLevel', data=edu_df, label='Pre-AI Impact', color='skyblue')
sns.barplot(x='Avg_PostAI', y='_id.EducationLevel', data=edu_df, label='Post-AI Impact', color='orange')
plt.title('Education Level Impact on AI (Pre vs Post)')
plt.legend()
st.pyplot(fig4)

# 5️⃣ Predictive Model and Input
st.markdown("---")
st.header("5️⃣ Predict Automation Impact")
# Let the user input data for prediction
input_data = {
    'Country': st.selectbox("Select Country for Prediction", df['_id.Country'].unique(), key="input_country"),
    'Sector': st.selectbox("Select Sector", df['_id.Sector'].unique(), key="input_sector"),
    'Education Level': st.selectbox("Select Education Level", df['_id.EducationLevel'].unique(), key="input_edu"),
    'Year': st.selectbox("Select Year", df['_id.Year'].unique(), key="input_year")
}

# Prepare the input data for prediction
input_df = pd.DataFrame([input_data])

# Feature engineering for prediction (ensure it matches the model input features)
# Here, you would transform input_df to have the same format as your training data
# For simplicity, assume the model accepts these features directly
# You may need to perform encoding, scaling, etc., based on your model training process

prediction = model.predict(input_df)  # Replace with your feature preprocessing if necessary

st.write(f"Predicted Automation Impact: {prediction[0]:.2f}")

# 6️⃣ Export input + prediction
st.markdown("---")
if st.button("📁 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction[0]
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("✅ Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Developed by [Your Name] — Powered by Streamlit, XGBoost, Seaborn, Plotly")

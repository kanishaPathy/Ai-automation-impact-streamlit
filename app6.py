import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# Load your dataset (Make sure to upload your dataset to Streamlit)
@st.cache
def load_data():
     df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")   # Ensure to change path to the actual dataset
    return df

df = load_data()

# Function to simulate prediction (replace with your actual model prediction)
def predict_automation_impact(year, country, sector, education_level):
    # Dummy logic for prediction
    # Replace this with actual prediction using your trained model
    return df[(df['_id.Year'] == year) & 
              (df['_id.Country'] == country) & 
              (df['_id.Sector'] == sector) & 
              (df['_id.EducationLevel'] == education_level)]['Avg_Automation_Impact'].mean()

# 1. 📌 Prediction Inputs Section
st.markdown("# 📌 Predict Automation Impact")
st.header("Select Prediction Inputs")

# 1.1. Select Year (Slider)
year = st.slider("Select Year", min_value=2010, max_value=2024, value=2015)
st.write(f"Year Selected: {year}")

# 1.2. Select Country
country = st.selectbox("Select Country", df['_id.Country'].unique())
st.write(f"Country Selected: {country}")

# 1.3. Select Sector (Dropdown)
sector = st.selectbox("Select Sector", df['_id.Sector'].unique())
st.write(f"Sector Selected: {sector}")

# 1.4. Select Education Level (Dropdown)
education_level = st.selectbox("Select Education Level", df['_id.EducationLevel'].unique())
st.write(f"Education Level Selected: {education_level}")

# 2. 📊 Show Selected Input Summary
st.markdown("### Input Summary:")
st.write(f"Prediction for Year: {year}, Country: {country}, Sector: {sector}, Education Level: {education_level}")

# 3. 🚀 Predict Automation Impact
if st.button("🚀 Predict Automation Impact"):
    # Call the prediction function (replace with your trained model)
    predicted_impact = predict_automation_impact(year, country, sector, education_level)
    st.markdown(f"### Predicted Automation Impact: {predicted_impact:.2f}")

    # Plotting the prediction (could be a bar plot or other visualization)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=['Pre-AI', 'Post-AI'], y=[predicted_impact - 10, predicted_impact + 10], ax=ax, palette="coolwarm")
    ax.set_title(f"Predicted Automation Impact for {sector} in {country}")
    st.pyplot(fig)

# 4. 📈 Country-wise Comparison
st.markdown("---")
st.header("1️⃣ Country-wise Comparison of Automation Impact")

country1 = st.selectbox("Select First Country for Comparison", df['_id.Country'].unique(), key='country1_select')
country2 = st.selectbox("Select Second Country for Comparison", df['_id.Country'].unique(), key='country2_select')

# Ensure countries are not the same for comparison
if country1 != country2:
    compare_df = df[df['_id.Country'].isin([country1, country2])]
    
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=compare_df, x='_id.Sector', y='Avg_Automation_Impact', hue='_id.Country', ax=ax1, palette='viridis')
    ax1.set_title(f"Automation Impact: {country1} vs {country2}", fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    st.pyplot(fig1)
else:
    st.warning("Please select different countries for comparison.")

# 5. 📉 Unemployment Trend Over Time (Pre vs Post-AI)
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

# 6. 📊 Sector-wise Comparison of Unemployment
st.markdown("---")
st.header("3️⃣ Sector-wise Unemployment Comparison")

sector_choice = st.selectbox("Select Sector", df['_id.Sector'].unique(), key='sector_choice_select')
df_sector = df[df['_id.Sector'] == sector_choice]
df_sector_grouped = df_sector.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()

fig3, ax3 = plt.subplots(figsize=(8, 4))
df_sector_grouped.set_index('_id.Year').plot(kind='bar', ax=ax3, color=['skyblue', 'orange'])
ax3.set_title(f"Unemployment Impact in {sector_choice}", fontsize=12)
ax3.set_ylabel("Impact Score")
ax3.set_xlabel("Year")
st.pyplot(fig3)

# 7. 🎓 Education Level Impact on Automation
st.markdown("---")
st.header("4️⃣ Education Level Impact on Automation")

edu_df = df.groupby('_id.EducationLevel').agg({'Avg_PreAI': 'mean', 'Avg_PostAI': 'mean', 'Avg_Automation_Impact': 'mean'}).reset_index()

fig4, ax4 = plt.subplots(figsize=(8, 4))
sns.barplot(x='Avg_PreAI', y='_id.EducationLevel', data=edu_df, label='Pre-AI Impact', color='skyblue')
sns.barplot(x='Avg_PostAI', y='_id.EducationLevel', data=edu_df, label='Post-AI Impact', color='orange')
plt.title('Education Level Impact on AI (Pre vs Post)')
plt.legend()
st.pyplot(fig4)

# 8. 💾 Save Prediction to CSV
st.markdown("---")
if st.button("📁 Save Prediction to CSV"):
    prediction_data = {
        "Year": year,
        "Country": country,
        "Sector": sector,
        "Education Level": education_level,
        "Predicted Automation Impact": predicted_impact
    }
    prediction_df = pd.DataFrame([prediction_data])
    prediction_df.to_csv("saved_prediction.csv", index=False)
    st.success("✅ Prediction saved to **saved_prediction.csv**")

# 9. 📝 Footer
st.markdown("---")
st.caption("📊 Developed by [Your Name] — Powered by Streamlit, XGBoost, Seaborn, Plotly")

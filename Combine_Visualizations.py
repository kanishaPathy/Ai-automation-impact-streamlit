import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and data
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
skill_df = pd.read_csv("reskilling_dataset_cleaned.csv")

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Select Parameters for Prediction")
col1, col2, col3, col4 = st.columns(4)
year_range = col1.slider("Select Year Range", int(df['_id.Year'].min()), int(df['_id.Year'].max()), (2010, 2022))
country = col2.selectbox("Country", sorted(df['_id.Country'].unique()))
sector = col3.selectbox("Sector", sorted(df['_id.Sector'].unique()))
education = col4.selectbox("Education Level", sorted(df['_id.EducationLevel'].unique()))

# Prepare input data for prediction (using first year in range)
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year_range[0]],
    '_id.EducationLevel': [education],
})

# Encoding and predictions
X_train = df.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]

st.success(f"🔮 Predicted Automation Impact Score for {year_range[0]}: **{prediction:.2f}**")

# ---------- Visualization Section ----------
st.markdown("---")
st.header(f"🌍 Country Comparison from {year_range[0]} to {year_range[1]}")
cols = st.columns(2)
country1 = cols[0].selectbox("Select First Country", df['_id.Country'].unique(), key='country1')
country2 = cols[1].selectbox("Select Second Country", [c for c in df['_id.Country'].unique() if c != country1], key='country2')

compare_df = df[(df['_id.Country'].isin([country1, country2])) & (df['_id.Year'] >= year_range[0]) & (df['_id.Year'] <= year_range[1])]
fig1 = px.bar(compare_df, x='_id.Sector', y='Avg_Automation_Impact', color='_id.Country',
              title=f'Automation Impact: {country1} vs {country2} ({year_range[0]} - {year_range[1]})',
              barmode='group', height=400)
st.plotly_chart(fig1, use_container_width=True)

# Unemployment Over Time Visualization
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
unemp = df[(df['_id.Year'] >= year_range[0]) & (df['_id.Year'] <= year_range[1])]
unemp = unemp.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig2 = px.line(unemp, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
               title=f'Unemployment Impact (Pre-AI vs Post-AI) from {year_range[0]} to {year_range[1]}')
st.plotly_chart(fig2, use_container_width=True)

# Sector-wise Trend Visualization
st.markdown("---")
st.header("🏭 Sector-wise Unemployment Comparison")
sector_selected = st.selectbox("Select Sector", df['_id.Sector'].unique(), key='sector_analysis')
df_sec = df[(df['_id.Sector'] == sector_selected) & (df['_id.Year'] >= year_range[0]) & (df['_id.Year'] <= year_range[1])]
df_sec = df_sec.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3 = px.bar(df_sec, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title=f'Unemployment Trend in {sector_selected} ({year_range[0]} - {year_range[1]})')
st.plotly_chart(fig3, use_container_width=True)

# Education Level Impact Visualization
st.markdown("---")
st.header("🎓 Education Level Impact")
edu_impact = df[(df['_id.Year'] >= year_range[0]) & (df['_id.Year'] <= year_range[1])]
edu_impact = edu_impact.groupby('_id.EducationLevel')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig4 = px.bar(edu_impact, x='_id.EducationLevel', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title='Education Level vs AI Impact')
st.plotly_chart(fig4, use_container_width=True)

# Skill Level Impact (Reskilling Dataset)
st.markdown("---")
st.header("💼 Skill Level Impact")
# Ensure 'Automation_Impact_Level' is numeric (convert non-numeric to NaN and handle them)
skill_df['Automation_Impact_Level'] = pd.to_numeric(skill_df['Automation_Impact_Level'], errors='coerce')
skill_df['Automation_Impact_Level'].fillna(0, inplace=True)

# Simulate PreAI and PostAI scores based on Automation_Impact_Level
skill_df['Avg_PreAI'] = skill_df['Automation_Impact_Level'] * 0.6  # Simulated example
skill_df['Avg_PostAI'] = skill_df['Automation_Impact_Level'] * 1.1

fig7 = px.bar(skill_df, x='Skill_Level', y=['Avg_PreAI', 'Avg_PostAI'], barmode='group', title="Skill Level Impact on PreAI vs PostAI")
st.plotly_chart(fig7, use_container_width=True)

# Gender Distribution in IT vs Retail (Assuming Gender_Distribution column exists in skill_df)
st.markdown("---")
st.header("👩‍💻 Gender Distribution in IT vs Retail")
gender_df = skill_df[skill_df['Sector'].isin(['IT', 'Retail'])]
gender_df = gender_df.groupby(['Sector', 'Gender_Distribution']).size().reset_index(name='Count')
fig6 = px.bar(gender_df, x='Sector', y='Count', color='Gender_Distribution', title='Gender Distribution in IT vs Retail')
st.plotly_chart(fig6, use_container_width=True)

# Export Prediction Option
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

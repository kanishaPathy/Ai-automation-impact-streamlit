import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and data
model = joblib.load("xgboost_model.pkl")
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")

# Rename columns in df2 to match df1
df2.rename(columns={
    'Country': '_id.Country',
    'Sector': '_id.Sector',
    'Year': '_id.Year',
}, inplace=True)

# Merge the dataframes
merged_df = pd.merge(df1, df2, how='inner', on=['_id.Country', '_id.Sector', '_id.Year'])

# Set Streamlit page configuration
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Select Parameters for Prediction")
col1, col2, col3, col4 = st.columns(4)
year_range = col1.slider("Select Year Range", int(df1['_id.Year'].min()), int(df1['_id.Year'].max()), (2010, 2022))
country = col2.selectbox("Country", sorted(df1['_id.Country'].unique()))
sector = col3.selectbox("Sector", sorted(df1['_id.Sector'].unique()))
education = col4.selectbox("Education Level", sorted(df1['_id.EducationLevel'].unique()))

# Prepare input data for prediction (using first year in range)
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year_range[0]],
    '_id.EducationLevel': [education],
})

# Encoding and predictions
X_train = df1.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]

st.success(f"🔮 Predicted Automation Impact Score for {year_range[0]}: **{prediction:.2f}**")

# ---------- Visualization Section ----------
st.markdown("---")
st.header(f"🌍 Country Comparison from {year_range[0]} to {year_range[1]}")
cols = st.columns(2)
country1 = cols[0].selectbox("Select First Country", df1['_id.Country'].unique(), key='country1')
country2 = cols[1].selectbox("Select Second Country", [c for c in df1['_id.Country'].unique() if c != country1], key='country2')

compare_df = df1[(df1['_id.Country'].isin([country1, country2])) & (df1['_id.Year'] >= year_range[0]) & (df1['_id.Year'] <= year_range[1])]
fig1 = px.bar(compare_df, x='_id.Sector', y='Avg_Automation_Impact', color='_id.Country',
              title=f'Automation Impact: {country1} vs {country2} ({year_range[0]} - {year_range[1]})',
              barmode='group', height=400)
st.plotly_chart(fig1, use_container_width=True)

# Unemployment Over Time Visualization
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
unemp = df1[(df1['_id.Year'] >= year_range[0]) & (df1['_id.Year'] <= year_range[1])]
unemp = unemp.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig2 = px.line(unemp, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
               title=f'Unemployment Impact (Pre-AI vs Post-AI) from {year_range[0]} to {year_range[1]}')
st.plotly_chart(fig2, use_container_width=True)

# Sector-wise Trend Visualization
st.markdown("---")
st.header("🏭 Sector-wise Unemployment Comparison")
sector_selected = st.selectbox("Select Sector", df1['_id.Sector'].unique(), key='sector_analysis')
df_sec = df1[(df1['_id.Sector'] == sector_selected) & (df1['_id.Year'] >= year_range[0]) & (df1['_id.Year'] <= year_range[1])]
df_sec = df_sec.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3 = px.bar(df_sec, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title=f'Unemployment Trend in {sector_selected} ({year_range[0]} - {year_range[1]})')
st.plotly_chart(fig3, use_container_width=True)

# Education Level Impact Visualization
st.markdown("---")
st.header("🎓 Education Level Impact")
edu_impact = df1[(df1['_id.Year'] >= year_range[0]) & (df1['_id.Year'] <= year_range[1])]
edu_impact = edu_impact.groupby('_id.EducationLevel')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig4 = px.bar(edu_impact, x='_id.EducationLevel', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title='Education Level vs AI Impact')
st.plotly_chart(fig4, use_container_width=True)

# Country vs All Sectors Comparison
st.markdown("---")
st.header("🌐 Country vs Sector Comparison")
col_c1, col_c2 = st.columns(2)
country_vs = col_c1.selectbox("Select Country", df1['_id.Country'].unique(), key='country_vs')
sector_vs = col_c2.selectbox("Compare With Sector (Optional)", ['All'] + list(df1['_id.Sector'].unique()), key='sector_vs')

filter_df = df1[(df1['_id.Country'] == country_vs) & (df1['_id.Year'] >= year_range[0]) & (df1['_id.Year'] <= year_range[1])]
if sector_vs != 'All':
    filter_df = filter_df[filter_df['_id.Sector'] == sector_vs]

fig5 = px.bar(filter_df, x='_id.Sector', y='Avg_Automation_Impact', color='_id.Year',
              title=f'{country_vs} vs {"All Sectors" if sector_vs == "All" else sector_vs} Impact Comparison')
st.plotly_chart(fig5, use_container_width=True)

# Gender Distribution in IT vs Retail (Assuming Gender_Distribution column exists in skill_df)
st.markdown("---")
st.header("👩‍💻 Gender Distribution in IT vs Retail")
gender_df = df2[df2['Sector'].isin(['IT', 'Retail'])]
gender_df = gender_df.groupby(['Sector', 'Gender_Distribution']).size().reset_index(name='Count')
fig6 = px.bar(gender_df, x='Sector', y='Count', color='Gender_Distribution', title='Gender Distribution in IT vs Retail')
st.plotly_chart(fig6, use_container_width=True)

# Skill Level Impact based on PreAI vs PostAI
st.markdown("---")
st.header("💼 Skill Level Impact")
# Ensure 'Automation_Impact_Level' is numeric (convert non-numeric to NaN and handle them)
df2['Automation_Impact_Level'] = pd.to_numeric(df2['Automation_Impact_Level'], errors='coerce')
df2['Automation_Impact_Level'].fillna(0, inplace=True)

# Simulate PreAI and PostAI scores based on Automation_Impact_Level
df2['Avg_PreAI'] = df2['Automation_Impact_Level'] * 0.6  # Simulated example
df2['Avg_PostAI'] = df2['Automation_Impact_Level'] * 1.1

fig7 = px.bar(df2, x='Skill_Level', y=['Avg_PreAI', 'Avg_PostAI'], barmode='group', title="Skill Level Impact on PreAI vs PostAI")
st.plotly_chart(fig7, use_container_width=True)

# Export Prediction Option
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

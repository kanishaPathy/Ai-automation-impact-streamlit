import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and main dataset
model = joblib.load("xgboost_model.pkl")
df1 = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")  # <--- second dataset file

# Rename columns in df2 for consistency
df2 = df2.rename(columns={
    'Country': '_id.Country',
    'Sector': '_id.Sector',
    'Year': '_id.Year',
    'Education_Level': '_id.EducationLevel',
    'Automation_Impact_Level': 'SecondDS_Automation_Impact',
    'Reskilling_Demand': 'SecondDS_Reskilling_Demand',
    'Skills_Gap': 'SecondDS_Skills_Gap',
    'Upskilling_Programs': 'SecondDS_Upskilling_Programs'
})

# Merge both datasets
combined_df = pd.merge(df1, df2, on=['_id.Country', '_id.Sector', '_id.Year', '_id.EducationLevel'], how='inner')

# Streamlit UI
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs ----------
st.markdown("### 🎯 Choose Parameters for Prediction")
col1, col2, col3, col4 = st.columns(4)
year = col1.selectbox("Year", sorted(df1['_id.Year'].unique()))
country = col2.selectbox("Country", sorted(df1['_id.Country'].unique()))
sector = col3.selectbox("Sector", sorted(df1['_id.Sector'].unique()))
education = col4.selectbox("Education Level", sorted(df1['_id.EducationLevel'].unique()))

input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year],
    '_id.EducationLevel': [education],
})

# Encoding for prediction
X_train = df1.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]

st.success(f"🔮 Predicted Automation Impact Score: **{prediction:.2f}**")

# ---------- Country Comparison ----------
st.markdown("---")
st.header("🌍 Country Comparison")
cols = st.columns(2)
country1 = cols[0].selectbox("Select First Country", df1['_id.Country'].unique(), key='country1')
country2 = cols[1].selectbox("Select Second Country", [c for c in df1['_id.Country'].unique() if c != country1], key='country2')

compare_df = df1[df1['_id.Country'].isin([country1, country2])]
fig1 = px.bar(compare_df, x='_id.Sector', y='Avg_Automation_Impact', color='_id.Country',
              title=f'Automation Impact: {country1} vs {country2}',
              barmode='group', height=400)
st.plotly_chart(fig1, use_container_width=True)

# ---------- Unemployment Trend ----------
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
unemp = df1.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig2 = px.line(unemp, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
               title='Unemployment Impact (Pre-AI vs Post-AI) Over Years')
st.plotly_chart(fig2, use_container_width=True)

# ---------- Sector-wise Visualization ----------
st.markdown("---")
st.header("🏭 Sector-wise Unemployment Comparison")
sector_selected = st.selectbox("Select Sector", df1['_id.Sector'].unique(), key='sector_analysis')
df_sec = df1[df1['_id.Sector'] == sector_selected].groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3 = px.bar(df_sec, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title=f'Unemployment Trend in {sector_selected}')
st.plotly_chart(fig3, use_container_width=True)

# ---------- NEW: Education Level Comparison ----------
st.markdown("---")
st.header("🎓 Education Level Impact Comparison (Both Datasets)")
edu_compare = combined_df.groupby('_id.EducationLevel')[['Avg_Automation_Impact', 'SecondDS_Automation_Impact']].mean().reset_index()
fig4 = px.bar(edu_compare, x='_id.EducationLevel', y=['Avg_Automation_Impact', 'SecondDS_Automation_Impact'],
              barmode='group', title='Automation Impact by Education Level (Dataset1 vs Dataset2)')
st.plotly_chart(fig4, use_container_width=True)

# ---------- NEW: Dataset Comparison Over Time ----------
st.markdown("---")
st.header("⏳ Automation Impact Comparison Over Years (Both Datasets)")
time_compare = combined_df.groupby('_id.Year')[['Avg_Automation_Impact', 'SecondDS_Automation_Impact']].mean().reset_index()
fig5 = px.line(time_compare, x='_id.Year', y=['Avg_Automation_Impact', 'SecondDS_Automation_Impact'],
               title="Automation Impact Trend (Dataset1 vs Dataset2)",
               labels={'value': 'Impact Score', 'variable': 'Dataset'})
st.plotly_chart(fig5, use_container_width=True)

# ---------- Save Prediction ----------
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# ---------- Footer ----------
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

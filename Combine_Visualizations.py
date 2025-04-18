import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and datasets
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned.csv")

# Combine datasets
combined_df = pd.merge(
    df, df2,
    left_on=['_id.Country', '_id.Sector', '_id.Year', '_id.EducationLevel'],
    right_on=['Country', 'Sector', 'Year', 'Education_Level'],
    how='inner'
)

# Fill missing education values and convert numeric fields
combined_df['_id.EducationLevel'].fillna('Unknown', inplace=True)
combined_df['Avg_Automation_Impact'] = pd.to_numeric(combined_df['Avg_Automation_Impact'], errors='coerce')
combined_df['SecondDS_Automation_Impact'] = pd.to_numeric(combined_df['Automation_Impact_Level'], errors='coerce')

# Streamlit page setup
st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# User Inputs
st.markdown("### 🎯 Enter Parameters for Prediction")
col1, col2, col3, col4 = st.columns(4)
year_range = col1.slider("Select Year Range", 2010, 2022, (2010, 2022), step=1)
country = col2.selectbox("Country", sorted(df['_id.Country'].unique()))
sector = col3.selectbox("Sector", sorted(df['_id.Sector'].unique()))
education = col4.selectbox("Education Level", sorted(df['_id.EducationLevel'].unique()))

# Dynamic Title
st.markdown(f"### 🚀 AI Automation Impact from {year_range[0]} to {year_range[1]}")

# Prepare input for prediction
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year_range[0]],
    '_id.EducationLevel': [education],
})

# Encode and predict
X_train = df.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]

st.success(f"🔮 Predicted Automation Impact Score: **{prediction:.2f}**")

# Country Comparison
st.markdown("---")
st.header("🌍 Country Comparison")
cols = st.columns(2)
country1 = cols[0].selectbox("Select First Country", df['_id.Country'].unique(), key='country1')
country2 = cols[1].selectbox("Select Second Country", [c for c in df['_id.Country'].unique() if c != country1], key='country2')

compare_df = df[df['_id.Country'].isin([country1, country2])]
fig1 = px.bar(compare_df, x='_id.Sector', y='Avg_Automation_Impact', color='_id.Country',
              title=f'Automation Impact: {country1} vs {country2}', barmode='group', height=400)
st.plotly_chart(fig1, use_container_width=True)

# Unemployment Trend
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
unemp = df.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig2 = px.line(unemp, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
               title='Unemployment Impact (Pre-AI vs Post-AI) Over Years')
st.plotly_chart(fig2, use_container_width=True)

# Sector-wise Trend
st.markdown("---")
st.header("🏭 Sector-wise Unemployment Comparison")
sector_selected = st.selectbox("Select Sector", df['_id.Sector'].unique(), key='sector_analysis')
df_sec = df[df['_id.Sector'] == sector_selected].groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3 = px.bar(df_sec, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title=f'Unemployment Trend in {sector_selected}')
st.plotly_chart(fig3, use_container_width=True)

# Education Level Comparison
st.markdown("---")
st.header("🎓 Education Level Comparison")
edu_compare = combined_df.groupby('_id.EducationLevel')[['Avg_Automation_Impact', 'SecondDS_Automation_Impact']].mean().reset_index()
fig4 = px.bar(edu_compare, x='_id.EducationLevel', y=['Avg_Automation_Impact', 'SecondDS_Automation_Impact'],
              title="AI Automation Impact Based on Education Level", barmode='group', height=400)
st.plotly_chart(fig4, use_container_width=True)

# Save Prediction
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

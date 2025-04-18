import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load model and data
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Enter Parameters for Prediction")
col1, col2, col3, col4 = st.columns(4)
year = col1.selectbox("Year", sorted(df['_id.Year'].unique()))
country = col2.selectbox("Country", sorted(df['_id.Country'].unique()))
sector = col3.selectbox("Sector", sorted(df['_id.Sector'].unique()))
education = col4.selectbox("Education Level", sorted(df['_id.EducationLevel'].unique()))

col5, col6, col7 = st.columns(3)
pre_ai = col5.slider("Pre-AI Impact", 0, 100, 50)
post_ai = col6.slider("Post-AI Impact", 0, 100, 50)
automation_impact = col7.slider("Automation Impact", 0, 100, 50)

col8, col9, col10 = st.columns(3)
ai_roles = col8.slider("AI Role Jobs", 0, 100, 50)
reskill = col9.slider("Reskilling Programs", 0, 100, 50)
econ_impact = col10.slider("Economic Impact", 0, 100, 50)

sector_growth = st.slider("Sector Growth", 0, 100, 50)

# Prepare DataFrame
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year],
    '_id.EducationLevel': [education],
    'Avg_PreAI': [pre_ai],
    'Avg_PostAI': [post_ai],
    'Avg_Automation_Impact': [automation_impact],
    'Avg_AI_Role_Jobs': [ai_roles],
    'Avg_ReskillingPrograms': [reskill],
    'Avg_EconomicImpact': [econ_impact],
    'Avg_SectorGrowth': [sector_growth]
})

# Predict
X_train = df.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]

st.success(f"🔮 Predicted Automation Impact Score: **{prediction:.2f}**")

# ---------- File Upload ----------
st.markdown("---")
st.header("📂 Upload Your Own Dataset")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    st.write("Dataset Loaded Successfully!")
    st.write(new_data.head())  # Preview first few rows of the uploaded data
    df = new_data  # Re-load the data
else:
    st.info("Upload a CSV file to begin.")

# ---------- Comparison Section ----------
st.markdown("---")
st.header("🌍 Country Comparison")
cols = st.columns(2)
country1 = cols[0].selectbox("Select First Country", df['_id.Country'].unique(), key='country1')
country2 = cols[1].selectbox("Select Second Country", [c for c in df['_id.Country'].unique() if c != country1], key='country2')

compare_df = df[df['_id.Country'].isin([country1, country2])]
fig1 = px.bar(compare_df, x='_id.Sector', y='Avg_Automation_Impact', color='_id.Country',
              title=f'Automation Impact: {country1} vs {country2}',
              barmode='group', height=400)
st.plotly_chart(fig1, use_container_width=True)

# ---------- Unemployment Over Time ----------
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
unemp = df.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig2 = px.line(unemp, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
               title='Unemployment Impact (Pre-AI vs Post-AI) Over Years')
st.plotly_chart(fig2, use_container_width=True)

# ---------- Sector Trend ----------
st.markdown("---")
st.header("🏭 Sector-wise Unemployment Comparison")
sector_selected = st.selectbox("Select Sector", df['_id.Sector'].unique(), key='sector_analysis')
df_sec = df[df['_id.Sector'] == sector_selected].groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3 = px.bar(df_sec, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title=f'Unemployment Trend in {sector_selected}')
st.plotly_chart(fig3, use_container_width=True)

# ---------- Sector and Country Trend Comparisons ----------
st.markdown("---")
st.header("🔍 Compare Similar Sectors or Trends")

# Select Country for comparison
compare_country = st.selectbox("Select Country for Trend Analysis", df['_id.Country'].unique(), key="compare_country")

# Filter dataset for the selected country
country_data = df[df['_id.Country'] == compare_country]

# Optionally filter by sector, else show overall trends
sector_to_compare = st.selectbox("Select Sector for Comparison", ["All Sectors"] + sorted(df['_id.Sector'].unique()), key="sector_compare")

if sector_to_compare != "All Sectors":
    country_data = country_data[country_data['_id.Sector'] == sector_to_compare]

# Plot comparison of sector trends (e.g., PreAI vs PostAI)
sector_trends = country_data.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()

fig_sector_trends = px.line(sector_trends, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
                            labels={'value': 'Impact Score', 'variable': 'Impact Type'},
                            title=f'Sector Impact Over Time in {compare_country}' if sector_to_compare == "All Sectors" 
                            else f'{sector_to_compare} Impact Over Time in {compare_country}')
st.plotly_chart(fig_sector_trends, use_container_width=True)

# Compare more sectors
sector_comparison = df.groupby(['_id.Country', '_id.Sector', '_id.Year'])[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()

fig_comparison = px.line(sector_comparison, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
                         color='_id.Sector', line_group='_id.Country', markers=True,
                         title=f'Comparison of AI Automation Impact Across Sectors and Countries')
st.plotly_chart(fig_comparison, use_container_width=True)

# ---------- Export Prediction ----------
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

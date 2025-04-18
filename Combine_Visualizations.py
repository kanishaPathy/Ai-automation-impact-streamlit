import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and datasets
model = joblib.load("xgboost_model.pkl")
df_main = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")
df_reskill = pd.read_csv("reskilling_dataset_cleaned.csv")

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Select Parameters for Prediction")
col1, col2, col3, col4 = st.columns(4)
year_range = col1.slider("Select Year Range", int(df_main['_id.Year'].min()), int(df_main['_id.Year'].max()), (2010, 2022))
country = col2.selectbox("Country", sorted(df_main['_id.Country'].unique()))
sector = col3.selectbox("Sector", sorted(df_main['_id.Sector'].unique()))
education = col4.selectbox("Education Level", sorted(df_main['_id.EducationLevel'].unique()))

# Prepare input data for prediction
input_df = pd.DataFrame({
    '_id.Country': [country],
    '_id.Sector': [sector],
    '_id.Year': [year_range[0]],
    '_id.EducationLevel': [education],
})

# Encoding and prediction
X_train = df_main.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]
st.success(f"🔮 Predicted Automation Impact Score for {year_range[0]}: **{prediction:.2f}**")

# ---------- Visualization Section ----------
st.markdown("---")
st.header(f"🌍 Country Comparison from {year_range[0]} to {year_range[1]}")
col_c1, col_c2 = st.columns(2)
country1 = col_c1.selectbox("First Country", df_main['_id.Country'].unique(), key='country1')
country2 = col_c2.selectbox("Second Country", [c for c in df_main['_id.Country'].unique() if c != country1], key='country2')

compare_df = df_main[(df_main['_id.Country'].isin([country1, country2])) & 
                     (df_main['_id.Year'] >= year_range[0]) & (df_main['_id.Year'] <= year_range[1])]
fig1 = px.bar(compare_df, x='_id.Sector', y='Avg_Automation_Impact', color='_id.Country',
              title=f'Automation Impact: {country1} vs {country2} ({year_range[0]} - {year_range[1]})', barmode='group')
st.plotly_chart(fig1, use_container_width=True)

# ---------- Unemployment Over Time ----------
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
unemp = df_main[(df_main['_id.Year'] >= year_range[0]) & (df_main['_id.Year'] <= year_range[1])]
unemp = unemp.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig2 = px.line(unemp, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
               title=f'Unemployment Impact (Pre-AI vs Post-AI) from {year_range[0]} to {year_range[1]}')
st.plotly_chart(fig2, use_container_width=True)

# ---------- Sector-wise Trend ----------
st.markdown("---")
st.header("🏭 Sector-wise Unemployment Comparison")
sector_selected = st.selectbox("Select Sector", df_main['_id.Sector'].unique(), key='sector_analysis')
df_sec = df_main[(df_main['_id.Sector'] == sector_selected) & 
                 (df_main['_id.Year'] >= year_range[0]) & (df_main['_id.Year'] <= year_range[1])]
df_sec = df_sec.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3 = px.bar(df_sec, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'], barmode='group',
              title=f'Unemployment Trend in {sector_selected}')
st.plotly_chart(fig3, use_container_width=True)

# ---------- Education Level Impact ----------
st.markdown("---")
st.header("🎓 Education Level Impact")
edu_impact = df_main[(df_main['_id.Year'] >= year_range[0]) & (df_main['_id.Year'] <= year_range[1])]
edu_impact = edu_impact.groupby('_id.EducationLevel')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig4 = px.bar(edu_impact, x='_id.EducationLevel', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title='Education Level vs AI Impact')
st.plotly_chart(fig4, use_container_width=True)

# ---------- Country vs Sector ----------
st.markdown("---")
st.header("🌐 Country vs Sector Comparison")
col_cv1, col_cv2 = st.columns(2)
country_vs = col_cv1.selectbox("Select Country", df_main['_id.Country'].unique(), key='country_vs')
sector_vs = col_cv2.selectbox("Select Sector (or All)", ['All'] + list(df_main['_id.Sector'].unique()), key='sector_vs')

filter_df = df_main[(df_main['_id.Country'] == country_vs) & 
                    (df_main['_id.Year'] >= year_range[0]) & (df_main['_id.Year'] <= year_range[1])]
if sector_vs != 'All':
    filter_df = filter_df[filter_df['_id.Sector'] == sector_vs]

fig5 = px.bar(filter_df, x='_id.Sector', y='Avg_Automation_Impact', color='_id.Year',
              title=f'{country_vs} vs {"All Sectors" if sector_vs == "All" else sector_vs}')
st.plotly_chart(fig5, use_container_width=True)

# ---------- Gender Representation by Sector ----------
st.markdown("---")
st.header("🚻 Gender Representation Across Sectors")
gender_sector = df_reskill[df_reskill['Year'].between(year_range[0], year_range[1])]
gender_sector = gender_sector.groupby(['Sector', 'Gender_Distribution']).size().reset_index(name='Count')
fig6 = px.bar(gender_sector, x='Sector', y='Count', color='Gender_Distribution',
              title='Gender Representation by Sector', barmode='group')
st.plotly_chart(fig6, use_container_width=True)

# ---------- Skill Level Pre-AI vs Post-AI ----------
st.markdown("---")
st.header("💼 Skill Level Impact: Pre-AI vs Post-AI")
skill_df = df_reskill[df_reskill['Year'].between(year_range[0], year_range[1])]
skill_df['Avg_PreAI'] = skill_df['Automation_Impact_Level'] * 0.6  # Simulated example
skill_df['Avg_PostAI'] = skill_df['Automation_Impact_Level'] * 1.1
skill_chart = skill_df.groupby('Skill_Level')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig7 = px.bar(skill_chart, x='Skill_Level', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title="Skill Level vs AI Impact (Simulated)")
st.plotly_chart(fig7, use_container_width=True)

# ---------- Export Prediction Option ----------
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# ---------- Footer ----------
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

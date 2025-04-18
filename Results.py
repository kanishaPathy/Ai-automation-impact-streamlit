import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and data
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Select Parameters for Prediction")
col1, col2, col3, col4 = st.columns(4)
year_range = col1.slider("Year Range", int(df['_id.Year'].min()), int(df['_id.Year'].max()), (2010, 2022))
country = col2.selectbox("Country", sorted(df['_id.Country'].unique()))
sector = col3.selectbox("Sector", sorted(df['_id.Sector'].unique()))
education = col4.selectbox("Education Level", sorted(df['_id.EducationLevel'].unique()))

# Predict year-wise automation impact
st.markdown("---")
st.subheader("📈 Year-wise Automation Impact Prediction")
predictions = []
for y in range(year_range[0], year_range[1] + 1):
    input_df = pd.DataFrame({
        '_id.Country': [country],
        '_id.Sector': [sector],
        '_id.Year': [y],
        '_id.EducationLevel': [education],
    })
    X_train = df.drop(columns=['Avg_Automation_Impact'])
    X_encoded = pd.get_dummies(X_train)
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)
    pred = model.predict(input_encoded)[0]
    predictions.append((y, pred))

pred_df = pd.DataFrame(predictions, columns=['Year', 'Predicted_Impact'])
fig_pred = px.line(pred_df, x='Year', y='Predicted_Impact', title=f"Predicted Impact (2010–2022): {sector} - {education} in {country}")
st.plotly_chart(fig_pred, use_container_width=True)

# ---------- Education Impact Comparison ----------
st.markdown("---")
st.subheader("🎓 Education Level Impact (2022)")
edu_impacts = []
for edu in sorted(df['_id.EducationLevel'].unique()):
    input_df = pd.DataFrame({
        '_id.Country': [country],
        '_id.Sector': [sector],
        '_id.Year': [2022],
        '_id.EducationLevel': [edu],
    })
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)
    pred = model.predict(input_encoded)[0]
    edu_impacts.append((edu, pred))

e_df = pd.DataFrame(edu_impacts, columns=['Education Level', 'Predicted_Impact'])
fig_edu = px.bar(e_df, x='Education Level', y='Predicted_Impact', title=f"Education Level Comparison in {sector} - {country} (2022)")
st.plotly_chart(fig_edu, use_container_width=True)

# ---------- Country vs All Sectors ----------
st.markdown("---")
st.subheader("🏭 Country vs All Sectors (2022)")
sector_impacts = []
for sec in sorted(df['_id.Sector'].unique()):
    input_df = pd.DataFrame({
        '_id.Country': [country],
        '_id.Sector': [sec],
        '_id.Year': [2022],
        '_id.EducationLevel': [education],
    })
    input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)
    pred = model.predict(input_encoded)[0]
    sector_impacts.append((sec, pred))

s_df = pd.DataFrame(sector_impacts, columns=['Sector', 'Predicted_Impact'])
fig_sector_all = px.bar(s_df, x='Sector', y='Predicted_Impact', title=f"All Sector Comparison in {country} - {education} (2022)")
st.plotly_chart(fig_sector_all, use_container_width=True)

# ---------- Country Comparison Visualization ----------
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

# ---------- Unemployment Over Time Visualization ----------
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
unemp = df.groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig2 = px.line(unemp, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
               title='Unemployment Impact (Pre-AI vs Post-AI) Over Years')
st.plotly_chart(fig2, use_container_width=True)

# ---------- Sector-wise Trend Visualization ----------
st.markdown("---")
st.header("🏭 Sector-wise Unemployment Comparison")
sector_selected = st.selectbox("Select Sector", df['_id.Sector'].unique(), key='sector_analysis')
df_sec = df[df['_id.Sector'] == sector_selected].groupby('_id.Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3 = px.bar(df_sec, x='_id.Year', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title=f'Unemployment Trend in {sector_selected}')
st.plotly_chart(fig3, use_container_width=True)

# ---------- Export Prediction Option ----------
st.markdown("---")
if st.button("📂 Save Prediction to CSV"):
    latest_input = pd.DataFrame({
        '_id.Country': [country],
        '_id.Sector': [sector],
        '_id.Year': [year_range[1]],
        '_id.EducationLevel': [education],
        'Predicted_Automation_Impact': [predictions[-1][1]]
    })
    latest_input.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# ---------- Footer ----------
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

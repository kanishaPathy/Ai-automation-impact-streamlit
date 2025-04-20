import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and datasets
model = joblib.load("xgboost_model.pkl")

df1 = pd.read_csv("Unemployment_cleaned_df1.csv")
df2 = pd.read_csv("reskilling_dataset_cleaned_df2.csv")
df3 = pd.read_csv("Sector_cleaned_df3.csv")

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Select Parameters for Prediction")

# Layout for input columns
col1, col2, col3, col4 = st.columns(4)

# Sliders and dropdowns for selecting parameters
year_range = col1.slider("Select Year Range", int(df1['Year'].min()), int(df1['Year'].max()), (2010, 2022))
country = col2.selectbox("Select Country", sorted(df1['Country'].unique()))
sector = col3.selectbox("Select Sector", sorted(df1['Sector'].unique()))
education = col4.selectbox("Select Education Level", sorted(df1['EducationLevel'].unique()))

# Prepare input data for prediction (using the first year in the range)
input_df = pd.DataFrame({
    'Country': [country],
    'Sector': [sector],
    'Year': [year_range[0]],
    'EducationLevel': [education],
})

# ---------- Data Preprocessing ----------
# Merge all three datasets
merged = df1.merge(df2, on=["Country", "Sector", "Year", "EducationLevel"], how="left")
merged = merged.merge(df3, on=["Country", "Sector", "Year"], how="left")

# Encoding the input data for prediction
X_train = merged.drop(columns=['Avg_Automation_Impact'])  # Target variable
X_encoded = pd.get_dummies(X_train)
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

# Predict with the model
with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]

st.success(f"🔮 Predicted Automation Impact Score for {year_range[0]}: **{prediction:.2f}**")

# ---------- Visualizations Section ----------
st.markdown("---")
st.header(f"🌍 Country Comparison from {year_range[0]} to {year_range[1]}")

# Country comparison selection
cols = st.columns(2)
country1 = cols[0].selectbox("Select First Country", df1['Country'].unique(), key='country1')
country2 = cols[1].selectbox("Select Second Country", [c for c in df1['Country'].unique() if c != country1], key='country2')

# Filter and compare data for selected countries
compare_df = merged[(merged['Country'].isin([country1, country2])) & (merged['Year'] >= year_range[0]) & (merged['Year'] <= year_range[1])]
fig1 = px.bar(compare_df, x='Sector', y='Avg_Automation_Impact', color='Country',
              title=f'Automation Impact: {country1} vs {country2} ({year_range[0]} - {year_range[1]})', barmode='group', height=400)
st.plotly_chart(fig1, use_container_width=True)

# Unemployment trend visualization (PreAI vs PostAI)
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
unemp = merged[(merged['Year'] >= year_range[0]) & (merged['Year'] <= year_range[1])]
unemp = unemp.groupby('Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig2 = px.line(unemp, x='Year', y=['Avg_PreAI', 'Avg_PostAI'],
               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
               title=f'Unemployment Impact (Pre-AI vs Post-AI) from {year_range[0]} to {year_range[1]}')
st.plotly_chart(fig2, use_container_width=True)

# Sector-wise trend visualization
st.markdown("---")
st.header("🏭 Sector-wise Unemployment Comparison")
sector_selected = st.selectbox("Select Sector for Unemployment Trend", df1['Sector'].unique(), key='sector_analysis')
df_sec = merged[(merged['Sector'] == sector_selected) & (merged['Year'] >= year_range[0]) & (merged['Year'] <= year_range[1])]
df_sec = df_sec.groupby('Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3 = px.bar(df_sec, x='Year', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title=f'Unemployment Trend in {sector_selected} ({year_range[0]} - {year_range[1]})')
st.plotly_chart(fig3, use_container_width=True)

# Education Level Impact Visualization
st.markdown("---")
st.header("🎓 Education Level Impact")
edu_impact = merged[(merged['Year'] >= year_range[0]) & (merged['Year'] <= year_range[1])]
edu_impact = edu_impact.groupby('EducationLevel')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig4 = px.bar(edu_impact, x='EducationLevel', y=['Avg_PreAI', 'Avg_PostAI'],
              barmode='group', title='Education Level vs AI Impact')
st.plotly_chart(fig4, use_container_width=True)

# Export Prediction Option
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by [Your Name] | Powered by Streamlit + Plotly + XGBoost")

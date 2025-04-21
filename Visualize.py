import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and data
model = joblib.load("xgboost_model_ai_impact.pkl")
df = pd.read_csv("merged_data_updated.csv")

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- User Inputs Section ----------
st.markdown("### 🎯 Select Parameters for Prediction")
col1, col2, col3, col4 = st.columns(4)
year_range = col1.slider("Select Year Range", int(df['Year'].min()), int(df['Year'].max()), (2010, 2022))
country = col2.selectbox("Country", sorted(df['Country'].unique()))
sector = col3.selectbox("Sector", sorted(df['Sector'].unique()))
education = col4.selectbox("Education Level", sorted(df['EducationLevel'].unique()))

# Prepare input data for prediction (first year of selected range)
input_df = pd.DataFrame({
    'Country': [country],
    'Sector': [sector],
    'Year': [year_range[0]],
    'Education_Level': [education]
})

# Get the training data's feature names (encoded columns)
X_encoded = pd.get_dummies(df.drop(columns=['Avg_Automation_Impact']))  # Drop target column during encoding

# Reindex the input data to match the training data's feature set
input_encoded = pd.get_dummies(input_df).reindex(columns=X_encoded.columns, fill_value=0)

# Make the prediction
with st.spinner("Predicting Automation Impact..."):
    prediction = model.predict(input_encoded)[0]

st.success(f"🔮 Predicted Automation Impact Score for {year_range[0]}: **{prediction:.2f}**")

# ---------- Visualizations ----------
st.markdown("---")
st.header(f"🌍 Country Comparison ({year_range[0]} to {year_range[1]})")
col1, col2 = st.columns(2)
country1 = col1.selectbox("Country 1", sorted(df['Country'].unique()), key="c1")
country2 = col2.selectbox("Country 2", [c for c in sorted(df['Country'].unique()) if c != country1], key="c2")

compare_df = df[(df['Country'].isin([country1, country2])) & (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
fig1 = px.bar(compare_df, x='Sector', y='Avg_Automation_Impact', color='Country',
              barmode='group', title=f'{country1} vs {country2} Automation Impact')
st.plotly_chart(fig1, use_container_width=True)

# Unemployment Trends
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
trend_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
trend_df = trend_df.groupby('Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig2 = px.line(trend_df, x='Year', y=['Avg_PreAI', 'Avg_PostAI'], title='Pre-AI vs Post-AI Unemployment Trends')
st.plotly_chart(fig2, use_container_width=True)

# Sector-wise Unemployment
st.markdown("---")
st.header("🏭 Sector-wise Comparison")
sector_selected = st.selectbox("Select Sector", sorted(df['Sector'].unique()), key="sector_select")
sector_df = df[(df['Sector'] == sector_selected) & (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
sector_grouped = sector_df.groupby('Year')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig3 = px.bar(sector_grouped, x='Year', y=['Avg_PreAI', 'Avg_PostAI'], barmode='group', title=f'{sector_selected} Sector Trend')
st.plotly_chart(fig3, use_container_width=True)

# Education Impact
st.markdown("---")
st.header("🎓 Education Level Impact")
edu_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
edu_grouped = edu_df.groupby('Education_Level')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
fig4 = px.bar(edu_grouped, x='Education_Level', y=['Avg_PreAI', 'Avg_PostAI'], barmode='group', title='Education Level Impact')
st.plotly_chart(fig4, use_container_width=True)

# Country vs All Sectors
st.markdown("---")
st.header("🌐 Country-Sector Automation Impact")
col1, col2 = st.columns(2)
country_vs = col1.selectbox("Country", sorted(df['Country'].unique()), key="country_vs")
sector_vs = col2.selectbox("Compare With Sector (or All)", ['All'] + sorted(df['Sector'].unique()), key="sector_vs")

filter_df = df[(df['Country'] == country_vs) & (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
if sector_vs != 'All':
    filter_df = filter_df[filter_df['Sector'] == sector_vs]

fig5 = px.bar(filter_df, x='Sector', y='Avg_Automation_Impact', color='Year',
              title=f'{country_vs} Sector-wise Automation Impact')
st.plotly_chart(fig5, use_container_width=True)

# Export Prediction
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved as `saved_prediction.csv`")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

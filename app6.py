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


# ---------- Education Level Impact Section ----------
# st.markdown("---")
# st.header("🎓 Impact by Education Level")

# # Allow multiple education levels to be selected and show comparison
# education_levels = ['High School', 'Associate', 'Bachelor', 'Masters', 'PhD']
# selected_education_levels = st.multiselect("Select Education Levels for Comparison", education_levels, default=education_levels)

# edu_df = df[df['_id.EducationLevel'].isin(selected_education_levels)]
# edu_avg = edu_df.groupby('_id.EducationLevel')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()

# # Plotting Education Level vs AI Impact
# fig4 = px.bar(edu_avg, y='_id.EducationLevel', x=['Avg_PreAI', 'Avg_PostAI'],
#               orientation='h', barmode='group', title='Education Level vs AI Impact')
# st.plotly_chart(fig4, use_container_width=True)

# Calculate Percentiles (25th, 50th, 75th)
edu_percentiles = edu_df.groupby('_id.EducationLevel')['Avg_Automation_Impact'].quantile([0.25, 0.5, 0.75]).unstack()
edu_percentiles.reset_index(inplace=True)
edu_percentiles.columns = ['Education Level', '25th Percentile', '50th Percentile (Median)', '75th Percentile']

# Create an interactive bar chart with color scale to represent automation impact
# fig6 = px.bar(edu_percentiles, 
#               x='Education Level', 
#               y=['25th Percentile', '50th Percentile (Median)', '75th Percentile'],
#               title="Percentile Distribution of Automation Impact by Education Level",
#               labels={'value': 'Automation Impact', 'Education Level': 'Education Level'},
#               color_discrete_sequence=px.colors.sequential.RdBu)
# st.plotly_chart(fig6, use_container_width=True)
 # Additional Visualization: Education Level vs Automation Impact
# edu_impact_df = edu_df.groupby('_id.EducationLevel')[['Avg_Automation_Impact']].mean().reset_index()
# fig5 = px.bar(edu_impact_df, y='_id.EducationLevel', x='Avg_Automation_Impact',
#               orientation='h', title='Average Automation Impact by Education Level')
# st.plotly_chart(fig5, use_container_width=True)

# # Stacked Bar: Pre-AI vs Post-AI Impact by Education Level
# edu_impact_df = edu_df.groupby('_id.EducationLevel')[['Avg_PreAI', 'Avg_PostAI']].mean().reset_index()
# fig7 = px.bar(edu_impact_df, 
#               x='_id.EducationLevel', 
#               y=['Avg_PreAI', 'Avg_PostAI'],
#               title="Pre-AI vs Post-AI Impact by Education Level",
#               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
#               barmode='stack', 
#               color='variable')
# st.plotly_chart(fig7, use_container_width=True)
# ---------- Education Level Insights ----------
# ---------- Education Level Insights ----------
st.markdown("---")
st.header("🎓 Impact by Education Level")

# Prepare the education-level DataFrame
edu_df = df[['_id.EducationLevel', 'Avg_Automation_Impact']]  # Now properly defined

# Filter Option
all_levels = sorted(edu_df['_id.EducationLevel'].unique())
selected_levels = st.multiselect("🔍 Select Education Levels to Display", all_levels, default=all_levels)
filtered_df = edu_df[edu_df['_id.EducationLevel'].isin(selected_levels)]

# Create Tabs for views
tab1, tab2 = st.tabs(["📊 Percentile Distribution", "📈 Average Impact"])

with tab1:
    # Percentile Calculation
    edu_percentiles = filtered_df.groupby('_id.EducationLevel')['Avg_Automation_Impact'].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    edu_percentiles.columns = ['Education Level', '25th Percentile', '50th Percentile (Median)', '75th Percentile']

    # Plot
    fig6 = px.bar(edu_percentiles, 
                  x='Education Level', 
                  y=['25th Percentile', '50th Percentile (Median)', '75th Percentile'],
                  barmode='group',
                  title="📊 Percentile Distribution of Automation Impact by Education Level",
                  labels={'value': 'Automation Impact', 'Education Level': 'Education Level'},
                  color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig6, use_container_width=True)

with tab2:
    # Average Impact Calculation
    edu_avg_impact = filtered_df.groupby('_id.EducationLevel')[['Avg_Automation_Impact']].mean().reset_index()

    # Plot
    fig5 = px.bar(edu_avg_impact, 
                  y='_id.EducationLevel', 
                  x='Avg_Automation_Impact',
                  orientation='h',
                  title='📈 Average Automation Impact by Education Level',
                  color='Avg_Automation_Impact',
                  color_continuous_scale='Plasma',
                  hover_data={'Avg_Automation_Impact': ':.2f'})
    st.plotly_chart(fig5, use_container_width=True)

# Optional: Show summary table
with st.expander("📋 Show Data Table"):
    st.dataframe(edu_avg_impact.style.format({'Avg_Automation_Impact': '{:.2f}'}))

# ---------- Export Prediction ----------
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

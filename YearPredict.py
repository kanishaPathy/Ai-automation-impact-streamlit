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
st.markdown("---")
st.header("🎓 Education Level vs Automation Impact")

# ✅ Step 1: Check if required columns exist in the main dataframe
required_columns = ['_id.EducationLevel', 'Avg_Automation_Impact']
if all(col in df.columns for col in required_columns):

    # ✅ Step 2: Filter necessary columns and drop missing values
    edu_df = df[['_id.EducationLevel', 'Avg_Automation_Impact']].dropna()

    if not edu_df.empty:
        # ✅ Step 3: Group and calculate percentiles
        try:
            edu_percentiles = edu_df.groupby('_id.EducationLevel')['Avg_Automation_Impact'] \
                                     .quantile([0.25, 0.5, 0.75]) \
                                     .unstack().reset_index()

            edu_percentiles.columns = ['Education Level', '25th Percentile', '50th Percentile (Median)', '75th Percentile']

            # ✅ Step 4: Percentile Distribution Plot
            fig_percentiles = px.bar(edu_percentiles,
                                     x='Education Level',
                                     y=['25th Percentile', '50th Percentile (Median)', '75th Percentile'],
                                     title="Percentile Distribution of Automation Impact by Education Level",
                                     labels={'value': 'Automation Impact'},
                                     barmode='group',
                                     color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig_percentiles, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error while calculating percentiles: {e}")

        # ✅ Step 5: Average Impact Horizontal Bar Chart
        edu_avg = edu_df.groupby('_id.EducationLevel')['Avg_Automation_Impact'].mean().reset_index()

        fig_avg = px.bar(edu_avg,
                         y='_id.EducationLevel',
                         x='Avg_Automation_Impact',
                         orientation='h',
                         title="Average Automation Impact by Education Level",
                         color='Avg_Automation_Impact',
                         color_continuous_scale='Viridis')
        st.plotly_chart(fig_avg, use_container_width=True)

    else:
        st.warning("⚠️ No data available after removing missing values.")
else:
    st.error("❗ Required columns '_id.EducationLevel' or 'Avg_Automation_Impact' not found in the dataset.")

# ---------- Export Prediction ----------
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and data
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

# Streamlit page configuration
st.set_page_config(page_title="AI & Automation Impact Predictor", layout="wide")
st.title("🤖 AI & Automation Impact Predictor")

# Tab layout for a clean separation
tab1, tab2 = st.tabs(["📊 Input Parameters", "📈 Prediction Results"])

with tab1:
    st.subheader("Select Parameters")

    # Form for input parameters
    with st.form(key='input_form'):
        # Column layout for neat input sections
        col1, col2 = st.columns(2)

        with col1:
            selected_years = st.multiselect("📅 Select Year(s)", sorted(df['Year'].unique()), default=[df['Year'].max()])
            selected_countries = st.multiselect("🌍 Select Country(ies)", sorted(df['Country'].unique()), default=["Ireland", "USA"])
            selected_sector = st.selectbox("🏢 Select Sector", sorted(df['Sector'].unique()))
        
        with col2:
            selected_edu = st.selectbox("🎓 Select Education Level", sorted(df['Education Level'].unique()))
            pre_ai = st.slider("Pre-AI Impact (%)", 0, 100, 50)
            post_ai = st.slider("Post-AI Impact (%)", 0, 100, 50)
            automation_impact = st.slider("Automation Impact (%)", 0, 100, 50)
            ai_role_jobs = st.slider("AI Role Jobs (%)", 0, 100, 50)
            reskilling_programs = st.slider("Reskilling Programs (%)", 0, 100, 50)
            economic_impact = st.slider("Economic Impact (%)", 0, 100, 50)

        submit_button = st.form_submit_button(label="🔍 Run Prediction")

    # Handling prediction after form submission
    if submit_button:
        with tab2:
            st.subheader("📈 Prediction Result")

            # Create input dataframe for prediction
            input_data = {
                "Year": selected_years[-1],
                "Country": selected_countries[0],
                "Sector": selected_sector,
                "Education Level": selected_edu,
                "PreAI": pre_ai,
                "PostAI": post_ai,
                "Automation_Impact": automation_impact,
                "AI_Role_Jobs": ai_role_jobs,
                "Reskilling_Programs": reskilling_programs,
                "Economic_Impact": economic_impact
            }

            input_df = pd.DataFrame([input_data])

            # Make prediction
            prediction = model.predict(input_df)[0]

            # Display prediction result
            st.metric(label="Predicted Automation Impact Level", value=f"{round(prediction, 2)}%")

            # Show actual data in comparison
            filtered_df = df[(df['Year'].isin(selected_years)) & (df['Country'].isin(selected_countries))]

            fig = px.bar(filtered_df, x='Country', y='Automation_Impact', color='Year',
                         barmode='group', title='Actual Automation Impact by Country and Year')
            st.plotly_chart(fig, use_container_width=True)

# ---------- Country Comparison Section ----------
st.markdown("---")
st.header("🌍 Country Comparison")

cols = st.columns(2)
country1 = cols[0].selectbox("Select First Country", df['Country'].unique(), key='country1')
country2 = cols[1].selectbox("Select Second Country", [c for c in df['Country'].unique() if c != country1], key='country2')

compare_df = df[df['Country'].isin([country1, country2])]
fig1 = px.bar(compare_df, x='Sector', y='Automation_Impact', color='Country',
              title=f'Automation Impact: {country1} vs {country2}',
              barmode='group', height=400)
st.plotly_chart(fig1, use_container_width=True)

# ---------- Unemployment Trend Over Time ----------
st.markdown("---")
st.header("📈 Unemployment Trend Over Time")
unemp = df.groupby('Year')[['PreAI', 'PostAI']].mean().reset_index()
fig2 = px.line(unemp, x='Year', y=['PreAI', 'PostAI'],
               labels={'value': 'Impact Score', 'variable': 'Impact Type'},
               title='Unemployment Impact (Pre-AI vs Post-AI) Over Years')
st.plotly_chart(fig2, use_container_width=True)

# ---------- Sector Trend ----------
st.markdown("---")
st.header("🏭 Sector-wise Unemployment Comparison")
sector_selected = st.selectbox("Select Sector", df['Sector'].unique(), key='sector_analysis')
df_sec = df[df['Sector'] == sector_selected].groupby('Year')[['PreAI', 'PostAI']].mean().reset_index()
fig3 = px.bar(df_sec, x='Year', y=['PreAI', 'PostAI'],
              barmode='group', title=f'Unemployment Trend in {sector_selected}')
st.plotly_chart(fig3, use_container_width=True)

# ---------- Education Level Insights ----------
st.markdown("---")
st.header("🎓 Education Level vs Automation Impact")

# Check if necessary columns are available
if 'Education Level' in df.columns and 'Automation_Impact' in df.columns:
    edu_df = df[['Education Level', 'Automation_Impact']].dropna()

    # Calculate percentiles
    edu_percentiles = edu_df.groupby('Education Level')['Automation_Impact'].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    edu_percentiles.columns = ['Education Level', '25th Percentile', '50th Percentile (Median)', '75th Percentile']

    # Percentile Distribution Plot
    fig_percentiles = px.bar(edu_percentiles,
                             x='Education Level',
                             y=['25th Percentile', '50th Percentile (Median)', '75th Percentile'],
                             title="Percentile Distribution of Automation Impact by Education Level",
                             labels={'value': 'Automation Impact'},
                             barmode='group',
                             color_discrete_sequence=px.colors.sequential.Plasma)
    st.plotly_chart(fig_percentiles, use_container_width=True)

    # Average Impact Bar Chart
    edu_avg = edu_df.groupby('Education Level')['Automation_Impact'].mean().reset_index()
    fig_avg = px.bar(edu_avg,
                     y='Education Level',
                     x='Automation_Impact',
                     orientation='h',
                     title="Average Automation Impact by Education Level",
                     color='Automation_Impact',
                     color_continuous_scale='Viridis')
    st.plotly_chart(fig_avg, use_container_width=True)

# ---------- Save Prediction ----------
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

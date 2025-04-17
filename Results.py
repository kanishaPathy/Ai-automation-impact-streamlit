import streamlit as st
import pandas as pd
import plotly.express as px

# Dummy dataset for dropdown population (replace with your actual DataFrame)
df = pd.read_csv("your_data.csv")

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI & Automation Impact Predictor")

# --- TAB Layout (for cleaner separation if needed)
tab1, tab2 = st.tabs(["📊 Input Parameters", "📈 Prediction Results"])

with tab1:
    st.subheader("Select Parameters")

    # Create columns for neat layout
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

    st.markdown("---")

    # Optional: Summary card style text
    st.info(f"📝 Analyzing impact for **{selected_sector}** sector in **{', '.join(selected_countries)}** for year(s) **{', '.join(map(str, selected_years))}** with education level **{selected_edu}**.")

    if st.button("🔍 Run Prediction"):
        with tab2:
            st.subheader("📈 Prediction Result")

            # Create input dataframe
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

            # --- Your model prediction (replace with actual model)
            prediction = model.predict(input_df)[0]

            # --- Display result as metric or styled text
            st.metric(label="Predicted Automation Impact Level", value=f"{round(prediction, 2)}%")

            # Optional: Visualization of similar real data
            filtered_df = df[(df['Year'].isin(selected_years)) & (df['Country'].isin(selected_countries))]

            fig = px.bar(filtered_df, x='Country', y='Automation_Impact', color='Year',
                         barmode='group', title='Actual Automation Impact by Country and Year')
            st.plotly_chart(fig, use_container_width=True)

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
# edu_percentiles = edu_df.groupby('_id.EducationLevel')['Avg_Automation_Impact'].quantile([0.25, 0.5, 0.75]).unstack()
# edu_percentiles.reset_index(inplace=True)
# edu_percentiles.columns = ['Education Level', '25th Percentile', '50th Percentile (Median)', '75th Percentile']

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
# ---------- Education Level Insights ----------st.markdown("---")
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

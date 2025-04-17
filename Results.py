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

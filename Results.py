import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load model and data
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to categorical columns in the original DataFrame
df['_id.Country'] = label_encoder.fit_transform(df['_id.Country'])
df['_id.Sector'] = label_encoder.fit_transform(df['_id.Sector'])
df['_id.EducationLevel'] = label_encoder.fit_transform(df['_id.EducationLevel'])

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI & Automation Impact Predictor")

# --- TAB Layout (for cleaner separation if needed)
tab1, tab2 = st.tabs(["📊 Input Parameters", "📈 Prediction Results"])

with tab1:
    st.subheader("Select Parameters")

    # Create columns for neat layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        year = st.selectbox("Year", sorted(df['_id.Year'].unique()))
        
    with col2:
        country = st.selectbox("Country", sorted(df['_id.Country'].unique()))
        
    with col3:
        sector = st.selectbox("Sector", sorted(df['_id.Sector'].unique()))
        
    with col4:
        education = st.selectbox("Education Level", sorted(df['_id.EducationLevel'].unique()))

    # Sliders for additional parameters
    pre_ai = st.slider("Pre-AI Impact (%)", 0, 100, 50)
    post_ai = st.slider("Post-AI Impact (%)", 0, 100, 50)
    automation_impact = st.slider("Automation Impact (%)", 0, 100, 50)
    ai_role_jobs = st.slider("AI Role Jobs (%)", 0, 100, 50)
    reskilling_programs = st.slider("Reskilling Programs (%)", 0, 100, 50)
    economic_impact = st.slider("Economic Impact (%)", 0, 100, 50)

    st.markdown("---")

    # Optional: Summary card style text
    st.info(f"📝 Analyzing impact for **{sector}** sector in **{country}** for year **{year}** with education level **{education}**.")

    if st.button("🔍 Run Prediction"):
        with tab2:
            st.subheader("📈 Prediction Result")

            # Create input dataframe (make sure to encode categorical values)
            input_data = {
                "_id.Year": year,
                "_id.Country": label_encoder.transform([country])[0],  # Encode country
                "_id.Sector": label_encoder.transform([sector])[0],  # Encode sector
                "_id.EducationLevel": label_encoder.transform([education])[0],  # Encode education level
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
            filtered_df = df[(df['_id.Year'] == year) & (df['_id.Country'] == label_encoder.transform([country])[0])]

            fig = px.bar(filtered_df, x='_id.Country', y='Automation_Impact', color='_id.Year',
                         barmode='group', title='Actual Automation Impact by Country and Year')
            st.plotly_chart(fig, use_container_width=True)

# ---------- Comparison Section ----------
st.markdown("---")
st.header("🌍 Country Comparison")
cols = st.columns(2)
country1 = cols[0].selectbox("Select First Country", df['_id.Country'].unique(), key='country1')
country2 = cols[1].selectbox("Select Second Country", [c for c in df['_id.Country'].unique() if c != country1], key='country2')

compare_df = df[df['_id.Country'].isin([country1, country2])]
fig1 = px.bar(compare_df, x='_id.Sector', y='Automation_Impact', color='_id.Country',
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

# ---------- Export Prediction ----------
st.markdown("---")
if st.button("💾 Save Prediction to CSV"):
    input_df['Predicted_Automation_Impact'] = prediction
    input_df.to_csv("saved_prediction.csv", index=False)
    st.success("📁 Prediction saved to **saved_prediction.csv**")

# Footer
st.markdown("---")
st.caption("📊 Built with ❤️ by Kanisha Pathy | Powered by Streamlit + Plotly + XGBoost")

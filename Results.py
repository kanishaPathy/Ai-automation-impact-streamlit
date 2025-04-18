import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

# Load model and data
model = joblib.load("xgboost_model.pkl")
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

st.set_page_config(page_title="AI Automation Impact", layout="wide")
st.title("🤖 AI Automation Impact Prediction & Insights")

# ---------- Selection Inputs ----------
st.markdown("### 🎯 Select Parameters")
col1, col2, col3 = st.columns(3)
country = col1.selectbox("Country", sorted(df['_id.Country'].unique()))
sector = col2.selectbox("Sector", sorted(df['_id.Sector'].unique()))
education = col3.selectbox("Education Level", sorted(df['_id.EducationLevel'].unique()))

# ---------- Predict Impact Over Years ----------
st.markdown("### ⏳ Automation Impact Over Years (2010–2022)")
year_range = list(range(2010, 2023))
multi_year_df = pd.DataFrame({
    '_id.Year': year_range,
    '_id.Country': country,
    '_id.Sector': sector,
    '_id.EducationLevel': education
})

# Prepare for prediction
X_train = df.drop(columns=['Avg_Automation_Impact'])
X_encoded = pd.get_dummies(X_train)
multi_encoded = pd.get_dummies(multi_year_df).reindex(columns=X_encoded.columns, fill_value=0)

multi_year_df['Predicted_Impact'] = model.predict(multi_encoded)

# Plot predictions over years
fig_time = px.line(multi_year_df, x="_id.Year", y="Predicted_Impact", markers=True,
                   title=f"📉 Predicted Automation Impact (2010–2022) for {sector}, {education} in {country}")
st.plotly_chart(fig_time, use_container_width=True)

# ---------- Education Level Impact ----------
st.markdown("### 📘 Education Level Impact Comparison")
edu_levels = df['_id.EducationLevel'].unique()
edu_df = pd.DataFrame({
    '_id.Year': [2022] * len(edu_levels),
    '_id.Country': [country] * len(edu_levels),
    '_id.Sector': [sector] * len(edu_levels),
    '_id.EducationLevel': edu_levels
})

edu_encoded = pd.get_dummies(edu_df).reindex(columns=X_encoded.columns, fill_value=0)
edu_df['Predicted_Impact'] = model.predict(edu_encoded)

fig_edu = px.bar(edu_df, x='_id.EducationLevel', y='Predicted_Impact', color='_id.EducationLevel',
                 title=f"🎓 Predicted Automation Impact by Education Level – {country}, {sector}, 2022")
st.plotly_chart(fig_edu, use_container_width=True)

# ---------- Country vs All Sectors ----------
st.markdown("### 🌍 Country vs All Sectors")
all_sectors = df['_id.Sector'].unique()
sector_df = pd.DataFrame({
    '_id.Year': [2022] * len(all_sectors),
    '_id.Country': [country] * len(all_sectors),
    '_id.Sector': all_sectors,
    '_id.EducationLevel': [education] * len(all_sectors)
})

sector_encoded = pd.get_dummies(sector_df).reindex(columns=X_encoded.columns, fill_value=0)
sector_df['Predicted_Impact'] = model.predict(sector_encoded)

fig_sector = px.bar(sector_df, x='_id.Sector', y='Predicted_Impact', color='_id.Sector',
                    title=f"🏭 Automation Impact by Sector – {country}, {education}, 2022")
st.plotly_chart(fig_sector, use_container_width=True)

# ---------- Save Year-wise Predictions ----------
st.markdown("---")
if st.button("💾 Save Year-wise Predictions to CSV"):
    multi_year_df.to_csv("yearwise_predictions.csv", index=False)
    st.success("📁 Saved to **yearwise_predictions.csv**")

# ---------- Footer ----------
st.markdown("---")
st.caption("📊 Built by Kanisha Pathy | AI Automation Impact Dashboard | Streamlit + Plotly")

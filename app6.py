import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("xgboost_model.pkl")

# Load data (make sure this file is in the same folder as the app)
df = pd.read_csv("Unemployment_jobcreation_db.Unemployment_data.csv")

st.title("AI & Automation Impact Prediction")
st.markdown("Predict the impact of automation/AI based on key factors")

# Interactive form with dynamic input
with st.form("input_form"):
    country = st.selectbox("Select Country", df['Country'].unique())
    sector = st.selectbox("Select Sector", df['Sector'].unique())
    year = st.slider("Select Year", int(df['Year'].min()), int(df['Year'].max()))
    education = st.selectbox("Select Education Level", df['EducationLevel'].unique())
    ai_adoption = st.slider("AI Adoption Rate (%)", 0, 100, step=5)
    automation_impact = st.slider("Automation Impact Score", 0.0, 10.0, step=0.1)
    submitted = st.form_submit_button("Predict Impact")

if submitted:
    input_df = pd.DataFrame({
        'Country': [country],
        'Sector': [sector],
        'Year': [year],
        'EducationLevel': [education],
        'AI_Adoption_Rate': [ai_adoption],
        'Automation_Impact': [automation_impact]
    })

    # Optional: preprocess input_df if needed (encoding, scaling, etc.)

    prediction = model.predict(input_df)
    st.success(f"Predicted AI/Automation Impact Score: {prediction[0]:.2f}")

    # Show a bar chart of prediction
    st.subheader("Prediction Visual")
    fig, ax = plt.subplots()
    ax.bar(["Predicted Impact"], [prediction[0]], color='orange')
    ax.set_ylabel("Impact Score")
    st.pyplot(fig)

    # Feature importance
    if st.button("Show Feature Importance"):
        importances = model.feature_importances_
        columns = ['AI_Adoption_Rate', 'Automation_Impact']  # Match model training features
        importance_df = pd.DataFrame({"Feature": columns, "Importance": importances[:len(columns)]})
        st.bar_chart(importance_df.set_index("Feature"))

# Optional comparison toggle
if st.checkbox("Compare with Another Country"):
    country2 = st.selectbox("Select Another Country", df['Country'].unique(), key='country2')
    compare_data = df[df['Country'] == country2].groupby("Year")["Automation_Impact"].mean()
    st.line_chart(compare_data)
